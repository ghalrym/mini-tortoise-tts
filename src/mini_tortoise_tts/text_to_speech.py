import logging
import os
import random
from time import time

import psutil
import torch
from torch import Tensor
from torch.nn import functional
from torch.types import Device

from mini_tortoise_tts.config import Config
from mini_tortoise_tts.external import get_model_path
from mini_tortoise_tts.load_voices import Voice
from mini_tortoise_tts.torch_tokenizers import VoiceBpeTokenizer
from mini_tortoise_tts.torch_modules import TorchMelSpectrogram, UnifiedVoice, HifiganGenerator

logger = logging.getLogger(__name__)


UNIFIED_VOICE = UnifiedVoice(
    max_mel_tokens=604,
    max_text_tokens=402,
    max_conditioning_inputs=2,
    layers=30,
    model_dim=1024,
    heads=16,
    number_text_tokens=255,
    start_text_token=255,
    checkpointing=False,
    train_solo_embeddings=False,
)
HIFIGAN = HifiganGenerator(
    in_channels=1024,
    out_channels=1,
    resblock_type="1",
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    resblock_kernel_sizes=[3, 7, 11],
    up_sample_kernel_sizes=[16, 16, 4, 4],
    up_sample_initial_channel=512,
    up_sample_factors=[8, 8, 2, 2],
    cond_channels=1024,
)
BYTES_IN_GB = 1024**3


class ToMuchText(Exception):

    def __init__(self):
        super().__init__("Text is too long to create speech.")


def pick_best_batch_size_for_gpu(device: Device) -> int:
    """Tries to pick a batch size that will fit in your GPU. These sizes aren't guaranteed to work,
    but they should give you a good shot."""
    if device.type == "cuda":
        _, available = torch.cuda.mem_get_info()
        available_gb = available / BYTES_IN_GB
        if available_gb > 14:
            return 16
        elif available_gb > 10:
            return 8
        elif available_gb > 7:
            return 4
    if device.type == "mps":
        available = psutil.virtual_memory().total
        available_gb = available / BYTES_IN_GB
        if available_gb > 14:
            return 16
        elif available_gb > 10:
            return 8
        elif available_gb > 7:
            return 4
    return 1


def format_conditioning(clip: Tensor, device: Device, cond_length: int = 132300) -> Tensor:
    """Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models."""
    gap = clip.shape[-1] - cond_length
    if gap < 0:
        clip = functional.pad(clip, pad=(0, abs(gap)))
    elif gap > 0:
        rand_start = random.randint(0, gap)
        clip = clip[:, rand_start : rand_start + cond_length]
    mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
    return mel_clip.unsqueeze(0).to(device)


def set_seed(seed: int | None = None):
    seed = int(time()) if seed is None else seed
    torch.manual_seed(seed)
    random.seed(seed)
    return seed


class TextToSpeech:
    __slots__ = (
        "_voice",
        "models_dir",
        "_preset",
        "tokenizer",
        "autoregressive_batch_size",
        "device",
        "aligner",
        "half",
        "autoregressive",
        "hifi_decoder",
        "rlg_auto",
    )

    def __init__(
        self,
        voice: Voice,
        autoregressive_batch_size=None,
        models_dir: str = Config.default_models_directory,
        kv_cache: bool = False,
        use_deepspeed: bool = False,
        half: bool = False,
        device: str | None = "cuda" if torch.cuda.is_available() else "cpu",
        tokenizer_vocab_file: str | None = None,
        tokenizer_basic=False,
        preset: str = "fast",
        seed: int | None = None,
    ):
        self._voice = voice
        self._preset = {
            "temperature": 0.8,
            "length_penalty": 1.0,
            "repetition_penalty": 2.0,
            "top_p": 0.8,
            # "cond_free_k": 2.0,
            # "diffusion_temperature": 1.0,
            "num_autoregressive_samples": 16,
            **Config.get_model_presets().get(preset, "fast"),
        }

        self.device = torch.device(device)
        self.models_dir = models_dir
        self.tokenizer = VoiceBpeTokenizer(tokenizer_vocab_file or Config.default_tokenizer_file(), tokenizer_basic)
        self.autoregressive_batch_size = autoregressive_batch_size or pick_best_batch_size_for_gpu(self.device)
        self.half = half

        if os.path.exists(f"{models_dir}/autoregressive.ptt"):
            self.autoregressive = torch.jit.load(f"{models_dir}/autoregressive.ptt")
        else:
            autoregressive_model_path = get_model_path("autoregressive.pth", models_dir)
            self.autoregressive = UNIFIED_VOICE.to(self.device).eval()
            self.autoregressive.load_state_dict(torch.load(autoregressive_model_path, weights_only=True), strict=False)
            self.autoregressive.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=self.half)

        self.hifi_decoder = HIFIGAN.to(self.device).eval()
        self.hifi_decoder.load_state_dict(
            torch.load(get_model_path("hifidecoder.pth"), weights_only=False), strict=False
        )
        self.rlg_auto = None  # Random latent generators (RLGs) are loaded lazily.
        set_seed(seed=seed)

    def get_conditioning_latents(self, voice_samples):
        with torch.no_grad():
            voice_samples: list[Tensor] = [v.to(self.device) for v in voice_samples]
            auto_conds: list[Tensor] = [format_conditioning(vs, device=self.device) for vs in voice_samples]
            auto_latent = self.autoregressive.get_conditioning(torch.stack(auto_conds, dim=1))

        return auto_latent

    def tts(
        self,
        text: str,
        voice_samples: list[Tensor] = None,
        k: int = 1,
        # Autoregressive generation parameters follow
        temperature: float = 0.8,
        length_penalty: float = 1.0,
        repetition_penalty: float = 2.0,
        top_p: float = 0.8,
        # CVVP parameters follow
        **hf_generate_kwargs,
    ) -> Tensor:
        encoded_text = self.tokenizer.encode(text)
        text_tokens: Tensor = torch.IntTensor(encoded_text).unsqueeze(0).to(self.device)
        text_tokens = functional.pad(text_tokens, (0, 1))
        auto_conditioning = self.get_conditioning_latents(voice_samples).to(self.device)

        logger.info("Generating autoregressive samples..")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.half):
                codes = self.autoregressive.inference_speech(
                    auto_conditioning,
                    text_tokens,
                    top_k=50,
                    top_p=top_p,
                    temperature=temperature,
                    do_sample=True,
                    num_beams=1,
                    num_return_sequences=1,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    output_hidden_states=True,
                    **hf_generate_kwargs,
                )
                gpt_latents = self.autoregressive(
                    auto_conditioning.repeat(k, 1),
                    text_tokens.repeat(k, 1),
                    torch.tensor([text_tokens.shape[-1]], device=text_tokens.device),
                    codes,
                    torch.tensor(
                        [codes.shape[-1] * self.autoregressive.mel_length_compression], device=text_tokens.device
                    ),
                    return_latent=True,
                    clip_inputs=False,
                )
            logger.info("generating audio..")
        return self.hifi_decoder.inference(gpt_latents.to(self.device), auto_conditioning)

    def generate(self, text: str) -> Tensor:
        return self.tts(text, voice_samples=self._voice.torch_state, **self._preset)
