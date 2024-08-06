import os

import librosa
import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import read
from torch import Tensor

from mini_tortoise_tts.config import Config


class MissingVoiceException(Exception):
    def __init__(self, voice: str, voice_dir: str):
        super().__init__(f"Missing voice : {voice} ({voice_dir})")


class UnsupportedAudioFormat(Exception):

    def __init__(self, audio_format: str):
        super().__init__(f"Unsupported audio format provided: {audio_format}")


def _load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    if data.dtype == np.int32:
        norm_fix = 2 ** 31
    elif data.dtype == np.int16:
        norm_fix = 2 ** 15
    elif data.dtype == np.float16 or data.dtype == np.float32:
        norm_fix = 1.0
    else:
        raise NotImplemented(f"Provided data dtype not supported: {data.dtype}")
    return torch.FloatTensor(data.astype(np.float32)) / norm_fix, sampling_rate


def _load_audio(audio_path: str, sampling_rate) -> Tensor:
    extension = os.path.splitext(audio_path)[1].casefold()
    if extension == ".wav":
        audio, lsr = _load_wav_to_torch(audio_path)
    elif extension == ".mp3":
        audio, lsr = librosa.load(audio_path, sr=sampling_rate)
        audio = torch.FloatTensor(audio)
    else:
        raise UnsupportedAudioFormat(audio_path[-4:])

    # Remove any channel data.
    if len(audio.shape) > 1:
        if audio.shape[0] < 5:
            audio = audio[0]
        else:
            assert audio.shape[1] < 5
            audio = audio[:, 0]

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    if torch.any(audio > 2) or not torch.any(audio < 0):
        ...  # TODO: Product of old code, this may be removable
    audio.clip_(-1, 1)

    return audio.unsqueeze(0)


class Voice:
    __slots__ = ("voice_name", "voice_dir", "_loaded", "_torch_state")

    def __init__(self, voice_name: str, voice_dir: str):
        self.voice_name = voice_name  # Todo: Save voice in json format file
        self.voice_dir = voice_dir
        self._loaded: bool = False
        self._torch_state = None

    @property
    def torch_state(self):
        if not self._loaded:
            self.lazy_load()
        return self._torch_state

    def lazy_load(self):
        self._loaded: bool = True
        files = [os.path.join(self.voice_dir, sub) for sub in os.listdir(self.voice_dir)]
        voice_samples = list(filter(lambda f: f.endswith(".wav") or f.endswith(".mp3"), files))
        conditioning_latents = [_load_audio(cond_path, 22050) for cond_path in voice_samples]

        # Load torch file
        torch_save_file = next(filter(lambda f: f.endswith(".pth"), files), None)
        self._torch_state = torch.load(torch_save_file, weights_only=True) if torch_save_file else None
        if not self._torch_state:
            pth_save_path = os.path.join(os.path.dirname(self.voice_dir), f"{self.voice_name}.pth")
            torch.save(conditioning_latents, pth_save_path)
            self._torch_state = torch.load(pth_save_path)

    def validate_voice(self):
        if not self.torch_state:
            raise MissingVoiceException(self.voice_name, self.voice_dir)


class Voices:
    ALL_VOICES: dict[str, Voice] = {}

    def __getitem__(self, voice: str) -> Voice:
        if Voices.ALL_VOICES == {}:
            Voices.ALL_VOICES = {
                voice_dir: Voice(voice_dir, os.path.join(Config.get_voice_dir, voice_dir))
                for voice_dir in os.listdir(Config.get_voice_dir)
            }

        if voice not in self.ALL_VOICES:
            raise MissingVoiceException(voice, Config.get_voice_dir)
        return self.ALL_VOICES.get(voice, None)


def safe_load_voice(voice: str) -> Voice:
    voice_obj = Voices()[voice]
    voice_obj.validate_voice()
    return voice_obj
