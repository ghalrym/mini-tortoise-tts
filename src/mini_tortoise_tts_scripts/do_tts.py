import os

import torch
import torchaudio
from typer import Typer, Argument

from mini_tortoise_tts import TextToSpeech, safe_load_voice

app = Typer()


def run(
    text: str = Argument(help="Text to speak."),
    voice: str = Argument(
        help="Selects the voice to use for generation. See options in voices/ directory (and add your own!)"
    ),
    preset: str = Argument(help="Which voice preset to use.", default="fast"),
    use_deepspeed: bool = Argument(help="Use deepspeed for speed bump.", default=False),
    kv_cache: bool = Argument(
        help="If you disable this please wait for a long a time to get the output",
        default=True,
    ),
    half: bool = Argument(
        help="float16(half) precision inference if True it's faster and take less vram and ram",
        default=True,
    ),
    output_path: str = Argument(help="Where to store outputs.", default="results/"),
    candidates: int = Argument(
        help="How many output candidates to produce per-voice.", default=3
    ),
    seed: int = Argument(
        help="Random seed which can be used to reproduce results.", default=None
    ),
    cvvp_amount: float = Argument(
        help=(
            "How much the CVVP model should influence the output."
            "Increasing this can in some cases reduce the likelihood of multiple speakers. Defaults to 0 (disabled)"
        ),
        default=0.0,
    ),
):
    # Ensure output path exists
    os.makedirs(output_path, exist_ok=True)

    # Deepspeed not supported on mac
    if torch.backends.mps.is_available():
        use_deepspeed = False

    # Load voice
    text_to_speech = TextToSpeech(
        safe_load_voice(voice),
        preset=preset,
        use_deepspeed=use_deepspeed,
        kv_cache=kv_cache,
        half=half,
        seed=seed,
    )

    generated_audio = text_to_speech.generate(
        text,
        k=candidates,
        cvvp_amount=cvvp_amount,
    )

    # Save output
    if not isinstance(generated_audio, list):
        generated_audio = [generated_audio]

    for audio_index, audio_tensor in enumerate(generated_audio):
        torchaudio.save(
            os.path.join(output_path, f"{voice}_{audio_index}.wav"),
            audio_tensor.squeeze(0).cpu(),
            24000,
        )


def start():
    app()
