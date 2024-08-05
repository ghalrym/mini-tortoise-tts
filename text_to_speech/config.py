import os
from typing import Final

_TEXT_TO_SPEECH_DIR: Final[str] = os.path.dirname(os.path.realpath(__file__))

DEFAULT_MEL_NORM_FILE: Final[str] = os.path.join(_TEXT_TO_SPEECH_DIR, ".data/build_ins/mel_norms.pth")
DEFAULT_TOKENIZE_FILE: Final[str] = os.path.join(_TEXT_TO_SPEECH_DIR, ".data/build_ins/tokenizer.json")
MODELS_DIR: Final[str] = os.path.join(_TEXT_TO_SPEECH_DIR, ".data/models/")

MODELS: Final[dict[str, str]] = {
    "autoregressive.pth": "https://huggingface.co/Manmay/tortoise-tts/resolve/main/autoregressive.pth",
    "classifier.pth": "https://huggingface.co/Manmay/tortoise-tts/resolve/main/classifier.pth",
    "rlg_auto.pth": "https://huggingface.co/Manmay/tortoise-tts/resolve/main/rlg_auto.pth",
    "hifidecoder.pth": "https://huggingface.co/Manmay/tortoise-tts/resolve/main/hifidecoder.pth",
}

MODEL_PRESETS: Final[dict[str, dict[str, int]]] = {
    "ultra_fast": {
        "num_autoregressive_samples": 1,
        # "diffusion_iterations": 10,
    },
    "fast": {
        "num_autoregressive_samples": 32,
        # "diffusion_iterations": 50,
    },
    "standard": {
        "num_autoregressive_samples": 256,
        # "diffusion_iterations": 200,
    },
    "high_quality": {
        "num_autoregressive_samples": 256,
        # "diffusion_iterations": 400,
    },
}
