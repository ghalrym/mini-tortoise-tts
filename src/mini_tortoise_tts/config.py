import os


class classproperty(property):

    def __get__(self, instance, owner):
        return self.fget(owner)


class Config:

    @classproperty
    def text_to_speech_dir(cls) -> str:
        return os.environ.get(
            "MINI_TORTOISE_TTS", os.path.dirname(os.path.realpath(__file__))
        )

    @classproperty
    def default_mel_norm_file(cls) -> str:
        return os.path.join(cls.text_to_speech_dir, "build_ins/mel_norms.pth")

    @classproperty
    def default_tokenizer_file(cls) -> str:
        return os.path.join(cls.text_to_speech_dir, "build_ins/tokenizer.json")

    @classproperty
    def default_models_directory(cls) -> str:
        return os.path.join(cls.text_to_speech_dir, "models/")

    @classproperty
    def get_voice_dir(cls) -> str:
        return os.path.join(cls.text_to_speech_dir, "voices/")

    @classproperty
    def get_model_presets(cls) -> dict[str, dict[str, int]]:
        return {
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

    @classproperty
    def get_models(cls) -> dict[str, str]:
        return {
            "autoregressive.pth": "https://huggingface.co/Manmay/tortoise-tts/resolve/main/autoregressive.pth",
            "classifier.pth": "https://huggingface.co/Manmay/tortoise-tts/resolve/main/classifier.pth",
            "rlg_auto.pth": "https://huggingface.co/Manmay/tortoise-tts/resolve/main/rlg_auto.pth",
            "hifidecoder.pth": "https://huggingface.co/Manmay/tortoise-tts/resolve/main/hifidecoder.pth",
        }
