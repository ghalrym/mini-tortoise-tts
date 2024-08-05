from load_voices import Voices, Voice

_ALL_VOICES = sum(
    (
        Voices(voice)
        for voice in [
            r"C:\Users\Andrew\PycharmProjects\RoseAI\src\.voices",
            r"C:\Users\Andrew\PycharmProjects\RoseAI\.micro_services\tortoise-tts\tortoise\voices",
        ]
    ),
    Voices(None),
)


def safe_load_voice(voice: str) -> Voice:
    voice_obj = _ALL_VOICES[voice]
    voice_obj.validate_voice()
    return voice_obj
