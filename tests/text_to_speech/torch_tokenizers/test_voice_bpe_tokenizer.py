import pytest

from mini_tortoise_tts.config import Config
from mini_tortoise_tts.torch_tokenizers import VoiceBpeTokenizer
from mini_tortoise_tts.torch_tokenizers.voice_bpe_tokenizer import (
    english_cleaners,
    basic_cleaners,
)

BASIC_CLEANER_TOKENIZER = VoiceBpeTokenizer(Config.default_tokenizer_file, True)
ENGLISH_CLEARNER_TOKENIZER = VoiceBpeTokenizer(Config.default_tokenizer_file, False)


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("Hello World", "hello world"),
        pytest.param("\r\n", " "),
        pytest.param("password123", "password123"),
        pytest.param("1 sentence... 2nd sentence", "1 sentence... 2nd sentence"),
        pytest.param("words ar misspelled", "words ar misspelled"),
        pytest.param("[SPACE]", "[space]"),
        pytest.param("[UNK]", "[unk]"),
    ],
)
def test_basic_cleaners(text, expected):
    assert basic_cleaners(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("Hello World", "hello world"),
        pytest.param("\r\n", " "),
        pytest.param("password123", "password one hundred twenty-three"),
        pytest.param(
            "1 sentence... 2nd sentence",
            "one sentence... second sentence",
        ),
        pytest.param("words ar misspelled", "words ar misspelled"),
        pytest.param("[SPACE]", "[space]"),
        pytest.param("[UNK]", "[unk]"),
    ],
)
def test_english_cleaners(text, expected):
    assert english_cleaners(text) == expected


@pytest.mark.parametrize(
    "test_phrase, expected",
    [
        pytest.param("Hello World", [62, 84, 28, 2, 179, 79]),
        pytest.param("password123", [29, 55, 32, 179, 17, 1, 1, 1]),
        pytest.param(
            "1 sentence... 2nd sentence",
            [1, 2, 32, 86, 50, 117, 9, 9, 9, 2, 1, 27, 17, 2, 32, 86, 50, 117],
        ),
        pytest.param(
            "words ar misspelled", [179, 17, 32, 2, 59, 2, 26, 54, 32, 124, 84, 49]
        ),
        pytest.param("[SPACE]", [1, 248, 238, 1]),
        pytest.param("[UNK]", [1, 97, 24, 1]),
    ],
)
def test_voice_bpe_encode(test_phrase, expected):
    assert BASIC_CLEANER_TOKENIZER.encode(test_phrase) == expected


@pytest.mark.parametrize(
    "test_tokens, expected",
    [
        pytest.param([62, 84, 28, 2, 179, 79], "hello world"),
        pytest.param([29, 55, 32, 179, 17, 1, 1, 1], "password"),
        pytest.param(
            [1, 2, 32, 86, 50, 117, 9, 9, 9, 2, 1, 27, 17, 2, 32, 86, 50, 117],
            " sentence... nd sentence",
        ),
        pytest.param(
            [179, 17, 32, 2, 59, 2, 26, 54, 32, 124, 84, 49], "words ar misspelled"
        ),
        pytest.param([1, 248, 238, 1], "space"),
        pytest.param([1, 97, 24, 1], "unk"),
    ],
)
def test_voice_bpe_decode(test_tokens, expected):
    assert BASIC_CLEANER_TOKENIZER.decode(test_tokens) == expected


@pytest.mark.parametrize(
    "test_phrase, expected",
    [
        pytest.param("Hello World", [62, 84, 28, 2, 179, 79]),
        pytest.param(
            "password12",
            [29, 55, 32, 179, 17, 2, 33, 100, 25, 76],
        ),
        pytest.param(
            "1 sentence... 2nd sentence",
            [110, 2, 32, 86, 50, 117, 9, 9, 9, 2, 66, 135, 17, 2, 32, 86, 50, 117],
        ),
        pytest.param(
            "words ar misspelled", [179, 17, 32, 2, 59, 2, 26, 54, 32, 124, 84, 49]
        ),
        pytest.param("[SPACE]", [1, 248, 238, 1]),
        pytest.param("[UNK]", [1, 97, 24, 1]),
    ],
)
def test_voice_bpe_encode(test_phrase, expected):
    assert ENGLISH_CLEARNER_TOKENIZER.encode(test_phrase) == expected


@pytest.mark.parametrize(
    "test_tokens, expected",
    [
        pytest.param([62, 84, 28, 2, 179, 79], "hello world"),
        pytest.param(
            [29, 55, 32, 179, 17, 2, 33, 100, 25, 76],
            "password twelve",
        ),
        pytest.param(
            [110, 2, 32, 86, 50, 117, 9, 9, 9, 2, 66, 135, 17, 2, 32, 86, 50, 117],
            "one sentence... second sentence",
        ),
        pytest.param(
            [179, 17, 32, 2, 59, 2, 26, 54, 32, 124, 84, 49], "words ar misspelled"
        ),
        pytest.param([1, 248, 238, 1], "space"),
        pytest.param([1, 97, 24, 1], "unk"),
    ],
)
def test_voice_bpe_decode(test_tokens, expected):
    assert ENGLISH_CLEARNER_TOKENIZER.decode(test_tokens) == expected
