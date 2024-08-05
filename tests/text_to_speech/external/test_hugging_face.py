import pytest

from unittest.mock import patch, Mock

from config import MODELS_DIR
from external.hugging_face import get_model_path, ModelDoesNotExist


@pytest.mark.parametrize(
    "model_file,is_model",
    [
        pytest.param("autoregressive.pth", True, id="autoregressive-expected"),
        pytest.param("classifier.pth", True, id="classifier-expected"),
        pytest.param("rlg_auto.pth", True, id="rlg_auto-expected"),
        pytest.param("hifidecoder.pth", True, id="hifidecoder-expected"),
        pytest.param("ai-voice.pth", False, id="ai_voice-unexpected"),
    ]
)
def test_get_model_path(model_file: str, is_model: bool):
    if is_model:
        with patch("external.hugging_face.hf_hub_download", return_value=Mock()) as hf_hub_download:
            get_model_path(model_file)
            hf_hub_download.assert_called_once_with(
                repo_id="Manmay/tortoise-tts",
                filename=model_file,
                cache_dir=MODELS_DIR
            )
    else:
        with pytest.raises(ModelDoesNotExist):
            get_model_path(model_file)
