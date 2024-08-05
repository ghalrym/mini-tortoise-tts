from huggingface_hub import hf_hub_download

from config import MODELS, MODELS_DIR


class ModelDoesNotExist(Exception):

    def __init__(self, model_name):
        super().__init__("Model does not exist: {}".format(model_name))



def get_model_path(model_name: str) -> str:
    """Get path to given model, download it to cached directory if it doesn't exist."""
    if model_name not in MODELS:
        raise ModelDoesNotExist(model_name)
    model_path = hf_hub_download(repo_id="Manmay/tortoise-tts", filename=model_name, cache_dir=MODELS_DIR)
    return model_path
