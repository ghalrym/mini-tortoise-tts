import torch
from torch import Tensor
from transformers import GPT2Model, GPT2Config

from mini_tortoise_tts.gpt.learned_position_embeddings import LearnedPositionEmbeddings


HfGptTransformer = tuple[GPT2Model, LearnedPositionEmbeddings, LearnedPositionEmbeddings]


class GPT2ModelNullEmbeddings(GPT2Model):

    def wpe(self, range: Tensor):
        return torch.zeros((range.shape[0], range.shape[1], self.config.n_embd), device=range.device)

    def wte(self, *args, **kwargs):
        """This function is never used now"""
        raise NotImplementedError()


def build_hf_gpt_transformer(
    n_layers: int, model_dim: int, n_heads: int, max_mel_seq_len: int, max_text_seq_len: int, checkpointing: bool
) -> HfGptTransformer:
    """GPT-2 implemented by the HuggingFace library."""
    gpt = GPT2ModelNullEmbeddings(
        GPT2Config(
            vocab_size=256,  # Unused.
            n_positions=max_mel_seq_len + max_text_seq_len,
            n_ctx=max_mel_seq_len + max_text_seq_len,
            n_embd=model_dim,
            n_layer=n_layers,
            n_head=n_heads,
            gradient_checkpointing=checkpointing,
            use_cache=not checkpointing,
        )
    )
    return (
        gpt,
        LearnedPositionEmbeddings(max_mel_seq_len, model_dim),
        LearnedPositionEmbeddings(max_text_seq_len, model_dim),
    )
