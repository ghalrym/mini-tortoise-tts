from torch import nn

from mini_tortoise_tts.torch_modules.attention_block import AttentionBlock


class ConditioningEncoder(nn.Module):
    __slots__ = ("init", "attn", "dim", "do_checkpointing", "mean")

    def __init__(
        self,
        spec_dim: int,
        embedding_dim: int,
        attn_blocks: int = 6,
        num_attn_heads: int = 4,
        do_checkpointing: bool = False,
        mean: bool = False,
    ):
        super().__init__()
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        attn = [AttentionBlock(embedding_dim, num_attn_heads) for _ in range(attn_blocks)]
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.attn(self.init(x))
        return h.mean(dim=2) if self.mean else h[:, :, 0]
