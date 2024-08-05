from torch import nn

from torch_modules.qkv_attention_legacy import QKVAttentionLegacy
from torch_modules.relative_position_bias import RelativePositionBias


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> GroupNorm32:
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    else:
        groups = 32

    while channels % groups != 0:
        groups = int(groups / 2)

    return GroupNorm32(groups, channels)


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        do_checkpoint: bool = True,
        relative_pos_embeddings: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        self.num_heads = num_heads if (num_head_channels == -1) else self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(
                scale=(channels // self.num_heads) ** 0.5,
                causal=False,
                heads=num_heads,
                num_buckets=32,
                max_distance=64,
            )
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
