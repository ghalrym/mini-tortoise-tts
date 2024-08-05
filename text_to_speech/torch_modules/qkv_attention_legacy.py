import math

import torch
from torch import nn


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        # More stable with f16 than dividing afterwards
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)

        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(
                bs * self.n_heads, weight.shape[-2], weight.shape[-1]
            )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        if mask is not None:
            # The proper way to do this is to mask before the softmax using -inf,
            #    but that doesn't work properly on CPUs.
            mask = mask.repeat(self.n_heads, 1).unsqueeze(1)
            weight = weight * mask

        return torch.einsum("bts,bcs->bct", weight, v).reshape(bs, -1, length)
