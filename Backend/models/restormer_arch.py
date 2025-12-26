# models/restormer_arch.py

import torch
import torch.nn as nn
from einops import rearrange

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class Restormer(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4,6,6,8],
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66
    ):
        super().__init__()

        self.embedding = nn.Conv2d(inp_channels, dim, 3, padding=1)
        self.body = nn.Sequential(
            *[FeedForward(dim, ffn_expansion_factor) for _ in range(sum(num_blocks))]
        )
        self.output = nn.Conv2d(dim, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.body(x)
        return self.output(x)
