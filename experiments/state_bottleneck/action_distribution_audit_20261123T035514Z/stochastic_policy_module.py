import torch
from torch import nn, Tensor


class StochasticCTBR(nn.Module):
    def __init__(self, base, std):
        super().__init__()
        self.base = base
        self.register_buffer("std", std.reshape(1, 4))

    def forward(self, corners: Tensor, history: Tensor) -> Tensor:
        mean = self.base(corners, history)
        mean = mean.clamp(-0.999999, 0.999999)

        raw_mean = torch.atanh(mean)
        sampled_raw = raw_mean + torch.randn_like(raw_mean) * self.std

        return torch.tanh(sampled_raw)
