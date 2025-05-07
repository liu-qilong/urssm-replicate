import torch
from torch import nn

class PositionalEncoding(torch.nn.Module):
    def __init__(self, device, b: float = 1.25, l: int = 100) -> None:
        super().__init__()
        self.device = device
        self.b = b
        self.l = l
        self.w = self.b ** torch.arange(int(self.l / 2), device=self.device) * torch.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.sin(self.w * x.unsqueeze(-1)), torch.cos(self.w * x.unsqueeze(-1))], dim=-1)