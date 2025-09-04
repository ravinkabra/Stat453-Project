from torch import nn
import torch
from ..base.config import BaseNetworkConfig

class BaseNetwork(nn.Module):
    def __init__(self, config: BaseNetworkConfig):
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
