import torch.nn as nn
import torch

from .config import LeNetConfig


class LeNet(nn.Module):
    def __init__(self, config: LeNetConfig):
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            nn.Conv2d(config.input_channels, 6, 5),  # 28x28 -> 24x24
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24x24 -> 12x12
            nn.Conv2d(6, 16, 5),  # 12x12 -> 8x8
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
