from .config import ExampleNetworkConfig
from torch import nn


class ExampleNetwork(nn.Module):
    def __init__(self, config: ExampleNetworkConfig):
        super(ExampleNetwork, self).__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.hidden_layers = config.hidden_layers or []
        self.activation = config.activation

        layers = []
        in_features = self.input_size

        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(getattr(nn, self.activation)())
            in_features = hidden_size

        layers.append(nn.Linear(in_features, self.output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
