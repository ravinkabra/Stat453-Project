# Network Module

[English](README.md) | [中文](README_zh.md)

The network module focuses on neural network architecture definition, decoupled from training logic.

## Directory Structure

```
network/
├── base/              # Base network abstract classes
│   ├── network.py     # BaseNetwork base class
│   └── config.py      # Network configuration classes
└── tutorial_lenet/    # LeNet tutorial implementation
    ├── network.py     # LeNet network implementation
    └── config.py      # LeNet configuration
```

## Core Components

### BaseNetwork

The base class for all network architectures, providing a unified interface:

```python
class BaseNetwork(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        """Network forward propagation"""
        pass
```

### Usage Example

```python
from src.network.tutorial_lenet.network import LeNet
from src.network.tutorial_lenet.config import LeNetConfig

# Configure network
config = LeNetConfig(num_classes=10)
network = LeNet(config)

# Use network
output = network(input_tensor)
```

## Extension Guide

To add a new network architecture:

1. Create a new folder under `network/`
2. Implement network class (inheriting BaseNetwork is recommended but not mandatory)
3. Create corresponding configuration class
4. Update `__init__.py` to export the new network
