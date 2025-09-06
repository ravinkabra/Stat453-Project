# Network Module

[English](README.md) | [中文](README_zh.md)

The network module focuses on neural network architecture definitions, decoupled from training logic.

## Directory Structure

```text
network/
├── base/              # Base network abstract classes
│   ├── network.py     # BaseNetwork base class
│   └── config.py      # Network configuration class
└── tutorial_lenet/    # LeNet tutorial implementation
    ├── network.py     # LeNet network implementation
    └── config.py      # LeNet configuration
```

## Core Components

### BaseNetwork

Base class for all network architectures, providing unified interfaces:

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

## Main Features

### Network Architecture Abstraction

- **Unified interface**: Standardized network forward propagation interface
- **Modular design**: Supports complex network architecture combinations
- **Configuration management**: Manages network parameters through configuration classes
- **Type safety**: Configuration validation using Pydantic

### LeNet Implementation

Specific implementation of LeNet network:

```python
class LeNet(BaseNetwork):
    def __init__(self, config: LeNetConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, config.num_classes)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## Extension Guidelines

To add new network architectures:

1. Create a new folder under `network/`
2. Implement network class inheriting from `BaseNetwork` (not mandatory)
3. Create corresponding configuration class
4. Update `__init__.py` to export new network
