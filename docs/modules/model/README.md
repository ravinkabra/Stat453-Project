# Model Module

[English](README.md) | [中文](README_zh.md)

The model module is the core abstraction layer of the deep learning training framework, built on PyTorch Lightning, responsible for integrating networks, optimizers, learning rate schedulers, and metric management, providing unified model training interfaces.

> **Note**: This module is a tool class (prefixed with `_`), following the tool class definitions in the project specifications, commonly used but not frequently modified. For more information, please refer to the [Project Architecture Specification](../../architecture.md).

## Directory Structure

```text
model/
├── __init__.py                   # Module initialization and type definitions
├── base/                         # Base model abstraction
│   ├── __init__.py              # Submodule initialization
│   ├── config.py                # Base model configuration class
│   └── model.py                 # Base model implementation
├── tutorial_mnist/               # MNIST model example
│   ├── __init__.py              # Submodule initialization
│   ├── config.py                # MNIST model configuration class
│   └── model.py                 # MNIST model implementation
├── README_zh.md                  # Chinese documentation
└── README.md                     # English documentation
```

## Core Components

### BaseModel

Base model class - inherits from PyTorch Lightning's LightningModule:

```python
class BaseModel(LightningModule, ABC):
    """
    Base model abstract class - provides unified model training interfaces
    Integrates network, optimizer, scheduler, and metric management
    """
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.metric_manager = MetricManager(config.metric_manager)
```


### BaseModelConfig

Base model configuration class - defines the basic configuration structure for models:

```python
@dataclass
class BaseModelConfig:
    _target_: str = Field(default="src.model.base.model.BaseModel")

    optimizer: Optional[UnionOptimizerParams] = Field(
        default_factory=BaseOptimizerParams,
        description="Optimizer configuration"
    )

    lr_scheduler: Optional[UnionLRSchedulerParams] = Field(
        default_factory=BaseLRSchedulerParams,
        description="Learning rate scheduler configuration"
    )

    metric_manager: Optional[MetricManagerConfig] = Field(
        default_factory=MetricManagerConfig
    )
```

### MnistModel

MNIST model example - an example of specific model implementation:

```python
class MnistModel(BaseModel):
    """
    MNIST handwritten digit recognition model
    Integrates LeNet network and cross-entropy loss
    """
    def __init__(self, config: MnistModelConfig):
        super().__init__(config)
        self.network = LeNet(config.network)
        self.loss = nn.CrossEntropyLoss()
```

## Main Features

### Model Abstraction Layer

- **Unified interface**: Provides standardized training interfaces based on PyTorch Lightning
- **Component integration**: Automatically integrates networks, optimizers, schedulers, and metric management
- **Configuration-driven**: Supports model instantiation through configuration classes
- **Type safety**: Uses Pydantic to provide complete configuration validation

### Optimizer Management

- **Flexible configuration**: Supports single optimizer or module-level optimizer configuration
- **Automatic instantiation**: Automatically creates optimizers through Hydra's instantiate
- **Parameter passing**: Automatically passes model parameters to optimizers

### Learning Rate Scheduler

- **Scheduler integration**: Seamlessly integrates various learning rate scheduling strategies
- **Dynamic adjustment**: Supports dynamic learning rate adjustment during training
- **Monitoring integration**: Supports metric-based learning rate scheduling

### Metric Management

- **Unified management**: Centralized management of all training metrics
- **Automatic calculation**: Automatic calculation and recording of metrics
- **Logging integration**: Seamless integration with logging systems

## Usage Example

### Basic Model Implementation

```python
from src.model.base import BaseModel, BaseModelConfig

class MyModel(BaseModel):
    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        self.network = MyNetwork(config.network)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss
```

### Configuration Usage

```python
from src.model import UnionModelConfig, MnistModelConfig

# Create model configuration
model_config = MnistModelConfig(
    optimizer=AdamParams(lr=1e-3),
    lr_scheduler=StepLRParams(step_size=30, gamma=0.1),
    metric_manager=MetricManagerConfig(...)
)

# Use in project configuration
project_config = ProjectConfig(
    model=model_config
)
```
