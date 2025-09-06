# Model Module

[English](README.md) | [中文](README_zh.md)

The model module is the core abstraction layer of the deep learning training framework, built on PyTorch Lightning, responsible for integrating networks, optimizers, learning rate schedulers, and metric management, providing a unified model training interface.

> **Note**: This module is a utility class (prefixed with `_`), following the project specifications for utility classes that are commonly used but rarely modified. For more information, please refer to [Project Architecture Specifications](../../docs/architecture.md).

## Directory Structure

```text
model/
├── __init__.py                   # Module initialization and type definitions
├── base/                         # Base model abstraction
│   ├── __init__.py              # Submodule initialization
│   ├── config.py                # Base model configuration classes
│   └── model.py                 # Base model implementation
├── tutorial_mnist/               # MNIST model example
│   ├── __init__.py              # Submodule initialization
│   ├── config.py                # MNIST model configuration classes
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
    Base model abstract class - provides unified model training interface
    Integrates network, optimizer, scheduler, and metric management
    """
    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.metric_manager = MetricManager(config.metric_manager)
```

### MnistModel

MNIST model example - a concrete model implementation example:

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

- **Unified Interface**: Provides standardized training interface based on PyTorch Lightning
- **Component Integration**: Automatically integrates network, optimizer, scheduler, and metric management
- **Configuration-Driven**: Supports model instantiation through configuration classes
- **Type Safety**: Uses Pydantic to provide complete configuration validation

### Optimizer Management

- **Flexible Configuration**: Supports single optimizer or module-level optimizer configuration
- **Automatic Instantiation**: Automatically creates optimizers through Hydra's instantiate
- **Parameter Passing**: Automatically passes model parameters to optimizers

### Learning Rate Scheduler

- **Scheduler Integration**: Automatically associates with optimizers
- **Configuration Matching**: Supports scheduler configurations that match optimizer structure
- **Dynamic Adjustment**: Supports learning rate dynamic adjustment during training

### Metric Management Integration

- **Unified Management**: Integrates MetricManager for metric calculation and management
- **Automatic Recording**: Automatically updates and records metrics in training steps
- **Group Support**: Supports different phases and types of metric grouping

## Usage Examples

### Basic Usage

```python
from src.model import MnistModel, MnistModelConfig
from hydra.utils import instantiate

# 1. Configure model
config = MnistModelConfig(
    network=LeNetConfig(),  # Network configuration
    optimizer={
        "_target_": "torch.optim.Adam",
        "lr": 0.001,
        "weight_decay": 1e-4
    },
    lr_scheduler={
        "_target_": "torch.optim.lr_scheduler.StepLR",
        "step_size": 30,
        "gamma": 0.1
    },
    metric_manager=MetricManagerConfig(
        metrics={
            "classification": {
                "metrics": {
                    "accuracy": "torchmetrics.Accuracy",
                    "precision": "torchmetrics.Precision",
                    "recall": "torchmetrics.Recall"
                }
            }
        }
    )
)

# 2. Instantiate model
model = instantiate(config)

# 3. Use in training
trainer = Trainer(model=model)
trainer.fit(model, datamodule=datamodule)
```

### Advanced Configuration Example

```python
# Complex model configuration
config = MnistModelConfig(
    network=LeNetConfig(
        num_classes=10,
        dropout_rate=0.5
    ),
    optimizer={
        "_target_": "torch.optim.AdamW",
        "lr": 0.001,
        "betas": [0.9, 0.999],
        "weight_decay": 1e-4
    },
    lr_scheduler={
        "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
        "T_max": 100,
        "eta_min": 1e-6
    },
    metric_manager=MetricManagerConfig(
        metrics={
            "training_state": {
                "metrics": {
                    "loss": ValueRecorderParams(aggregation="mean"),
                    "learning_rate": ValueRecorderParams(aggregation="last")
                }
            },
            "classification": {
                "metrics": {
                    "accuracy": {
                        "_target_": "torchmetrics.Accuracy",
                        "task": "multiclass",
                        "num_classes": 10
                    },
                    "f1_score": {
                        "_target_": "torchmetrics.F1Score",
                        "task": "multiclass",
                        "num_classes": 10,
                        "average": "macro"
                    }
                }
            }
        }
    )
)
```

### Custom Model Implementation

```python
from src.model.base import BaseModel, BaseModelConfig

class CustomModel(BaseModel):
    def __init__(self, config: BaseModelConfig):
        super().__init__(config)
        # Initialize custom network
        self.network = CustomNetwork()
        self.loss = CustomLoss()

    def forward(self, batch):
        """Forward pass"""
        return self.network(batch)

    def training_step(self, batch, batch_idx):
        """Training step"""
        outputs = self.forward(batch)
        loss = self.loss(outputs, batch.targets)

        # Update metrics
        self.metric_manager.update(
            self, batch_idx, "training",
            loss=loss,
            custom_metric=custom_value
        )

        return {"loss": loss}
```

## Configuration Options Details

### BaseModelConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer` | `UnionOptimizerParams` | `BaseOptimizerParams()` | Optimizer configuration |
| `lr_scheduler` | `UnionLRSchedulerParams` | `BaseLRSchedulerParams()` | Learning rate scheduler configuration |
| `metric_manager` | `MetricManagerConfig` | `MetricManagerConfig()` | Metric manager configuration |

### MnistModelConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `network` | `LeNetConfig` | `LeNetConfig()` | LeNet network configuration |

## Design Philosophy

1. **Abstraction Layer Design**: Provides unified model abstraction interface through BaseModel
2. **Component Decoupling**: Network, optimizer, scheduler, and metric management are configured independently
3. **Configuration-Driven**: Implements declarative model definition through configuration classes
4. **Extensibility**: Supports custom model implementation based on abstract base classes
5. **Integration**: Automatically integrates all training-related components

## Extension Guide

To add new model types:

1. Create new configuration class, inheriting from `BaseModelConfig`
2. Implement concrete model class, inheriting from `BaseModel`
3. Set correct `_target_` field in configuration
4. Implement necessary abstract methods (`forward`, etc.)
5. Update type definitions in `__init__.py`
6. Add corresponding documentation and usage examples

## Best Practices

1. **Configuration Organization**: Separate model configuration from training configuration
2. **Component Reuse**: Make reasonable use of existing optimizer and scheduler configurations
3. **Metric Integration**: Make full use of MetricManager's grouping functionality
4. **Type Safety**: Use correct configuration types to avoid runtime errors
5. **Abstract Inheritance**: Implement concrete model logic based on BaseModel
