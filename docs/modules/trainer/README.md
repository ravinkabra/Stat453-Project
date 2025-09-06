# Trainer Module

[English](README.md) | [中文](README_zh.md)

The trainer module focuses on PyTorch Lightning Trainer configuration and management, unifying management of all training process parameters.

> **Note**: This module is a tool class (prefixed with `_`), following the tool class definitions in the project specifications, commonly used but not frequently modified. For more information, please refer to the [Project Architecture Specification](../../architecture.md).

## Directory Structure

```text
_trainer/
├── base/              # Base trainer configuration abstract classes
│   ├── __init__.py    # Submodule initialization
│   └── config.py      # BaseTrainerConfig configuration class
└── __init__.py        # Module initialization and type definitions
```

## Core Components

### BaseTrainerConfig

Trainer configuration base class - encapsulates all configuration parameters of PyTorch Lightning Trainer:

```python
@dataclass(config=ConfigDict(extra="allow"))
class BaseTrainerConfig:
    """Trainer configuration base class - encapsulates all configuration parameters of PyTorch Lightning Trainer"""

    # Hardware and accelerator configuration
    accelerator: Union[str, Any] = Field(
        default="auto",
        description="Accelerator type ('cpu', 'gpu', 'tpu', 'hpu', 'mps', 'auto')",
    )
    strategy: Union[str, Any] = Field(default="auto", description="Training strategy")
    devices: Union[List[int], str, int] = Field(
        default="auto", description="Devices to use"
    )
    num_nodes: int = Field(default=1, description="Number of GPU nodes for distributed training")

    # Precision configuration
    precision: Union[...] = Field(default="32-true", description="Training precision")

    # Logging and recording
    logger: Union[Any, List[Any], bool, None] = Field(
        default=True, description="Logger"
    )

    # Training control
    max_epochs: Optional[int] = Field(default=None, description="Maximum number of training epochs")
    max_steps: int = Field(default=-1, description="Maximum number of training steps")
    min_steps: Optional[int] = Field(default=None, description="Minimum number of training steps")

    # Validation control
    val_check_interval: Union[int, float, None] = Field(
        default=1.0, description="Validation check interval"
    )
    check_val_every_n_epoch: Optional[int] = Field(
        default=1, description="Perform validation every N epochs"
    )

    # Gradient control
    accumulate_grad_batches: int = Field(default=1, description="Number of gradient accumulation batches")
    gradient_clip_val: Union[int, float, None] = Field(
        default=None, description="Gradient clipping value"
    )
```

### UnionTrainerConfig

Unified trainer configuration type:

```python
UnionTrainerConfig = Union[BaseTrainerConfig, dict]
```

## Main Features

### Hardware Configuration

- **Multi-accelerator support**: Supports CPU, GPU, TPU, HPU, MPS
- **Distributed training**: Supports multiple distributed strategies and multi-node training
- **Auto-detection**: Automatically detects and configures available hardware
- **Mixed precision**: Supports multiple precision training modes

### Training Control

- **Flexible epochs**: Configurable maximum/minimum training epochs and steps
- **Validation scheduling**: Flexible validation frequency and batch limits
- **Gradient optimization**: Gradient accumulation and clipping control
- **Performance tuning**: Deterministic mode and benchmarking

### Logging and Monitoring

- **Multi-logger integration**: Supports multiple logger types
- **Progress monitoring**: Training progress and performance metric monitoring
- **Debugging support**: Quick development runs and overfitting testing
- **Exception detection**: Automatic exception detection and handling

## Usage Example

```python
from src._trainer import UnionTrainerConfig, BaseTrainerConfig

# Create trainer configuration
trainer_config = BaseTrainerConfig(
    accelerator="gpu",
    devices=1,
    max_epochs=100,
    precision="16-mixed"
)

# Use in configuration
config = SomeConfig(
    trainer=trainer_config
)
```
