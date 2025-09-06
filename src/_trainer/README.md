# Trainer Module

[English](README.md) | [中文](README_zh.md)

The trainer module focuses on PyTorch Lightning Trainer configuration and management, unifying all parameters for the training process.

> **Note**: This module is a utility class (prefixed with `_`), following the project specifications for utility classes that are commonly used but rarely modified. For more information, please refer to [Project Architecture Specifications](../../docs/architecture.md).

## Directory Structure

```text
_trainer/
├── base/              # Base trainer configuration abstract classes
│   ├── __init__.py    # Module initialization
│   └── config.py      # BaseTrainerConfig configuration class
└── __init__.py        # Module initialization and type definitions
```

## Core Components

### BaseTrainerConfig

Trainer configuration base class - encapsulates all configuration parameters for PyTorch Lightning Trainer:

```python
@dataclass(config=ConfigDict(extra="allow"))
class BaseTrainerConfig:
    """Trainer configuration base class - encapsulates all PyTorch Lightning Trainer configuration parameters"""

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

    # Logging and loggers
    logger: Union[Any, List[Any], bool, None] = Field(
        default=True, description="Logger"
    )

    # Training control
    max_epochs: Optional[int] = Field(default=None, description="Maximum training epochs")
    max_steps: int = Field(default=-1, description="Maximum training steps (-1 for unlimited)")
    min_steps: Optional[int] = Field(default=None, description="Minimum training steps")

    # Validation control
    val_check_interval: Union[int, float, None] = Field(
        default=1.0, description="Validation check interval"
    )
    check_val_every_n_epoch: Optional[int] = Field(
        default=1, description="Validate every N epochs"
    )

    # Gradient control
    accumulate_grad_batches: int = Field(default=1, description="Gradient accumulation batches")
    gradient_clip_val: Union[int, float, None] = Field(
        default=None, description="Gradient clipping value"
    )

    # Other important configurations...
```

### UnionTrainerConfig

Unified trainer configuration type:

```python
UnionTrainerConfig = Union[BaseTrainerConfig, dict]
```

## Main Features

- **Hardware Configuration**: Support for multiple accelerators (CPU, GPU, TPU, HPU, MPS)
- **Distributed Training**: Support for multiple distributed strategies and multi-node training
- **Precision Control**: Support for various mixed precision training modes
- **Training Control**: Flexible training epochs, steps, and time limits
- **Validation Management**: Configurable validation frequency and batch limits
- **Gradient Optimization**: Gradient accumulation, clipping, and synchronization
- **Performance Tuning**: Deterministic mode, benchmarking, anomaly detection
- **Debug Support**: Fast dev run, overfitting testing

## Usage Example

```python
from src._trainer import UnionTrainerConfig, BaseTrainerConfig

# Create trainer configuration
trainer_config = BaseTrainerConfig(
    accelerator="gpu",
    devices=1,
    max_epochs=100,
    precision="16-mixed",
    accumulate_grad_batches=2,
    gradient_clip_val=1.0,
    val_check_interval=0.5
)

# Use in project configuration
project_config = ProjectConfig(
    trainer=trainer_config
)
```

## Configuration Guide

### Hardware Configuration

- `accelerator`: Select compute device type
- `devices`: Specify number of devices or device IDs to use
- `num_nodes`: Number of nodes for distributed training

### Training Control

- `max_epochs`/`max_steps`: Training stop conditions
- `min_epochs`/`min_steps`: Minimum training execution requirements
- `max_time`: Training time limit

### Validation Configuration

- `val_check_interval`: Validation check frequency
- `check_val_every_n_epoch`: Validate every N epochs
- `limit_val_batches`: Validation data batch limit

### Performance Optimization

- `precision`: Mixed precision training settings
- `accumulate_grad_batches`: Gradient accumulation to simulate larger batch_size
- `gradient_clip_val`: Gradient clipping to prevent gradient explosion

## Extension Guide

To extend trainer configuration functionality:

1. Add new configuration fields in `base/config.py`
2. Update field descriptions and type annotations
3. Maintain compatibility with PyTorch Lightning Trainer
4. Update related documentation and usage examples
5. Test the effects of new configurations in actual training
