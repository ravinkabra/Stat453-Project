# LR Scheduler Module

[English](README.md) | [中文](README_zh.md)

The learning rate scheduler module focuses on learning rate adjustment strategies during training, decoupled from training logic.

> **Note**: This module is a utility class (prefixed with `_`), following the project specifications for utility classes, commonly used but rarely modified. For more information, please refer to the [Project Architecture Specifications](../../docs/architecture.md).

## Directory Structure

```text
_lr_scheduler/
├── base/                      # Base scheduler parameters abstract classes
│   └── config.py              # BaseLRSchedulerParams base class
├── step/                      # StepLR scheduler
│   └── config.py              # StepLRParams parameter class
├── multistep/                 # MultiStepLR scheduler
│   └── config.py              # MultiStepLRParams parameter class
├── exponential/               # ExponentialLR scheduler
│   └── config.py              # ExponentialLRParams parameter class
├── cosine/                    # CosineAnnealingLR scheduler
│   └── config.py              # CosineAnnealingLRParams parameter class
├── reduce_on_plateau/         # ReduceLROnPlateau scheduler
│   └── config.py              # ReduceLROnPlateauParams parameter class
├── cyclic/                    # CyclicLR scheduler
│   └── config.py              # CyclicLRParams parameter class
├── onecycle/                  # OneCycleLR scheduler
│   └── config.py              # OneCycleLRParams parameter class
├── cosine_warm_restarts/      # CosineAnnealingWarmRestarts scheduler
│   └── config.py              # CosineAnnealingWarmRestartsParams parameter class
└── __init__.py                # Module initialization
```

## Core Components

### BaseLRSchedulerParams

The base class for all learning rate scheduler parameters:

```python
@dataclass
class BaseLRSchedulerParams:
    """Base class for learning rate scheduler parameters"""

    _target_: str = Field("torch.optim.lr_scheduler.CosineAnnealingLR", description="Learning rate scheduler class")
    optimizer: Optional[Any] = Field(None, description="Optimizer object")
    interval: Literal["step", "epoch"] = Field("step", description="Interval for updating the learning rate")
    frequency: int = Field(1, description="Frequency of updating the learning rate")
    monitor: Optional[str] = Field(None, description="Metric to monitor for learning rate scheduling")
    strict: bool = Field(True, description="Whether to strictly enforce the configuration")
    name: Optional[str] = Field(None, description="Name of the learning rate scheduler")
    T_max: int = Field(10000, description="Total number of training steps")
    eta_min: float = Field(1e-6, description="Minimum learning rate")
```

### UnionLRSchedulerParams

Unified learning rate scheduler parameter type:

```python
UnionLRSchedulerParams = Union[
    BaseLRSchedulerParams,
    StepLRParams,
    MultiStepLRParams,
    ExponentialLRParams,
    CosineAnnealingLRParams,
    ReduceLROnPlateauParams,
    CyclicLRParams,
    OneCycleLRParams,
    CosineAnnealingWarmRestartsParams,
    dict,  # Allow dict for backward compatibility
]
```

### Supported Learning Rate Schedulers

- **StepLR**: Decays learning rate by a fixed step size
- **MultiStepLR**: Decays learning rate at specified milestones
- **ExponentialLR**: Decays learning rate exponentially
- **CosineAnnealingLR**: Decays learning rate with cosine annealing
- **ReduceLROnPlateau**: Reduces learning rate when a metric stops improving
- **CyclicLR**: Cycles learning rate between two boundaries
- **OneCycleLR**: Uses one cycle learning rate policy
- **CosineAnnealingWarmRestarts**: Cosine annealing with warm restarts

## Usage Example

```python
from src._lr_scheduler import UnionLRSchedulerParams, StepLRParams

# Create StepLR scheduler parameters
step_params = StepLRParams(
    step_size=30,
    gamma=0.1,
    interval="epoch"
)

# Use in configuration
config = SomeConfig(
    lr_scheduler=step_params
)
```

## Extension Guide

To add support for a new learning rate scheduler:

1. Create a new folder under `_lr_scheduler/` (e.g., `new_scheduler/`)
2. Create `config.py` implementing `NewSchedulerParams` class inheriting from `BaseLRSchedulerParams`
3. Add imports and update Union type in `__init__.py`
4. Update `__all__` export list
5. Update relevant documentation
