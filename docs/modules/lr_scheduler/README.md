# LR Scheduler Module

[English](README.md) | [中文](README_zh.md)

The learning rate scheduler module focuses on learning rate adjustment strategies during training, decoupled from training logic.

> **Note**: This module is a tool class (prefixed with `_`), following the tool class definitions in the project specifications, commonly used but not frequently modified. For more information, please refer to the [Project Architecture Specification](../../architecture.md).

## Directory Structure

```text
_lr_scheduler/
├── base/                      # Base scheduler parameter abstract classes
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

Base class for all learning rate scheduler parameters:

```python
@dataclass
class BaseLRSchedulerParams:
    """Learning rate scheduler parameters base class"""

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

Unified learning rate scheduler parameters type:

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

- **StepLR**: Decay learning rate by a fixed factor at fixed steps
- **MultiStepLR**: Decay learning rate at specified milestones
- **ExponentialLR**: Exponential decay of learning rate
- **CosineAnnealingLR**: Cosine annealing decay
- **ReduceLROnPlateau**: Adjust learning rate based on metric changes
- **CyclicLR**: Cyclic learning rate scheduling
- **OneCycleLR**: Single cycle learning rate scheduling
- **CosineAnnealingWarmRestarts**: Cosine annealing with warm restarts

## Main Features

### Scheduling Strategy Diversity

- **Fixed scheduling**: Adjust learning rate by step or milestone
- **Adaptive scheduling**: Dynamically adjust learning rate based on training metrics
- **Periodic scheduling**: Cyclic or single-cycle learning rate changes
- **Exponential decay**: Exponential form of decay strategies

### Configuration Flexibility

- **Interval control**: Support updating by step or epoch
- **Frequency setting**: Configurable update frequency
- **Metric monitoring**: Support for metric-based scheduling decisions
- **Parameter validation**: Strict configuration validation and type checking

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
