# Optimizer Module

[English](README.md) | [中文](README_zh.md)

The optimizer module focuses on configuration and management of model parameter optimization, decoupled from training logic.

> **Note**: This module is a tool class (prefixed with `_`), following the tool class definitions in the project specifications, commonly used but not frequently modified. For more information, please refer to the [Project Architecture Specification](../../architecture.md).

## Directory Structure

```text
_optimizer/
├── base/              # Base optimizer parameter abstract classes
│   └── config.py      # BaseOptimizerParams base class
├── adam/              # Adam optimizer
│   └── config.py      # AdamParams parameter class
├── sgd/               # SGD optimizer
│   └── config.py      # SGDParams parameter class
├── adamw/             # AdamW optimizer
│   └── config.py      # AdamWParams parameter class
├── adadelta/          # AdaDelta optimizer
│   └── config.py      # AdaDeltaParams parameter class
├── adagrad/           # AdaGrad optimizer
│   └── config.py      # AdaGradParams parameter class
├── rmsprop/           # RMSProp optimizer
│   └── config.py      # RMSPropParams parameter class
├── rprop/             # RProp optimizer
│   └── config.py      # RPropParams parameter class
├── lbfgs/             # LBFGS optimizer
│   └── config.py      # LBFGSParams parameter class
└── __init__.py        # Module initialization
```

## Core Components

### BaseOptimizerParams

Base class for all optimizer parameters:

```python
@dataclass
class BaseOptimizerParams:
    """Optimizer parameters base class"""

    _target_: str = Field("torch.optim.Adam", description="Optimizer class")
    lr: float = Field(1e-3, description="Learning rate")
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")
```

### UnionOptimizerParams

Unified optimizer parameters type:

```python
UnionOptimizerParams = Union[
    BaseOptimizerParams,
    AdamParams,
    SGDParams,
    AdamWParams,
    AdaDeltaParams,
    AdaGradParams,
    RMSPropParams,
    RPropParams,
    LBFGSParams,
    dict,  # Allow dict for backward compatibility
]
```

### Supported Optimizers

- **Adam**: Adaptive Moment Estimation optimizer
- **SGD**: Stochastic Gradient Descent
- **AdamW**: Adam with weight decay regularization
- **AdaDelta**: Adaptive learning rate optimizer
- **AdaGrad**: Adaptive Gradient optimizer
- **RMSProp**: Root Mean Square Propagation optimizer
- **RProp**: Resilient Backpropagation optimizer
- **LBFGS**: Limited-memory BFGS optimizer

## Main Features

### Optimization Algorithm Diversity

- **First-order optimization**: Gradient-based optimization algorithms
- **Adaptive optimization**: Adjust learning rate according to parameter history
- **Regularization**: Support for regularization techniques such as weight decay
- **Momentum optimization**: Support for momentum and Nesterov momentum

### Configuration Management

- **Learning rate control**: Flexible learning rate settings
- **Weight decay**: L2 regularization parameters
- **Algorithm parameters**: Specific parameters for each optimizer
- **Type safety**: Configuration validation and type checking using Pydantic

## Usage Example

```python
from src._optimizer import UnionOptimizerParams, AdamParams

# Create Adam optimizer parameters
adam_params = AdamParams(
    lr=1e-3,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Use in configuration
config = SomeConfig(
    optimizer=adam_params
)
```
