# Logger Module

[English](README.md) | [中文](README_zh.md)

The logger module focuses on experiment logging and management, decoupled from training logic.

> **Note**: This module is a utility class (prefixed with `_`), following the project specifications for utility classes, commonly used but rarely modified. For more information, please refer to the [Project Architecture Specifications](../../docs/architecture.md).

## Directory Structure

```text
_logger/
├── base/              # Base logger parameters abstract classes
│   └── config.py      # BaseLoggerParams base class
├── wandb/             # Weights & Biases logger
│   └── config.py      # WandbLoggerParams parameter class
├── tensorboard/       # TensorBoard logger
│   └── config.py      # TensorBoardLoggerParams parameter class
├── csv/               # CSV format logger
│   └── config.py      # CSVLoggerParams parameter class
├── mlflow/            # MLflow logger
│   └── config.py      # MLFlowLoggerParams parameter class
├── neptune/           # Neptune logger
│   └── config.py      # NeptuneLoggerParams parameter class
├── config.py          # Module configuration and Union type definitions
└── __init__.py        # Module initialization
```

## Core Components

### BaseLoggerParams

The base class for all logger parameters:

```python
@dataclass
class BaseLoggerParams(DictAccessMixin):
    """Base class for logger parameters"""

    _target_: str = Field(..., description="Full path to Logger class")
    version: Optional[Union[int, str]] = Field(None, description="Experiment version")
    prefix: str = Field("", description="Log prefix")
```

### UnionLoggerParams

Unified logger parameter type:

```python
UnionLoggerParams = Union[
    WandbLoggerParams,
    TensorBoardLoggerParams,
    CSVLoggerParams,
    MLFlowLoggerParams,
    NeptuneLoggerParams,
]
```

### Supported Loggers

- **WandbLogger**: Weights & Biases experiment tracking
- **TensorBoardLogger**: TensorBoard visualization logging
- **CSVLogger**: CSV format local logging
- **MLFlowLogger**: MLflow experiment management
- **NeptuneLogger**: Neptune AI platform integration

## Usage Example

```python
from src._logger import UnionLoggerParams, WandbLoggerParams

# Create WandB logger parameters
wandb_params = WandbLoggerParams(
    project="my_project",
    name="experiment_1",
    save_dir="./logs"
)

# Use in configuration
config = SomeConfig(
    logger=wandb_params
)
```

## Extension Guide

To add support for a new logger:

1. Create a new folder under `_logger/` (e.g., `new_logger/`)
2. Create `config.py` implementing `NewLoggerParams` class inheriting from `BaseLoggerParams`
3. Add imports and update Union type in `__init__.py`
4. Update imports in main `config.py` (if needed)
5. Update `__all__` export list
