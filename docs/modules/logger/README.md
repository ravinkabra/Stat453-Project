# Logger Module

[English](README.md) | [中文](README_zh.md)

The logger module focuses on experiment logging and management, decoupled from training logic.

> **Note**: This module is a tool class (prefixed with `_`), following the tool class definitions in the project specifications, commonly used but not frequently modified. For more information, please refer to the [Project Architecture Specification](../../architecture.md).

## Directory Structure

```text
_logger/
├── base/              # Base logger parameter abstract classes
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

Base class for all logger parameters:

```python
@dataclass
class BaseLoggerParams(DictAccessMixin):
    """Logger parameters base class"""

    _target_: str = Field(..., description="Complete path to Logger class")
    version: Optional[Union[int, str]] = Field(None, description="Experiment version")
    prefix: str = Field("", description="Log prefix")
```

### UnionLoggerParams

Unified logger parameters type:

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

## Main Features

### Multi-Platform Support

- **Experiment tracking**: Complete experiment lifecycle management
- **Metric recording**: Automatic recording of training metrics and hyperparameters
- **Visualization**: Support for multiple visualization tools integration
- **Version control**: Experiment version management and comparison

### Configuration Flexibility

- **Unified interface**: Standardized logger configuration interface
- **Type safety**: Complete type checking and validation using Pydantic
- **Extensibility**: Easy addition of new logger support

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

## Extension Guidelines

To add new logger support:

1. Create a new folder under `_logger/` (e.g., `new_logger/`)
2. Create `config.py` implementing `NewLoggerParams` class inheriting from `BaseLoggerParams`
3. Add imports and Union type updates in `__init__.py`
4. Update main `config.py` imports if needed
5. Update `__all__` export list
