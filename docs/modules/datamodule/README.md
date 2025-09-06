# DataModule Module

[English](README.md) | [中文](README_zh.md)

The datamodule focuses on data loading and preparation, decoupled from training logic.

> **Note**: This module is a tool class (prefixed with `_`), following the tool class definitions in the project specifications, commonly used but not frequently modified. For more information, please refer to the [Project Architecture Specification](../../architecture.md).

## Directory Structure

```text
_datamodule/
├── base/              # Base datamodule abstract classes
│   ├── datamodule.py  # BaseDataModule base class
│   └── config.py      # Datamodule configuration class
└── __init__.py        # Module initialization
```

## Core Components

### BaseDataModule

Base class for all datamodules, inherits from PyTorch Lightning's LightningDataModule, providing unified data loading interfaces:

```python
class BaseDataModule(LightningDataModule, ABC):
    def __init__(self, config: BaseDataModuleConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        """Set up datamodule for training/validation/testing"""
        pass

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader"""
        pass
```

### BaseDataModuleConfig

Datamodule configuration base class:

```python
@dataclass
class BaseDataModuleConfig:
    """Datamodule configuration base class"""

    _target_: Literal["src._datamodule.base.datamodule.BaseDataModule"]
    train: Optional[UnionDatasetConfig] = None
    val: Optional[UnionDatasetConfig] = None
    test: Optional[UnionDatasetConfig] = None
    predict: Optional[UnionDatasetConfig] = None
```

## Main Features

### Data Pipeline Management

- **Unified interface**: Standardized data loading and preparation processes
- **Stage separation**: Supports different stages of training, validation, testing, prediction
- **Configuration-driven**: Manages all data parameters through configuration classes

### DataLoader Configuration

- **Batch size**: Configurable batch processing size
- **Worker processes**: Support for multi-process data loading
- **Memory optimization**: Support for memory pinning and data shuffling

## Extension Guidelines

To add new datamodule implementations:

1. Create a new folder under `_datamodule/`
2. Implement datamodule class inheriting from `BaseDataModule` (not mandatory)
3. Create corresponding configuration class inheriting from `BaseDataModuleConfig`
4. Update `__init__.py` to export the new datamodule
