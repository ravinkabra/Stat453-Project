# DataModule Module

[English](README.md) | [中文](README_zh.md)

The data module focuses on data loading and preparation, decoupled from training logic.

> **Note**: This module is a utility class (prefixed with `_`), following the project specifications for utility classes, commonly used but rarely modified. For more information, please refer to the [Project Architecture Specifications](../../docs/architecture.md).

## Directory Structure

```text
_datamodule/
├── base/              # Base data module abstract classes
│   ├── datamodule.py  # BaseDataModule base class
│   └── config.py      # Data module configuration class
└── __init__.py        # Module initialization
```

## Core Components

### BaseDataModule

The base class for all data modules, inheriting from PyTorch Lightning's LightningDataModule, providing a unified data loading interface:

```python
class BaseDataModule(LightningDataModule, ABC):
    def __init__(self, config: BaseDataModuleConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        """Setup the data module for training/validation/testing"""
        pass

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader"""
        pass
```

### BaseDataModuleConfig

The base configuration class for data modules:

```python
@dataclass
class BaseDataModuleConfig:
    """Base configuration class for data modules"""

    _target_: Literal["src._datamodule.base.datamodule.BaseDataModule"]
    train: Optional[UnionDatasetConfig] = None
    val: Optional[UnionDatasetConfig] = None
    test: Optional[UnionDatasetConfig] = None
    predict: Optional[UnionDatasetConfig] = None
```

## Extension Guide

To add a new data module implementation:

1. Create a new folder under `_datamodule/`
2. Implement the data module class inheriting from `BaseDataModule` (not mandatory)
3. Create a corresponding configuration class inheriting from `BaseDataModuleConfig`
4. Update `__init__.py` to export the new data module
