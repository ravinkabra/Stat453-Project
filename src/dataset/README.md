# Dataset Module

> 📖 [中文](README_zh.md) | [English](README.md)

The dataset module focuses on unified management of data loading, preprocessing, and model input.

## Directory Structure

```bash
dataset/
├── base/              # Base dataset abstract classes
│   ├── dataset.py     # BaseDataset base class
│   ├── config.py      # Dataset configuration classes
│   └── model_input.py # Model input definitions
└── tutorial_mnist/    # MNIST tutorial implementation
    ├── dataset.py     # MNIST dataset implementation
    ├── config.py      # MNIST configuration
    └── model_input.py # MNIST model input
```

## Core Components

### BaseDataset

The base class for all datasets, inheriting from `torch.utils.data.Dataset`:

```python
class BaseDataset(Dataset, ABC):
    def __init__(self, config: BaseDatasetConfig):
        self.config = config

    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> BaseModelInput:
        pass
```

### BaseDatasetConfig

The base configuration class for datasets, where part of it is used to construct the dataloader, using Pydantic for type safety:

```python
@dataclass
class BaseDatasetConfig:
    _target_: str = "src.dataset.base.dataset.BaseDataset"
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = False
```

### BaseModelInput

The base definition for model input, using DictAccessMixin to ensure dataclass can be accessed as dict simultaneously:

```python
@dataclass
class BaseModelInput(DictAccessMixin):
    pass
```

## MNIST Dataset Implementation

For detailed MNIST dataset implementation, please refer to: [tutorial_mnist/README.md](tutorial_mnist/README.md)

This implementation includes:

- Complete data loading and preprocessing pipeline
- Data augmentation features (random rotation, cropping)
- Subset sampling support (for quick experiments)
- Standardized model input format

## Extension Guide

To add a new dataset, follow these steps:

1. **Create New Folder**: Create a new folder under `dataset/` (e.g., `cifar10/`)
2. **Implement Dataset Class**: Inherit from `BaseDataset` and implement required methods
3. **Configuration Class Inheritance**: Inherit from `BaseDatasetConfig` and add specific configurations
4. **Define Input Format**: Create specific `ModelInput` data structure
5. **Implement Data Logic**: Complete data loading, preprocessing, and augmentation functionality
6. **Update Exports**: Export the new dataset class in `__init__.py`
