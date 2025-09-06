# Dataset Module

[English](README.md) | [中文](README_zh.md)

The dataset module focuses on unified management of data loading, preprocessing, and model input.

## Directory Structure

```bash
dataset/
├── base/              # Base dataset abstract classes
│   ├── dataset.py     # BaseDataset base class
│   ├── config.py      # Dataset configuration class
│   └── model_input.py # Model input definition
└── tutorial_mnist/    # MNIST tutorial implementation
    ├── dataset.py     # MNIST dataset implementation
    ├── config.py      # MNIST configuration
    └── model_input.py # MNIST model input
```

## Core Components

### BaseDataset

Base class for all datasets, inherits from `torch.utils.data.Dataset`:

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

Dataset configuration base class, some of which is used to construct dataloader, using Pydantic for type safety:

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

Base definition for model input, using DictAccessMixin to ensure dataclass can be accessed as dict for data:

```python
@dataclass
class BaseModelInput(DictAccessMixin):
    pass
```

## MNIST Dataset Implementation

For detailed MNIST dataset implementation, please refer to: [tutorial_mnist/README.md](tutorial_mnist/README.md)

It includes:

- Complete data loading and preprocessing pipeline
- Data augmentation functionality (random rotation, cropping)
- Subset sampling support (for quick experiments)
- Standardized model input format

## Main Features

### Data Pipeline Management

- **Unified interface**: Standardized dataset access interface
- **Preprocessing flow**: Complete data preprocessing and augmentation
- **Memory optimization**: Efficient data loading and caching mechanisms
- **Multi-process support**: Support for multi-process data loading

### Configuration Management

- **Batch configuration**: Flexible batch size settings
- **Worker processes**: Configurable data loading process count
- **Memory management**: Memory pinning and data shuffling control
- **Type safety**: Configuration validation using Pydantic

### Model Input Standardization

- **Unified format**: Standardized model input data structure
- **Dictionary access**: Support for dictionary-style data access
- **Type conversion**: Automatic handling of data type conversion
- **Extensibility**: Easy addition of new input fields

## Extension Guidelines

To add new datasets, follow these steps:

1. **Create new folder**: Create a new folder under `dataset/` (e.g., `cifar10/`)
2. **Implement dataset class**: Inherit from `BaseDataset` and implement necessary methods
3. **Configuration inheritance**: Inherit from `BaseDatasetConfig` and add specific configurations
4. **Define input format**: Create specific `ModelInput` data structure
5. **Implement data logic**: Complete data loading, preprocessing, and augmentation functionality
6. **Export update**: Export new dataset class in `__init__.py`
