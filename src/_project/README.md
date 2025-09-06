# Project Module

[English](README.md) | [中文](README_zh.md)

The project module focuses on organizing and executing the entire training project, unifying the management of all training components.

> **Note**: This module is a utility class (prefixed with `_`), following the project specifications for utility classes, commonly used but rarely modified. For more information, please refer to the [Project Architecture Specifications](../../docs/architecture.md).

## Directory Structure

```text
_project/
├── base/              # Base project management abstract classes
│   ├── project.py     # ProjectManager core class
│   └── config.py      # ProjectConfig configuration class
└── __init__.py        # Module initialization
```

## Core Components

### ProjectManager

Project Manager - The core class that integrates all training components:

```python
class ProjectManager:
    """
    Project Manager - The core class that integrates all components

    Design Philosophy:
    - Serves as the unified entry point for all training components
    - Directly uses PyTorch Lightning native Trainer
    - Manages all parameters through configuration without additional wrapping
    - Supports complete training lifecycle management
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
        # Core components
        self.model: Optional[LightningModule] = None
        self.datamodule: Optional[LightningDataModule] = None
        self.trainer: Optional[Trainer] = None
```

### ProjectConfig

Project-level configuration class - Unified management of the entire training project configuration:

```python
@dataclass
class ProjectConfig:
    """Project-level configuration class - Unified management of the entire training project configuration"""

    # Project metadata
    name: str = "training_project"
    description: str = ""
    version: str = "1.0.0"

    # Output settings
    output_dir: str = "./outputs"
    experiment_name: Optional[str] = None

    # Core component configurations
    model: UnionModelConfig
    datamodule: Optional[UnionDataModuleConfig] = None
    trainer: Optional[UnionTrainerConfig] = None
```

## Main Features

- **Unified Component Management**: Coordinates model, datamodule, trainer, callbacks, loggers, etc.
- **One-stop Training Interface**: Provides complete training, validation, and testing processes
- **Experiment Management**: Supports experiment management and reproduction
- **Configuration-driven**: Manages all parameters through configuration

## Usage Example

```python
from src._project.base.project import ProjectManager
from src._project.base.config import ProjectConfig

# Create project configuration
config = ProjectConfig(
    name="my_experiment",
    model=model_config,
    datamodule=datamodule_config,
    trainer=trainer_config
)

# Create project manager
project = ProjectManager(config)

# Execute training
project.fit()
```

## Extension Guide

To extend project management functionality:

1. Add new management methods in `base/project.py`
2. Add new configuration fields in `base/config.py`
3. Update relevant documentation and type annotations
4. Maintain compatibility with PyTorch Lightning
