# Project Module

[English](README.md) | [中文](README_zh.md)

The project module focuses on the organization and execution of the entire training project, unifying management of all training components.

> **Note**: This module is a tool class (prefixed with `_`), following the tool class definitions in the project specifications, commonly used but not frequently modified. For more information, please refer to the [Project Architecture Specification](../../architecture.md).

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

Project manager - the core class that integrates all training components:

```python
class ProjectManager:
    """
    Project manager - the core class that integrates all components

    Design philosophy:
    - Serves as the unified entry point for all training components
    - Directly uses native PyTorch Lightning Trainer
    - Manages all parameters through configuration without additional packaging
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

Project-level configuration class - unifies management of the entire training project's configuration:

```python
@dataclass
class ProjectConfig:
    """Project-level configuration class - unifies management of the entire training project's configuration"""

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

### Unified Component Management

- **Component coordination**: Coordinates model, datamodule, trainer, callbacks, loggers, etc.
- **Lifecycle management**: Complete training, validation, testing processes
- **Configuration-driven**: Manages all parameters through configuration
- **Modular design**: Supports flexible component combinations

### One-Stop Training Interface

- **Simplified process**: Unified training entry point
- **Automatic configuration**: Automatically handles dependencies between components
- **State management**: Maintains state information during training
- **Error handling**: Unified exception handling mechanism

### Experiment Management

- **Experiment tracking**: Records experiment configurations and results
- **Version control**: Supports experiment version management and reproduction
- **Result management**: Automatically saves and organizes training results
- **Comparative analysis**: Supports multi-experiment comparison and analysis

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

## Extension Guidelines

To extend project management functionality:

1. Add new management methods in `ProjectManager`
2. Extend `ProjectConfig` to support new configuration options
3. Add new lifecycle management hooks
4. Update component integration logic
