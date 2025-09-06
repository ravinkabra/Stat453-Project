# Callback Module

[English](README.md) | [中文](README_zh.md)

The callback module focuses on PyTorch Lightning training callback configurations, currently providing configuration support for training sample saving, model checkpointing, and early stopping.

> **Note**: This module is a utility class (prefixed with `_`), following the project specifications for utility classes that are commonly used but rarely modified. For more information, please refer to [Project Architecture Specifications](../../docs/architecture.md).

## Directory Structure

```text
_callback/
├── sample_saver/                  # Sample saving callback
│   ├── __init__.py               # Submodule initialization
│   ├── config.py                 # Sample saving configuration classes
│   └── callback.py               # Sample saving callback implementation
├── checkpoint/                   # Model checkpoint configuration
│   ├── __init__.py               # Submodule initialization
│   └── config.py                 # Checkpoint configuration classes
├── early_stopping/               # Early stopping configuration
│   ├── __init__.py               # Submodule initialization
│   └── config.py                 # Early stopping configuration classes
├── __init__.py                   # Module initialization and type definitions
├── README_zh.md                  # Chinese documentation
└── README.md                     # English documentation
```

## Core Components

### SampleSaverCallback

Training sample saver callback - automatically saves key sample data during training:

```python
class SampleSaverCallback(Callback):
    """
    Training sample saver callback - saves key content during training
    """
    # Complete callback implementation supporting multiple data format saving
```

### ModelCheckpointParams

Model checkpoint configuration class - encapsulates all configurations for PyTorch Lightning ModelCheckpoint:

```python
@dataclass
class ModelCheckpointParams:
    """ModelCheckpoint callback configuration class"""

    _target_: str = Field(
        default="pytorch_lightning.callbacks.ModelCheckpoint"
    )

    # Basic saving configuration
    dirpath: Optional[str] = Field(default="./checkpoints")
    filename: Optional[str] = Field(default=None)
    monitor: Optional[str] = Field(default=None)
    save_top_k: Union[int, Literal[-1]] = Field(default=1)
    mode: Literal["min", "max"] = Field(default="min")

    # More configuration options...
```

### EarlyStoppingParams

Early stopping configuration class - encapsulates all configurations for PyTorch Lightning EarlyStopping:

```python
@dataclass
class EarlyStoppingParams:
    """EarlyStopping callback configuration class"""

    _target_: str = Field(
        default="pytorch_lightning.callbacks.EarlyStopping"
    )

    # Core monitoring configuration
    monitor: str = Field(description="Monitored metric name")
    patience: int = Field(default=3, description="Patience value")
    mode: Literal["min", "max"] = Field(default="min")
    min_delta: Union[int, float] = Field(default=0.0)

    # More configuration options...
```

## Main Features

### Sample Saving

- **Multiple Format Support**: Supports image, text, tensor, json, custom five save formats
- **Nested Key Access**: Supports complex data structure nested access like `["model_output", "logits"]`
- **Flexible Configuration**: Each save key can have independent frequency, sample count, phase settings
- **Directory Structure**: Supports multiple directory organization methods (flat, epoch, step, epoch_step)

### Model Checkpoint

- **Multiple Save Strategies**: Supports metric-based, time interval, training step based saving
- **Flexible Filenames**: Supports format strings like `best-{epoch:02d}-{val_loss:.2f}`
- **Auto Management**: Automatically saves best N models or periodic backups
- **Weight Separation**: Supports saving only weights or complete state

### Early Stopping

- **Multiple Monitor Modes**: Supports min monitoring (loss) and max monitoring (accuracy)
- **Flexible Stop Conditions**: Supports patience, minimum improvement threshold, stop threshold
- **Divergence Detection**: Supports detecting training divergence and early stopping
- **Custom Metrics**: Supports monitoring any training metric

## Usage Examples

### Basic Usage

```python
from src._callback import (
    SampleSaverCallback,
    ModelCheckpointParams,
    EarlyStoppingParams,
    UnionCallbackConfig
)

# 1. Sample saver callback
sample_saver = SampleSaverCallback(
    save_dir="./samples",
    save_keys={
        "generated_images": SaveKeyConfig(
            source="outputs",
            keys=["model_output", "generated"],
            format="image"
        )
    }
)

# 2. Model checkpoint configuration
checkpoint_config = ModelCheckpointParams(
    dirpath="./checkpoints",
    filename="best-{epoch:02d}-{val_loss:.2f}",
    monitor="val_loss",
    save_top_k=3,
    mode="min"
)

# 3. Early stopping configuration
early_stop_config = EarlyStoppingParams(
    monitor="val_loss",
    patience=10,
    mode="min",
    min_delta=0.001
)

# Create actual callbacks via instantiate
from hydra.utils import instantiate

checkpoint_callback = instantiate(checkpoint_config)
early_stop_callback = instantiate(early_stop_config)

# Use in training
trainer = Trainer(callbacks=[
    sample_saver,
    checkpoint_callback,
    early_stop_callback
])
```

### Using Predefined Configurations

```python
from src._callback.checkpoint.config import get_basic_checkpoint_config
from src._callback.early_stopping.config import get_conservative_early_stopping_config

# Use predefined configurations
checkpoint_config = get_basic_checkpoint_config()
early_stop_config = get_conservative_early_stopping_config()

# Customize modifications
checkpoint_config.dirpath = "./my_checkpoints"
early_stop_config.patience = 15
```

### Advanced Configuration Example

```python
# Complex checkpoint configuration
checkpoint_config = ModelCheckpointParams(
    dirpath="./checkpoints",
    filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
    monitor="val_loss",
    save_top_k=5,
    mode="min",
    save_last=True,
    every_n_epochs=10,  # Save every 10 epochs
    train_time_interval="01:00:00:00",  # Save every hour
    auto_insert_metric_name=False
)

# Accuracy-based early stopping
early_stop_config = EarlyStoppingParams(
    monitor="val_acc",
    patience=8,
    mode="max",
    min_delta=0.005,
    stopping_threshold=0.95  # Stop when accuracy reaches 95%
)
```

## Configuration Options Details

### ModelCheckpoint Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dirpath` | `str` | `"./checkpoints"` | Checkpoint save directory |
| `filename` | `str` | `None` | Filename template |
| `monitor` | `str` | `None` | Monitored metric |
| `save_top_k` | `int` | `1` | Save best k models |
| `mode` | `str` | `"min"` | Monitor mode |
| `every_n_epochs` | `int` | `None` | Save every N epochs |

### EarlyStopping Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `monitor` | `str` | - | Monitored metric |
| `patience` | `int` | `3` | Patience value |
| `mode` | `str` | `"min"` | Monitor mode |
| `min_delta` | `float` | `0.0` | Minimum improvement threshold |
| `stopping_threshold` | `float` | `None` | Stop threshold |

## Predefined Configuration Functions

### ModelCheckpoint

- `get_basic_checkpoint_config()` - Basic checkpoint configuration
- `get_comprehensive_checkpoint_config()` - Comprehensive checkpoint configuration
- `get_time_based_checkpoint_config()` - Time-based checkpoint configuration
- `get_weights_only_config()` - Weights-only saving configuration

### EarlyStopping

- `get_conservative_early_stopping_config()` - Conservative early stopping configuration
- `get_aggressive_early_stopping_config()` - Aggressive early stopping configuration
- `get_accuracy_based_early_stopping_config()` - Accuracy-based early stopping configuration
- `get_loss_based_early_stopping_config()` - Loss-based early stopping configuration

## Design Philosophy

1. **Configuration-Driven**: Manage all callback parameters through configuration classes, providing type safety and documentation
2. **Unified Interface**: All callback configurations follow the same pattern for easy management and extension
3. **No Reimplementation**: Don't reimplement PyTorch Lightning callbacks, but provide configuration encapsulation
4. **Flexible Customization**: Support passing additional parameters via `custom_*_kwargs`
5. **Predefined Configurations**: Provide predefined configurations for common scenarios to lower usage barrier

## Extension Guide

To add new callback configurations:

1. Create new folder under `_callback/` (like `new_callback/`)
2. Create `config.py` implementing configuration class inheriting from pydantic dataclass
3. Add `_target_` field pointing to actual PyTorch Lightning callback class
4. Add imports and Union type updates in `__init__.py`
5. Add predefined configuration functions
6. Update documentation and usage examples
