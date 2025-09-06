# Callback Module

[English](README.md) | [中文](README_zh.md)

The callback module focuses on callback function configuration during PyTorch Lightning training processes, currently providing configuration support for training sample saving, model checkpointing, and early stopping functions.

> **Note**: This module is a tool class (prefixed with `_`), following the tool class definitions in the project specifications, commonly used but not frequently modified. For more information, please refer to the [Project Architecture Specification](../../architecture.md).

## Directory Structure

```text
_callback/
├── sample_saver/                  # Sample saving callback
│   ├── __init__.py               # Submodule initialization
│   ├── config.py                 # Sample saving configuration class
│   └── callback.py               # Sample saving callback implementation
├── checkpoint/                   # Model checkpoint configuration
│   ├── __init__.py               # Submodule initialization
│   └── config.py                 # Checkpoint configuration class
├── early_stopping/               # Early stopping configuration
│   ├── __init__.py               # Submodule initialization
│   └── config.py                 # Early stopping configuration class
├── __init__.py                   # Module initialization and type definitions
├── README_zh.md                  # Chinese documentation
└── README.md                     # English documentation
```

## Core Components

### SampleSaverCallback

Training sample saving callback - automatically saves key sample data during training:

```python
class SampleSaverCallback(Callback):
    """
    Training sample saving callback - used to save key content during training
    """
    # Complete callback implementation supporting multiple data format saving
```

### ModelCheckpointParams

Model checkpoint configuration class - encapsulates all configurations of PyTorch Lightning ModelCheckpoint:

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

Early stopping configuration class - encapsulates all configurations of PyTorch Lightning EarlyStopping:

```python
@dataclass
class EarlyStoppingParams:
    """EarlyStopping callback configuration class"""

    _target_: str = Field(
        default="pytorch_lightning.callbacks.EarlyStopping"
    )

    # Core monitoring configuration
    monitor: str = Field(description="Name of the metric to monitor")
    patience: int = Field(default=3, description="Patience value")
    mode: Literal["min", "max"] = Field(default="min")
    min_delta: Union[int, float] = Field(default=0.0)

    # More configuration options...
```

## Main Features

### Sample Saving (Sample Saver)

- **Multi-format support**: Supports image, text, tensor, json, custom five saving formats
- **Nested key access**: Supports complex data structure nested access, such as `["model_output", "logits"]`
- **Flexible configuration**: Each saving key can have independent frequency, sample count, stage settings
- **Directory structure**: Supports multiple directory organization methods (flat, epoch, step, epoch_step)

### Model Checkpoint (Model Checkpoint)

- **Multiple saving strategies**: Supports saving based on metrics, time intervals, training steps
- **Flexible filenames**: Supports formatted strings such as `best-{epoch:02d}-{val_loss:.2f}`
- **Automatic management**: Automatically saves the best N models, or regular backups

### Early Stopping (Early Stopping)

- **Metric monitoring**: Makes training stop decisions based on specified metrics
- **Patience mechanism**: Waits for a specified number of rounds when metrics no longer improve
- **Flexible threshold**: Supports minimum improvement threshold settings
