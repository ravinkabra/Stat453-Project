# Metric Module

[English](README.md) | [中文](README_zh.md)

The metric module focuses on metric calculation, recording, and management during deep learning training, built on PyTorch Lightning and TorchMetrics, providing unified metric configuration and management interfaces.

> **Note**: This module is a tool class (prefixed with `_`), following the tool class definitions in the project specifications, commonly used but not frequently modified. For more information, please refer to the [Project Architecture Specification](../../architecture.md).

## Directory Structure

```text
metric/
├── __init__.py                   # Module initialization
├── base/                         # Base metric configuration
│   ├── __init__.py              # Submodule initialization
│   └── config.py                # Base metric configuration class
├── _manager/                     # Metric manager
│   ├── __init__.py              # Submodule initialization
│   ├── config.py                # Metric manager configuration class
│   └── manager.py               # Metric manager implementation
├── value_recoder/                # Value recorder metric
│   ├── __init__.py              # Submodule initialization
│   ├── config.py                # Value recorder configuration class
│   └── metric.py                # Value recorder metric implementation
├── README_zh.md                  # Chinese documentation
└── README.md                     # English documentation
```

## Core Components

### BaseMetricParams

Base metric configuration class - provides unified configuration interfaces for all metrics:

```python
@dataclass(config=ConfigDict(extra="allow"))
class BaseMetricParams:
    """Base metric configuration class"""
    _target_: str = Field(default="torchmetrics.Metric")
    # Support passing additional configuration parameters through extra="allow"
```

### MetricManager

Metric manager - unifies management of all training metrics:

```python
class MetricManager(torch.nn.Module):
    """
    Metric manager - manages metric collection state, calculation, and logging strategies
    Supports DDP distributed training
    """
    # Two-layer ModuleDict structure: group_name -> metric_name -> Metric
```

### ValueRecorderMetric

Value recorder metric - specialized for recording scalar values:

```python
class ValueRecorderMetric(Metric):
    """
    Numerical recording metric - records and aggregates scalar values during training
    Supports multiple aggregation methods: mean, sum, last
    """
```

## Main Features

### Metric Configuration Management

- **Multiple configuration methods**: Support configuration-driven, direct instantiation, string, dictionary four configuration methods
- **Type safety**: Use Pydantic to provide complete type checking and validation
- **Flexible extension**: Support passing additional parameters through `extra="allow"`

### Metric Grouping Management

- **Grouping organization**: Support organizing related metrics into different groups
- **Independent configuration**: Each group can have independent logging configuration
- **Modular management**: Use ModuleDict to ensure DDP compatibility

### Logging Configuration Management

- **Stage control**: Support train/val/test different stage logging control
- **Frequency control**: Support independent configuration of update frequency and calculation frequency
- **Multi-target logging**: Support progress bar, logger, per-step/per-epoch logging

### Value Recording Functionality

- **Scalar recording**: Specialized for recording loss values, learning rates and other scalar metrics
- **Aggregation calculation**: Support mean, sum, take last value and other aggregation methods
- **Automatic type conversion**: Automatically handle Tensor and numerical type conversion

## Usage Example

### Basic Usage

```python
from src.metric._manager import MetricManager, MetricManagerConfig
from src.metric.value_recoder import ValueRecorderParams

# Create metric manager configuration
metric_config = MetricManagerConfig(
    groups={
        "train": {
            "loss": ValueRecorderParams(name="train_loss"),
            "accuracy": {"_target_": "torchmetrics.Accuracy", "task": "multiclass", "num_classes": 10}
        }
    }
)

# Create metric manager
metric_manager = MetricManager(metric_config)
```

### Advanced Configuration

```python
# Configure logging strategy
log_config = MetricLogConfig(
    on_step=True,
    on_epoch=True,
    prog_bar=True,
    logger=True
)

# Use group management
metric_manager.log_group("train", {"loss": 0.5, "acc": 0.9}, log_config)
```
