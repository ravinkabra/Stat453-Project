# Metric Module

[English](README.md) | [中文](README_zh.md)

The metric module focuses on metric calculation, recording, and management during deep learning training processes, built on PyTorch Lightning and TorchMetrics, providing unified metric configuration and management interfaces.

> **Note**: This module is a utility class (prefixed with `_`), following the project specifications for utility classes that are commonly used but rarely modified. For more information, please refer to [Project Architecture Specifications](../../docs/architecture.md).

## Directory Structure

```text
metric/
├── __init__.py                   # Module initialization
├── base/                         # Base metric configuration
│   ├── __init__.py              # Submodule initialization
│   └── config.py                # Base metric configuration classes
├── _manager/                     # Metric manager
│   ├── __init__.py              # Submodule initialization
│   ├── config.py                # Metric manager configuration classes
│   └── manager.py               # Metric manager implementation
├── value_recoder/                # Value recorder metric
│   ├── __init__.py              # Submodule initialization
│   ├── config.py                # Value recorder configuration classes
│   └── metric.py                # Value recorder metric implementation
├── README_zh.md                  # Chinese documentation
└── README.md                     # English documentation
```

## Core Components

### BaseMetricParams

Base metric configuration class - provides unified configuration interface for all metrics:

```python
@dataclass(config=ConfigDict(extra="allow"))
class BaseMetricParams:
    """Base metric configuration class"""
    _target_: str = Field(default="torchmetrics.Metric")
    # Supports passing additional configuration parameters via extra="allow"
```

### MetricManager

Metric manager - unified management of all training metrics:

```python
class MetricManager(torch.nn.Module):
    """
    Metric manager - manages metric collection state, computation, and logging strategy
    Supports DDP distributed training
    """
    # Double-layer ModuleDict structure: group_name -> metric_name -> Metric
```

### ValueRecorderMetric

Value recorder metric - specialized metric for recording scalar values:

```python
class ValueRecorderMetric(Metric):
    """
    Value recorder metric - records and aggregates scalar values during training
    Supports multiple aggregation methods: mean, sum, last
    """
```

## Main Features

### Metric Configuration Management

- **Multiple Configuration Methods**: Supports configuration-driven, direct instantiation, string, and dictionary four configuration methods
- **Type Safety**: Uses Pydantic to provide complete type checking and validation
- **Flexible Extension**: Supports passing additional parameters via `extra="allow"`

### Metric Group Management

- **Group Organization**: Supports organizing related metrics into different groups
- **Independent Configuration**: Each group can have independent logging configuration
- **Modular Management**: Uses ModuleDict to ensure DDP compatibility

### Logging Configuration Management

- **Phase Control**: Supports train/val/test different phase logging control
- **Frequency Control**: Supports independent configuration of update and compute frequencies
- **Multi-target Logging**: Supports progress bar, logger, step-by-step/epoch logging

### Value Recording Function

- **Scalar Recording**: Specialized for recording loss values, learning rates, and other scalar metrics
- **Aggregation Calculation**: Supports mean, sum, last value, and other aggregation methods
- **Automatic Type Conversion**: Automatically handles Tensor and numeric type conversion

## Usage Examples

### Basic Usage

```python
from src.metric._manager import MetricManager, MetricManagerConfig
from src.metric.value_recoder import ValueRecorderParams
from hydra.utils import instantiate

# 1. Configure metric manager
config = MetricManagerConfig(
    metrics={
        "prediction_metrics": {
            "metrics": {
                "train_accuracy": "torchmetrics.Accuracy",
                "val_accuracy": "torchmetrics.Accuracy",
                "precision": "torchmetrics.Precision",
            },
            "log_config": {
                "phase": ["train", "val"],
                "prog_bar": True,
                "logger": True,
                "on_step": True,
                "on_epoch": True,
            }
        },
        "value_recorders": {
            "metrics": {
                "loss": ValueRecorderParams(aggregation="mean"),
                "learning_rate": ValueRecorderParams(aggregation="last"),
            },
            "log_config": {
                "phase": ["train", "val"],
                "update_frequency": 1,
                "prog_bar": False,
                "logger": True,
            }
        }
    }
)

# 2. Instantiate metric manager
metric_manager = instantiate(config)

# 3. Use in training
# Update metrics
metric_manager.update("prediction_metrics", "train_accuracy", preds, target)
metric_manager.update("value_recorders", "loss", loss_value)

# Compute and log metrics
metric_manager.log_metrics(pl_module, current_epoch, batch_idx, phase="train")
```

### Advanced Configuration Example

```python
# Complex metric configuration
config = MetricManagerConfig(
    metrics={
        "classification_metrics": {
            "metrics": {
                "accuracy": {
                    "_target_": "torchmetrics.Accuracy",
                    "task": "multiclass",
                    "num_classes": 10,
                },
                "f1_score": {
                    "_target_": "torchmetrics.F1Score",
                    "task": "multiclass",
                    "num_classes": 10,
                    "average": "macro",
                },
            },
            "log_config": {
                "phase": ["val", "test"],
                "update_frequency": 1,
                "compute_frequency": 10,  # Compute every 10 steps
                "prog_bar": True,
                "on_step": False,  # Don't log per step in validation phase
                "on_epoch": True,
                "sync_dist": True,
            }
        },
        "regression_metrics": {
            "metrics": {
                "mse": "torchmetrics.MeanSquaredError",
                "mae": "torchmetrics.MeanAbsoluteError",
            },
            "log_config": {
                "phase": ["train", "val"],
                "reduce_fx": "mean",
            }
        }
    }
)
```

### Value Recorder Usage

```python
from src.metric.value_recoder import ValueRecorderParams

# Configure value recorder
loss_recorder = ValueRecorderParams(
    aggregation="mean",  # Mean aggregation
)

lr_recorder = ValueRecorderParams(
    aggregation="last",  # Record only the last value
)

# Use in training loop
for batch in train_loader:
    # ... training code ...

    # Record loss
    loss_recorder.update(loss.item())

    # Record learning rate
    lr_recorder.update(current_lr)

# Compute aggregated results
avg_loss = loss_recorder.compute()
current_lr = lr_recorder.compute()
```

## Configuration Options Details

### MetricLogConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `phase` | `List[str]` | `["train", "val", "test"]` | Logging phases |
| `update_frequency` | `int` | `1` | Update frequency |
| `compute_frequency` | `int` | `None` | Compute frequency, must be multiple of update_frequency |
| `prog_bar` | `bool` | `False` | Whether to display in progress bar |
| `logger` | `bool` | `True` | Whether to log to logger |
| `on_step` | `bool` | `True` | Whether to log per step |
| `on_epoch` | `bool` | `True` | Whether to log per epoch |
| `sync_dist` | `bool` | `True` | Whether to synchronize in distributed training |
| `reduce_fx` | `str` | `"mean"` | Reduction function |

### ValueRecorderParams

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregation` | `str` | `"mean"` | Aggregation method: mean/sum/last |

## Design Philosophy

1. **Unified Metric Management**: Unified management of all metrics through MetricManager, providing consistent interface
2. **Flexible Configuration Methods**: Supports multiple metric configuration methods to adapt to different use cases
3. **DDP Compatibility**: Based on PyTorch ModuleDict to ensure distributed training compatibility
4. **Type Safety**: Uses Pydantic to provide complete configuration validation
5. **Modular Design**: Metric group management for easy organization and maintenance

## Extension Guide

To add new metric types:

1. Define new configuration class in `base/config.py`, inheriting from `BaseMetricParams`
2. Implement specific metric class, inheriting from `torchmetrics.Metric`
3. Set correct `_target_` field in configuration
4. Update documentation and usage examples

## Best Practices

1. **Group Organization**: Organize related metrics into the same group for easier management
2. **Frequency Configuration**: Reasonably set update and compute frequencies to avoid performance overhead
3. **Phase Control**: Select appropriate logging phases based on metric purposes
4. **Distributed Synchronization**: Keep `sync_dist=True` in distributed training
5. **Aggregation Selection**: Choose appropriate aggregation method based on metric characteristics
