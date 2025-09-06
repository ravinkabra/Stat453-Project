# Metric Module

[English](README.md) | [中文](README_zh.md)

指标模块专注于深度学习训练过程中的指标计算、记录和管理，基于 PyTorch Lightning 和 TorchMetrics 构建，提供统一的指标配置和管理接口。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../architecture_zh.md)。

## 目录结构

```text
metric/
├── __init__.py                   # 模块初始化
├── base/                         # 基础指标配置
│   ├── __init__.py              # 子模块初始化
│   └── config.py                # 基础指标配置类
├── _manager/                     # 指标管理器
│   ├── __init__.py              # 子模块初始化
│   ├── config.py                # 指标管理器配置类
│   └── manager.py               # 指标管理器实现
├── value_recoder/                # 值记录器指标
│   ├── __init__.py              # 子模块初始化
│   ├── config.py                # 值记录器配置类
│   └── metric.py                # 值记录器指标实现
├── README_zh.md                  # 中文文档
└── README.md                     # 英文文档
```

## 核心组件

### BaseMetricParams

基础指标配置类 - 为所有指标提供统一的配置接口：

```python
@dataclass(config=ConfigDict(extra="allow"))
class BaseMetricParams:
    """基础指标配置类"""
    _target_: str = Field(default="torchmetrics.Metric")
    # 支持通过 extra="allow" 传递额外的配置参数
```

### MetricManager

指标管理器 - 统一管理所有训练指标：

```python
class MetricManager(torch.nn.Module):
    """
    指标管理器 - 管理指标集合的状态、计算和日志策略
    支持 DDP 分布式训练
    """
    # 双层 ModuleDict 结构：group_name -> metric_name -> Metric
```

### ValueRecorderMetric

值记录器指标 - 专门用于记录标量值的指标：

```python
class ValueRecorderMetric(Metric):
    """
    数值记录指标 - 记录和聚合训练过程中的标量值
    支持多种聚合方式：mean, sum, last
    """
```

## 主要功能

### 指标配置管理

- **多种配置方式**：支持配置驱动、直接实例化、字符串、字典四种配置方式
- **类型安全**：使用 Pydantic 提供完整的类型检查和验证
- **灵活扩展**：通过 `extra="allow"` 支持传递额外参数

### 指标分组管理

- **分组组织**：支持将相关指标组织到不同的分组中
- **独立配置**：每个分组可以有独立的日志配置
- **模块化管理**：使用 ModuleDict 保证 DDP 兼容性

### 日志配置管理

- **阶段控制**：支持 train/val/test 不同阶段的日志控制
- **频率控制**：支持更新频率和计算频率的独立配置
- **多目标日志**：支持进度条、日志记录器、按步/按周期日志

### 值记录功能

- **标量记录**：专门用于记录损失值、学习率等标量指标
- **聚合计算**：支持均值、求和、取最后一个值等多种聚合方式
- **自动类型转换**：自动处理 Tensor 和数值类型的转换

## 使用示例

### 基础使用

```python
from src.metric._manager import MetricManager, MetricManagerConfig
from src.metric.value_recoder import ValueRecorderParams

# 创建指标管理器配置
metric_config = MetricManagerConfig(
    groups={
        "train": {
            "loss": ValueRecorderParams(name="train_loss"),
            "accuracy": {"_target_": "torchmetrics.Accuracy", "task": "multiclass", "num_classes": 10}
        }
    }
)

# 创建指标管理器
metric_manager = MetricManager(metric_config)
```

### 高级配置

```python
# 配置日志策略
log_config = MetricLogConfig(
    on_step=True,
    on_epoch=True,
    prog_bar=True,
    logger=True
)

# 使用分组管理
metric_manager.log_group("train", {"loss": 0.5, "acc": 0.9}, log_config)
```
