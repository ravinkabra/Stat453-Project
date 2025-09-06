# LR Scheduler Module

[English](README.md) | [中文](README_zh.md)

学习率调度器模块专注于训练过程中的学习率调整策略，与训练逻辑解耦。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../architecture_zh.md)。

## 目录结构

```text
_lr_scheduler/
├── base/                      # 基础调度器参数抽象类
│   └── config.py              # BaseLRSchedulerParams 基类
├── step/                      # StepLR 调度器
│   └── config.py              # StepLRParams 参数类
├── multistep/                 # MultiStepLR 调度器
│   └── config.py              # MultiStepLRParams 参数类
├── exponential/               # ExponentialLR 调度器
│   └── config.py              # ExponentialLRParams 参数类
├── cosine/                    # CosineAnnealingLR 调度器
│   └── config.py              # CosineAnnealingLRParams 参数类
├── reduce_on_plateau/         # ReduceLROnPlateau 调度器
│   └── config.py              # ReduceLROnPlateauParams 参数类
├── cyclic/                    # CyclicLR 调度器
│   └── config.py              # CyclicLRParams 参数类
├── onecycle/                  # OneCycleLR 调度器
│   └── config.py              # OneCycleLRParams 参数类
├── cosine_warm_restarts/      # CosineAnnealingWarmRestarts 调度器
│   └── config.py              # CosineAnnealingWarmRestartsParams 参数类
└── __init__.py                # 模块初始化
```

## 核心组件

### BaseLRSchedulerParams

所有学习率调度器参数的基类：

```python
@dataclass
class BaseLRSchedulerParams:
    """学习率调度器参数基类"""

    _target_: str = Field("torch.optim.lr_scheduler.CosineAnnealingLR", description="Learning rate scheduler class")
    optimizer: Optional[Any] = Field(None, description="Optimizer object")
    interval: Literal["step", "epoch"] = Field("step", description="Interval for updating the learning rate")
    frequency: int = Field(1, description="Frequency of updating the learning rate")
    monitor: Optional[str] = Field(None, description="Metric to monitor for learning rate scheduling")
    strict: bool = Field(True, description="Whether to strictly enforce the configuration")
    name: Optional[str] = Field(None, description="Name of the learning rate scheduler")
    T_max: int = Field(10000, description="Total number of training steps")
    eta_min: float = Field(1e-6, description="Minimum learning rate")
```

### UnionLRSchedulerParams

统一的学习率调度器参数类型：

```python
UnionLRSchedulerParams = Union[
    BaseLRSchedulerParams,
    StepLRParams,
    MultiStepLRParams,
    ExponentialLRParams,
    CosineAnnealingLRParams,
    ReduceLROnPlateauParams,
    CyclicLRParams,
    OneCycleLRParams,
    CosineAnnealingWarmRestartsParams,
    dict,  # Allow dict for backward compatibility
]
```

### 支持的学习率调度器

- **StepLR**: 按固定步长衰减学习率
- **MultiStepLR**: 在指定里程碑处衰减学习率
- **ExponentialLR**: 指数衰减学习率
- **CosineAnnealingLR**: 余弦退火衰减
- **ReduceLROnPlateau**: 根据指标变化调整学习率
- **CyclicLR**: 循环学习率调度
- **OneCycleLR**: 单周期学习率调度
- **CosineAnnealingWarmRestarts**: 带重启的余弦退火

## 主要功能

### 调度策略多样性

- **固定调度**：按步长或里程碑调整学习率
- **自适应调度**：根据训练指标动态调整
- **周期调度**：循环或单周期学习率变化
- **指数衰减**：指数形式的衰减策略

### 配置灵活性

- **间隔控制**：支持按步长或轮次更新
- **频率设置**：可配置更新频率
- **监控指标**：支持基于指标的调度决策
- **参数验证**：严格的配置验证和类型检查

## 使用示例

```python
from src._lr_scheduler import UnionLRSchedulerParams, StepLRParams

# 创建 StepLR 调度器参数
step_params = StepLRParams(
    step_size=30,
    gamma=0.1,
    interval="epoch"
)

# 在配置中使用
config = SomeConfig(
    lr_scheduler=step_params
)
```
