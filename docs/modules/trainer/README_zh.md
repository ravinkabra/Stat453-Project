# Trainer Module

[English](README.md) | [中文](README_zh.md)

训练器模块专注于 PyTorch Lightning Trainer 的配置和管理，统一管理训练过程的所有参数。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../architecture_zh.md)。

## 目录结构

```text
_trainer/
├── base/              # 基础训练器配置抽象类
│   ├── __init__.py    # 模块初始化
│   └── config.py      # BaseTrainerConfig 配置类
└── __init__.py        # 模块初始化和类型定义
```

## 核心组件

### BaseTrainerConfig

训练器配置基类 - 封装 PyTorch Lightning Trainer 的所有配置参数：

```python
@dataclass(config=ConfigDict(extra="allow"))
class BaseTrainerConfig:
    """训练器配置基类 - 封装 PyTorch Lightning Trainer 的所有配置参数"""

    # 硬件和加速器配置
    accelerator: Union[str, Any] = Field(
        default="auto",
        description="加速器类型 ('cpu', 'gpu', 'tpu', 'hpu', 'mps', 'auto')",
    )
    strategy: Union[str, Any] = Field(default="auto", description="训练策略")
    devices: Union[List[int], str, int] = Field(
        default="auto", description="使用的设备"
    )
    num_nodes: int = Field(default=1, description="分布式训练的GPU节点数量")

    # 精度配置
    precision: Union[...] = Field(default="32-true", description="训练精度")

    # 日志和记录器
    logger: Union[Any, List[Any], bool, None] = Field(
        default=True, description="日志记录器"
    )

    # 训练控制
    max_epochs: Optional[int] = Field(default=None, description="最大训练轮数")
    max_steps: int = Field(default=-1, description="最大训练步数")
    min_steps: Optional[int] = Field(default=None, description="最小训练步数")

    # 验证控制
    val_check_interval: Union[int, float, None] = Field(
        default=1.0, description="验证检查间隔"
    )
    check_val_every_n_epoch: Optional[int] = Field(
        default=1, description="每N个epoch进行一次验证"
    )

    # 梯度控制
    accumulate_grad_batches: int = Field(default=1, description="梯度累积批次数")
    gradient_clip_val: Union[int, float, None] = Field(
        default=None, description="梯度裁剪值"
    )
```

### UnionTrainerConfig

统一的训练器配置类型：

```python
UnionTrainerConfig = Union[BaseTrainerConfig, dict]
```

## 主要功能

### 硬件配置

- **多加速器支持**：支持 CPU、GPU、TPU、HPU、MPS
- **分布式训练**：支持多种分布式策略和多节点训练
- **自动检测**：自动检测和配置可用硬件
- **混合精度**：支持多种精度训练模式

### 训练控制

- **灵活轮次**：可配置的最大/最小训练轮次和步数
- **验证调度**：灵活的验证频率和批次限制
- **梯度优化**：梯度累积和裁剪控制
- **性能调优**：确定性模式和基准测试

### 日志和监控

- **多日志器集成**：支持多种日志记录器
- **进度监控**：训练进度和性能指标监控
- **调试支持**：快速开发运行和过拟合测试
- **异常检测**：自动异常检测和处理

## 使用示例

```python
from src._trainer import UnionTrainerConfig, BaseTrainerConfig

# 创建训练器配置
trainer_config = BaseTrainerConfig(
    accelerator="gpu",
    devices=1,
    max_epochs=100,
    precision="16-mixed"
)

# 在配置中使用
config = SomeConfig(
    trainer=trainer_config
)
```
