# Callback Module

[English](README.md) | [中文](README_zh.md)

回调模块专注于 PyTorch Lightning 训练过程中的回调功能配置，目前提供训练样本保存、模型检查点和早停等功能的配置支持。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../docs/architecture_zh.md)。

## 目录结构

```text
_callback/
├── sample_saver/                  # 样本保存回调
│   ├── __init__.py               # 子模块初始化
│   ├── config.py                 # 样本保存配置类
│   └── callback.py               # 样本保存回调实现
├── checkpoint/                   # 模型检查点配置
│   ├── __init__.py               # 子模块初始化
│   └── config.py                 # 检查点配置类
├── early_stopping/               # 早停配置
│   ├── __init__.py               # 子模块初始化
│   └── config.py                 # 早停配置类
├── __init__.py                   # 模块初始化和类型定义
├── README_zh.md                  # 中文文档
└── README.md                     # 英文文档
```

## 核心组件

### TrainingSampleSaverCallback

训练样本保存回调 - 在训练过程中自动保存关键样本数据：

```python
class TrainingSampleSaverCallback(Callback):
    """
    训练样本保存回调 - 用于在训练过程中保存关键内容
    """
    # 完整的回调实现，支持多种数据格式保存
```

### ModelCheckpointParams

模型检查点配置类 - 封装 PyTorch Lightning ModelCheckpoint 的所有配置：

```python
@dataclass
class ModelCheckpointParams:
    """ModelCheckpoint 回调配置类"""

    _target_: str = Field(
        default="pytorch_lightning.callbacks.ModelCheckpoint"
    )

    # 基本保存配置
    dirpath: Optional[str] = Field(default="./checkpoints")
    filename: Optional[str] = Field(default=None)
    monitor: Optional[str] = Field(default=None)
    save_top_k: Union[int, Literal[-1]] = Field(default=1)
    mode: Literal["min", "max"] = Field(default="min")

    # 更多配置选项...
```

### EarlyStoppingParams

早停配置类 - 封装 PyTorch Lightning EarlyStopping 的所有配置：

```python
@dataclass
class EarlyStoppingParams:
    """EarlyStopping 回调配置类"""

    _target_: str = Field(
        default="pytorch_lightning.callbacks.EarlyStopping"
    )

    # 核心监控配置
    monitor: str = Field(description="监控的指标名称")
    patience: int = Field(default=3, description="耐心值")
    mode: Literal["min", "max"] = Field(default="min")
    min_delta: Union[int, float] = Field(default=0.0)

    # 更多配置选项...
```

## 主要功能

### 样本保存 (Sample Saver)

- **多格式支持**：支持 image、text、tensor、json、custom 五种保存格式
- **嵌套键访问**：支持复杂的数据结构嵌套访问，如 `["model_output", "logits"]`
- **灵活配置**：每个保存键可以有独立的频率、样本数、阶段设置
- **目录结构**：支持多种目录组织方式（flat、epoch、step、epoch_step）

### 模型检查点 (Model Checkpoint)

- **多种保存策略**：支持基于指标、时间间隔、训练步骤的保存
- **灵活文件名**：支持格式化字符串，如 `best-{epoch:02d}-{val_loss:.2f}`
- **自动管理**：自动保存最好的 N 个模型，或定期备份
- **权重分离**：支持只保存权重或完整状态

### 早停 (Early Stopping)

- **多种监控模式**：支持最小值监控（损失）和最大值监控（准确率）
- **灵活停止条件**：支持耐心值、最小改善阈值、停止阈值
- **发散检测**：支持检测训练发散并提前停止
- **自定义指标**：支持监控任何训练指标

## 使用示例

### 基础使用

```python
from src._callback import (
    TrainingSampleSaverCallback,
    ModelCheckpointParams,
    EarlyStoppingParams,
    UnionCallbackConfig
)

# 1. 样本保存回调
sample_saver = TrainingSampleSaverCallback(
    save_dir="./training_samples",
    save_keys={
        "generated_images": SaveKeyConfig(
            source="outputs",
            keys=["model_output", "generated"],
            format="image"
        )
    }
)

# 2. 模型检查点配置
checkpoint_config = ModelCheckpointParams(
    dirpath="./checkpoints",
    filename="best-{epoch:02d}-{val_loss:.2f}",
    monitor="val_loss",
    save_top_k=3,
    mode="min"
)

# 3. 早停配置
early_stop_config = EarlyStoppingParams(
    monitor="val_loss",
    patience=10,
    mode="min",
    min_delta=0.001
)

# 通过 instantiate 创建实际的回调
from hydra.utils import instantiate

checkpoint_callback = instantiate(checkpoint_config)
early_stop_callback = instantiate(early_stop_config)

# 在训练中使用
trainer = Trainer(callbacks=[
    sample_saver,
    checkpoint_callback,
    early_stop_callback
])
```

### 使用预定义配置

```python
from src._callback.checkpoint.config import get_basic_checkpoint_config
from src._callback.early_stopping.config import get_conservative_early_stopping_config

# 使用预定义配置
checkpoint_config = get_basic_checkpoint_config()
early_stop_config = get_conservative_early_stopping_config()

# 自定义修改
checkpoint_config.dirpath = "./my_checkpoints"
early_stop_config.patience = 15
```

### 高级配置示例

```python
# 复杂的检查点配置
checkpoint_config = ModelCheckpointParams(
    dirpath="./checkpoints",
    filename="{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}",
    monitor="val_loss",
    save_top_k=5,
    mode="min",
    save_last=True,
    every_n_epochs=10,  # 每10个epoch保存一次
    train_time_interval="01:00:00:00",  # 每小时保存一次
    auto_insert_metric_name=False
)

# 基于准确率的早停
early_stop_config = EarlyStoppingParams(
    monitor="val_acc",
    patience=8,
    mode="max",
    min_delta=0.005,
    stopping_threshold=0.95  # 达到95%准确率时停止
)
```

## 配置选项详解

### ModelCheckpoint 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dirpath` | `str` | `"./checkpoints"` | 检查点保存目录 |
| `filename` | `str` | `None` | 文件名模板 |
| `monitor` | `str` | `None` | 监控指标 |
| `save_top_k` | `int` | `1` | 保存最好的 k 个模型 |
| `mode` | `str` | `"min"` | 监控模式 |
| `every_n_epochs` | `int` | `None` | 每 N 个 epoch 保存 |

### EarlyStopping 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `monitor` | `str` | - | 监控指标 |
| `patience` | `int` | `3` | 耐心值 |
| `mode` | `str` | `"min"` | 监控模式 |
| `min_delta` | `float` | `0.0` | 最小改善阈值 |
| `stopping_threshold` | `float` | `None` | 停止阈值 |

## 预定义配置函数

### ModelCheckpoint

- `get_basic_checkpoint_config()` - 基本的检查点配置
- `get_comprehensive_checkpoint_config()` - 全面的检查点配置
- `get_time_based_checkpoint_config()` - 基于时间的检查点配置
- `get_weights_only_config()` - 仅权重保存配置

### EarlyStopping

- `get_conservative_early_stopping_config()` - 保守的早停配置
- `get_aggressive_early_stopping_config()` - 激进的早停配置
- `get_accuracy_based_early_stopping_config()` - 基于准确率的早停配置
- `get_loss_based_early_stopping_config()` - 基于损失的早停配置

## 设计理念

1. **配置驱动**：通过配置类管理所有回调参数，提供类型安全和文档
2. **统一接口**：所有回调配置都遵循相同的模式，便于管理和扩展
3. **不重复实现**：不重新实现 PyTorch Lightning 的回调，而是提供配置封装
4. **灵活定制**：支持通过 `custom_*_kwargs` 传递额外参数
5. **预定义配置**：提供常用场景的预定义配置，降低使用门槛

## 扩展指南

要添加新的回调配置：

1. 在 `_callback/` 下创建新文件夹（如 `new_callback/`）
2. 创建 `config.py` 实现配置类，继承 pydantic dataclass
3. 添加 `_target_` 字段指向实际的 PyTorch Lightning 回调类
4. 在 `__init__.py` 中添加导入和 Union 类型更新
5. 添加预定义配置函数
6. 更新文档和使用示例
