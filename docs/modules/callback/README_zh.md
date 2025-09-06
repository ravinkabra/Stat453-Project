# Callback Module

[English](README.md) | [中文](README_zh.md)

回调模块专注于 PyTorch Lightning 训练过程中的回调功能配置，目前提供训练样本保存、模型检查点和早停等功能的配置支持。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../architecture_zh.md)。

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

### SampleSaverCallback

训练样本保存回调 - 在训练过程中自动保存关键样本数据：

```python
class SampleSaverCallback(Callback):
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

### 早停 (Early Stopping)

- **指标监控**：基于指定指标进行训练停止决策
- **耐心机制**：在指标不再改善时等待指定轮数
- **灵活阈值**：支持最小改善阈值设置
