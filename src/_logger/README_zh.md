# Logger Module

[English](README.md) | [中文](README_zh.md)

日志器模块专注于实验日志记录和管理，与训练逻辑解耦。

> **注意**: 此模块为工具类（以 `_` 开头），遵循项目规范中工具类的定义，常用但不常修改。如需了解更多，请参考 [项目架构规范](../../docs/architecture_zh.md)。

## 目录结构

```text
_logger/
├── base/              # 基础日志器参数抽象类
│   └── config.py      # BaseLoggerParams 基类
├── wandb/             # Weights & Biases 日志器
│   └── config.py      # WandbLoggerParams 参数类
├── tensorboard/       # TensorBoard 日志器
│   └── config.py      # TensorBoardLoggerParams 参数类
├── csv/               # CSV 格式日志器
│   └── config.py      # CSVLoggerParams 参数类
├── mlflow/            # MLflow 日志器
│   └── config.py      # MLFlowLoggerParams 参数类
├── neptune/           # Neptune 日志器
│   └── config.py      # NeptuneLoggerParams 参数类
├── config.py          # 模块配置和 Union 类型定义
└── __init__.py        # 模块初始化
```

## 核心组件

### BaseLoggerParams

所有日志器参数的基类：

```python
@dataclass
class BaseLoggerParams(DictAccessMixin):
    """日志器参数基类"""

    _target_: str = Field(..., description="Logger 类的完整路径")
    version: Optional[Union[int, str]] = Field(None, description="实验版本")
    prefix: str = Field("", description="日志前缀")
```

### UnionLoggerParams

统一的日志器参数类型：

```python
UnionLoggerParams = Union[
    WandbLoggerParams,
    TensorBoardLoggerParams,
    CSVLoggerParams,
    MLFlowLoggerParams,
    NeptuneLoggerParams,
]
```

### 支持的日志器

- **WandbLogger**: Weights & Biases 实验跟踪
- **TensorBoardLogger**: TensorBoard 可视化日志
- **CSVLogger**: CSV 格式本地日志
- **MLFlowLogger**: MLflow 实验管理
- **NeptuneLogger**: Neptune AI 平台集成

## 使用示例

```python
from src._logger import UnionLoggerParams, WandbLoggerParams

# 创建 WandB 日志器参数
wandb_params = WandbLoggerParams(
    project="my_project",
    name="experiment_1",
    save_dir="./logs"
)

# 在配置中使用
config = SomeConfig(
    logger=wandb_params
)
```

## 扩展指南

要添加新的日志器支持：

1. 在 `_logger/` 下创建新文件夹（如 `new_logger/`）
2. 创建 `config.py` 实现 `NewLoggerParams` 类继承 `BaseLoggerParams`
3. 在 `__init__.py` 中添加导入和 Union 类型更新
4. 更新主 `config.py` 的导入（如果需要）
5. 更新 `__all__` 导出列表
