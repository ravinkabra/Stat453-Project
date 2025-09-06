"""
🔧 Logger 模块 - 统一的日志器配置和管理

设计理念：
1. 区分 Config 和 Params：
   - XxxParams: 直接作为类实例化的参数（除了 _target_）
   - XxxConfig: 包含控制逻辑的配置（是否启用、选择哪个等）

2. Union 类型使用简洁名称，放在前面

使用方式：
```python
from src._logger import LoggerParams, WandbLoggerParams, LoggerConfig, LoggerPresets

# 方式1: 直接创建参数（用于 hydra instantiate）
wandb_params = WandbLoggerParams(project="my_project", name="experiment_1")

"""

from typing import Union

# 导入所有 Logger 参数类
from .wandb.config import WandbLoggerParams
from .tensorboard.config import TensorBoardLoggerParams
from .csv.config import CSVLoggerParams
from .mlflow.config import MLFlowLoggerParams
from .neptune.config import NeptuneLoggerParams


# ============================================================================
# Union 类型定义 - 放在前面，使用简洁名称
# ============================================================================

UnionLoggerParams = Union[
    WandbLoggerParams,
    TensorBoardLoggerParams,
    CSVLoggerParams,
    MLFlowLoggerParams,
    NeptuneLoggerParams,
]

__all__ = [
    # Union 类型
    "UnionLoggerParams",
    # 参数类
    "WandbLoggerParams",
    "TensorBoardLoggerParams",
    "CSVLoggerParams",
    "MLFlowLoggerParams",
    "NeptuneLoggerParams",
]
