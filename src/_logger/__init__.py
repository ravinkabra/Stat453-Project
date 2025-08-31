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

# 方式2: 创建配置（包含控制逻辑）
logger_config = LoggerConfig(
    enabled=True,
    params=wandb_params
)

# 方式3: 使用预设
wandb_params = LoggerPresets.wandb_basic("my_project", "experiment_1")

# 方式4: 多logger配置
multi_config = LoggerPresets.multi_logger_setup("my_project", "experiment_1")
enabled_loggers = multi_config.get_enabled_loggers()  # Dict[str, LoggerParams]
```
"""

from .config import (
    # Union 类型 - 简洁名称，放在前面
    UnionLoggerParams,
    # 参数类 - 直接作为实例化参数
    BaseLoggerParams,
    WandbLoggerParams,
    TensorBoardLoggerParams,
    CSVLoggerParams,
    MLFlowLoggerParams,
    NeptuneLoggerParams,
    # 配置类 - 包含控制逻辑
    LoggerConfig,
    MultiLoggerConfig,
    # 预设和工厂
    LoggerPresets,
    create_logger_params,
    create_logger_config,
)

__all__ = [
    # Union 类型
    "UnionLoggerParams",
    # 参数类
    "BaseLoggerParams",
    "WandbLoggerParams",
    "TensorBoardLoggerParams",
    "CSVLoggerParams",
    "MLFlowLoggerParams",
    "NeptuneLoggerParams",
    # 配置类
    "LoggerConfig",
    "MultiLoggerConfig",
    # 工具
    "LoggerPresets",
    "create_logger_params",
    "create_logger_config",
]
