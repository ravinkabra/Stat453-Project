from .sample_saver.config import SampleSaverCallbackConfig
from .sample_saver.callback import TrainingSampleSaverCallback
from .checkpoint.config import ModelCheckpointParams
from .early_stopping.config import EarlyStoppingParams

from typing import Union

# 回调配置联合类型
UnionCallbackConfig = Union[
    SampleSaverCallbackConfig, ModelCheckpointParams, EarlyStoppingParams, dict
]

__all__ = [
    "TrainingSampleSaverCallback",
    "SampleSaverCallbackConfig",
    "ModelCheckpointParams",
    "EarlyStoppingParams",
    "UnionCallbackConfig",
]
