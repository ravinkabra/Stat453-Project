from .sample_saver.config import TrainingSampleSaverCallbackConfig
from .sample_saver.callback import TrainingSampleSaverCallback

from typing import Union

UnionCallbackConfig = Union[TrainingSampleSaverCallbackConfig, dict]
