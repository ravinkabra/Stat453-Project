from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional, Union, Literal

from ...optimizer import UnionOptimizerParams
from ...lr_scheduler import UnionLRSchedulerParams
from ...optimizer.base import BaseOptimizerParams
from ...lr_scheduler.base import BaseLRSchedulerParams
from ...metric._manager import MetricManagerConfig, MetricManager

@dataclass
class BaseModelConfig:
    _target_: Literal["src.model.base.model.BaseModel"] = Field(default="src.model.base.model.BaseModel")
    optimizer: Optional[UnionOptimizerParams] = Field(default_factory=BaseOptimizerParams)
    lr_scheduler: Optional[UnionLRSchedulerParams] = Field(default_factory=BaseLRSchedulerParams)
    metrics: Optional[MetricManagerConfig] = Field(default_factory=MetricManagerConfig)
