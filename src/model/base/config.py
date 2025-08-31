from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional, Union, Literal, Dict

from ..._optimizer import UnionOptimizerParams
from ..._lr_scheduler import UnionLRSchedulerParams
from ..._optimizer.base import BaseOptimizerParams
from ..._lr_scheduler.base import BaseLRSchedulerParams
from ...metric._manager import MetricManagerConfig, MetricManager


@dataclass
class BaseModelConfig:
    _target_: Literal["src.model.base.model.BaseModel"] = Field(
        default="src.model.base.model.BaseModel"
    )

    optimizer: Optional[
        Union[UnionOptimizerParams, Dict[str, UnionOptimizerParams]]
    ] = Field(
        default_factory=BaseOptimizerParams,
        description="优化器配置：单个配置优化全模型，字典配置优化指定模块",
    )

    lr_scheduler: Optional[
        Union[UnionLRSchedulerParams, Dict[str, UnionLRSchedulerParams]]
    ] = Field(
        default_factory=BaseLRSchedulerParams,
        description="学习率调度器配置：需与 optimizer 结构匹配",
    )

    metrics: Optional[MetricManagerConfig] = Field(default_factory=MetricManagerConfig)
