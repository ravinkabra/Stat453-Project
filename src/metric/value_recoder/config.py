from pydantic.dataclasses import dataclass, Field, ConfigDict
from typing import Literal, Optional
from ..base.config import BaseMetricParams


@dataclass(config=ConfigDict(extra="allow"))
class ValueRecorderParams(BaseMetricParams):
    _target_: Literal["src.metric.value_recoder.metric.ValueRecorderMetric"] = Field(
        default="src.metric.value_recoder.metric.ValueRecorderMetric"
    )
    aggregation: Literal["mean", "sum", "last"] = Field(default="mean")
