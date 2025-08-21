from pydantic.dataclasses import dataclass, Field
from typing import Dict, Any, Literal, Union,Optional

# from torchmetrics import Metric

from ..base.config import BaseMetricParams


@dataclass
class MetricLogConfig:
    phase: list[Literal["train", "val", "test"]] = Field(
        default_factory=lambda: ["train", "val", "test"]
    )
    frequency: int = Field(
        default=1,
        description="Logging frequency. Note that the real logging frequency will be the Least Common Multiple of this and the logging frequency of the trainer.",
    )
    # the log params needed by lightning module
    prog_bar: bool = Field(
        default=False, description="Whether to log in the progress bar."
    )
    logger: bool = Field(default=True, description="Whether to log to the logger.")
    on_step: bool = Field(default=True, description="Whether to log on step.")
    on_epoch: bool = Field(default=True, description="Whether to log on epoch.")
    reduce_fx: Optional[Literal["mean", "sum", "max", "min"]] = Field(
        default="mean", description="Reduction function to apply to the logged values."
    )

@dataclass
class ManagedMetricConfig:
    metric: Union[BaseMetricParams, Any] = Field(default_factory=BaseMetricParams)
    log_config: MetricLogConfig = Field(default_factory=MetricLogConfig)


@dataclass
class MetricManagerConfig:
    _target_: Literal["src.metric._manager.manager.MetricManager"] = Field(
        default="src.metric._manager.manager.MetricManager",
        description="The target class for the metric manager.",
    )
    metrics: Dict[str, ManagedMetricConfig] = Field(default_factory=dict)
