from pydantic.dataclasses import dataclass, Field,ConfigDict
from typing import Dict, Any, Literal, Union, Optional

from torchmetrics import Metric

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


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ManagedMetricConfig:
    """
    管理的指标配置类

    每个 ManagedMetricConfig 管理一组相关的指标，支持多种指标配置方式：
    1. BaseMetricParams: 标准配置驱动方式 (推荐用于配置文件)
    2. Metric: 直接实例化方式 (推荐用于代码开发)
    3. str: 简化字符串方式 (快速原型开发)
    4. Dict[str, Any]: 原始字典方式 (向后兼容)

    示例：
    prediction_metrics:
      metrics:
        train_accuracy: "torchmetrics.Accuracy"
        val_accuracy: "torchmetrics.Accuracy"
        precision: "torchmetrics.Precision"
      log_config: {...}

    value_recorders:
      metrics:
        loss: "torchmetrics.MeanMetric"
        learning_rate: "torchmetrics.MeanMetric"
      log_config: {...}
    """

    metrics: Dict[str, Union[BaseMetricParams, Metric, str, Dict[str, Any]]] = Field(
        default_factory=dict, description="指标字典，键为指标名称，值为指标配置"
    )

    log_config: MetricLogConfig = Field(
        default_factory=MetricLogConfig, description="日志配置"
    )


@dataclass
class MetricManagerConfig:
    _target_: Literal["src.metric._manager.manager.MetricManager"] = Field(
        default="src.metric._manager.manager.MetricManager",
        description="The target class for the metric manager.",
    )
    metrics: Dict[str, ManagedMetricConfig] = Field(default_factory=dict)
