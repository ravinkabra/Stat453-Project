from pydantic.dataclasses import dataclass, Field, ConfigDict
from pydantic import field_validator, model_validator
from typing import Dict, Any, Literal, Union, Optional, Self

from torchmetrics import Metric

from ..base.config import BaseMetricParams


@dataclass
class MetricLogConfig:
    phase: list[Literal["train", "val", "test"]] = Field(
        default_factory=lambda: ["train", "val", "test"],
        description="When not in training phase, the 'on_step' logging will be forced to False. https://github.com/Lightning-AI/pytorch-lightning/issues/10436",
    )
    update_frequency: int = Field(
        default=1, description="Update frequency for the logged values."
    )
    compute_frequency: Optional[int] = Field(
        default=None,
        description="Compute frequency for the logged values. Must be a multiple of update_frequency. If not set, will default to update_frequency.",
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

    @model_validator(mode="after")
    def validate_compute_frequency(self) -> Self:
        """验证 compute_frequency 必须是 update_frequency 的倍数"""
        if self.compute_frequency is None:
            self.compute_frequency = self.update_frequency
        elif self.compute_frequency % self.update_frequency != 0:
            raise ValueError("compute_frequency must be a multiple of update_frequency")
        return self

    @model_validator(mode="after")
    def validate_phase_and_on_step(self) -> Self:
        """验证 phase 和 on_step 的组合"""
        if self.on_step and ("val" in self.phase or "test" in self.phase):
            raise ValueError(
                f"When not in training phase, 'on_step' logging must be False. But got on_step={self.on_step} and phase={self.phase}."
            )
        return self


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
