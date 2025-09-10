from pydantic.dataclasses import dataclass, Field, ConfigDict
from typing import Optional, Literal, Dict, Any
import dataclasses

from src.metric.base.config import BaseMetricParams


@dataclass(config=ConfigDict(extra="allow"))
class ClassificationMetricParams(BaseMetricParams):
    """通用的分类 metric 参数基类，包含 torchmetrics 分类常见的初始化参数"""
    task: Literal["binary", "multiclass", "multilabel"] = Field(default="multiclass")
    num_classes: Optional[int] = Field(default=None)
    average: Optional[str] = Field(default="macro")
    threshold: Optional[float] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    ignore_index: Optional[int] = Field(default=None)
    # mdmc_average: Optional[str] = Field(default=None)


@dataclass(config=ConfigDict(extra="allow"))
class AccuracyParams(ClassificationMetricParams):
    _target_: str = Field(default="torchmetrics.classification.Accuracy")


@dataclass(config=ConfigDict(extra="allow"))
class F1ScoreParams(ClassificationMetricParams):
    _target_: str = Field(default="torchmetrics.classification.F1Score")
    average: Optional[str] = Field(default="macro")


@dataclass(config=ConfigDict(extra="allow"))
class PrecisionParams(ClassificationMetricParams):
    _target_: str = Field(default="torchmetrics.classification.Precision")
    average: Optional[str] = Field(default="macro")


@dataclass(config=ConfigDict(extra="allow"))
class RecallParams(ClassificationMetricParams):
    _target_: str = Field(default="torchmetrics.classification.Recall")
    average: Optional[str] = Field(default="macro")


__all__ = [
    "ClassificationMetricParams",
    "AccuracyParams",
    "F1ScoreParams",
    "PrecisionParams",
    "RecallParams",
]
