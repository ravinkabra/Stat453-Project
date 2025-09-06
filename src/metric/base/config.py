from pydantic.dataclasses import dataclass,Field,ConfigDict
from typing import Optional, Union
from torchmetrics import MetricCollection
from torchmetrics.wrappers import MultitaskWrapper
from torchmetrics.classification import Accuracy, Precision, Recall
from torchmetrics.image import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure

@dataclass(config=ConfigDict(extra="allow"))
class BaseMetricParams:
    _target_:str = Field(default="torchmetrics.Metric")