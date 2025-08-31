from pydantic.dataclasses import dataclass,Field,ConfigDict
from typing import Optional, Union
from torchmetrics import MetricCollection
from torchmetrics.wrappers import MultitaskWrapper
from torchmetrics.classification import Accuracy, Precision, Recall
from torchmetrics.image import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
Accuracy.__new__()
Precision.__new__()
Recall.__new__()
PeakSignalNoiseRatio.__init__()
StructuralSimilarityIndexMeasure.__init__()

@dataclass(config=ConfigDict(extra="allow"))
class BaseMetricParams:
    _target_:str = Field(default="torchmetrics.Metric")