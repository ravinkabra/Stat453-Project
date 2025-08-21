from pydantic.dataclasses import dataclass,Field,ConfigDict
from typing import Optional, Union
# from torchmetrics import MetricTracker,MetricCollection
# from torchmetrics.wrappers import MultioutputWrapper,MultitaskWrapper
# from torchmetrics.classification import Accuracy, Precision, Recall
# from pytorch_lightning.callbacks import Callback
# Accuracy()
# Precision()
# Recall()

@dataclass(config=ConfigDict(extra="allow"))
class BaseMetricParams:
    _target_:str = Field(default="torchmetrics.Metric")