from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Literal

from ..base.config import BaseLRSchedulerParams


@dataclass(config=ConfigDict(extra="allow"))
class ReduceLROnPlateauParams(BaseLRSchedulerParams):
    """ReduceLROnPlateau Scheduler Parameters"""

    _target_: str = Field(
        "torch.optim.lr_scheduler.ReduceLROnPlateau",
        description="ReduceLROnPlateau scheduler class",
    )
    mode: Literal["min", "max"] = Field("min", description="Mode to monitor")
    factor: float = Field(
        0.1, description="Factor by which the learning rate will be reduced"
    )
    patience: int = Field(
        10,
        description="Number of epochs with no improvement after which learning rate will be reduced",
    )
    threshold: float = Field(
        1e-4, description="Threshold for measuring the new optimum"
    )
    threshold_mode: Literal["rel", "abs"] = Field("rel", description="Threshold mode")
    cooldown: int = Field(
        0,
        description="Number of epochs to wait before resuming normal operation after lr has been reduced",
    )
    min_lr: float = Field(0.0, description="Minimum learning rate")
    eps: float = Field(
        1e-8, description="Minimal decay applied to lr to avoid numerical issues"
    )
