from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Optional, Union, Literal, Any
from ..base import BaseLRSchedulerParams


@dataclass(config=ConfigDict(extra="allow"))
class ExampleLRSchedulerParams(BaseLRSchedulerParams):
    """Config for learning rate scheduler configuration."""

    _target_: str = Field("torch.optim.lr_scheduler.CosineAnnealingLR", description="Learning rate scheduler class")
    T_max: int = Field(10000, description="Total number of training steps")
    eta_min: float = Field(1e-6, description="Minimum learning rate")
