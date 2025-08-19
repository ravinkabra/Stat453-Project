from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Optional, Union, Literal, Any


@dataclass(config=ConfigDict(extra="allow"))
class BaseLRSchedulerParams:
    """Config for learning rate scheduler configuration."""

    _target_: str = Field("torch.optim.lr_scheduler.CosineAnnealingLR", description="Learning rate scheduler class")
    optimizer: Optional[Any] = Field(None, description="Optimizer object")
    interval: Literal["step", "epoch"] = Field("step", description="Interval for updating the learning rate")
    frequency: int = Field(1, description="Frequency of updating the learning rate")
    monitor: Optional[str] = Field(None, description="Metric to monitor for learning rate scheduling")
    strict: bool = Field(True, description="Whether to strictly enforce the configuration")
    name: Optional[str] = Field(None, description="Name of the learning rate scheduler")
    T_max: int = Field(10000, description="Total number of training steps")
    eta_min: float = Field(1e-6, description="Minimum learning rate")
