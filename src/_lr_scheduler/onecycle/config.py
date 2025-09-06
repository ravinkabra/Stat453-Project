from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Optional

from ..base.config import BaseLRSchedulerParams


@dataclass(config=ConfigDict(extra="allow"))
class OneCycleLRParams(BaseLRSchedulerParams):
    """OneCycleLR Scheduler Parameters"""

    _target_: str = Field(
        "torch.optim.lr_scheduler.OneCycleLR", description="OneCycleLR scheduler class"
    )
    max_lr: float = Field(
        1e-3, description="Upper learning rate boundaries in the cycle"
    )
    total_steps: Optional[int] = Field(
        None, description="Total number of steps in the cycle"
    )
    epochs: Optional[int] = Field(None, description="Total number of epochs")
    steps_per_epoch: Optional[int] = Field(
        None, description="Number of steps per epoch"
    )
    pct_start: float = Field(
        0.3, description="Percentage of the cycle spent increasing the learning rate"
    )
    anneal_strategy: str = Field(
        "cos", description="Annealing strategy: 'linear' or 'cos'"
    )
    cycle_momentum: bool = Field(True, description="Whether to cycle momentum")
    base_momentum: float = Field(
        0.85, description="Initial momentum which is the lower boundary in the cycle"
    )
    max_momentum: float = Field(
        0.95, description="Upper momentum boundaries in the cycle"
    )
    div_factor: float = Field(
        25.0,
        description="Determines the initial learning rate via initial_lr = max_lr/div_factor",
    )
    final_div_factor: float = Field(
        1e4,
        description="Determines the minimum learning rate via min_lr = initial_lr/final_div_factor",
    )
    three_phase: bool = Field(False, description="Whether to use three phases")
