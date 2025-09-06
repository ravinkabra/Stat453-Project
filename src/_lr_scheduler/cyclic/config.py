from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Optional, Literal

from ..base.config import BaseLRSchedulerParams


@dataclass(config=ConfigDict(extra="allow"))
class CyclicLRParams(BaseLRSchedulerParams):
    """CyclicLR Scheduler Parameters"""

    _target_: str = Field(
        "torch.optim.lr_scheduler.CyclicLR", description="CyclicLR scheduler class"
    )
    base_lr: float = Field(
        1e-5,
        description="Initial learning rate which is the lower boundary in the cycle",
    )
    max_lr: float = Field(
        1e-3, description="Upper learning rate boundaries in the cycle"
    )
    step_size_up: int = Field(
        2000,
        description="Number of training iterations in the increasing half of a cycle",
    )
    step_size_down: Optional[int] = Field(
        None,
        description="Number of training iterations in the decreasing half of a cycle",
    )
    mode: Literal["triangular", "triangular2", "exp_range"] = Field(
        "triangular", description="Policy for the cycle"
    )
    gamma: float = Field(1.0, description="Constant in 'exp_range' mode")
    scale_fn: Optional[object] = Field(
        None,
        description="Custom scaling policy defined by a single argument lambda function",
    )
    scale_mode: Literal["cycle", "iterations"] = Field(
        "cycle",
        description="Defines whether scale_fn is evaluated on cycle number or cycle iterations",
    )
    cycle_momentum: bool = Field(True, description="Whether to cycle momentum")
    base_momentum: float = Field(
        0.8, description="Initial momentum which is the lower boundary in the cycle"
    )
    max_momentum: float = Field(
        0.9, description="Upper momentum boundaries in the cycle"
    )
