from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseLRSchedulerParams


@dataclass(config=ConfigDict(extra="allow"))
class StepLRParams(BaseLRSchedulerParams):
    """StepLR Scheduler Parameters"""

    _target_: str = Field(
        "torch.optim.lr_scheduler.StepLR", description="StepLR scheduler class"
    )
    step_size: int = Field(30, description="Period of learning rate decay")
    gamma: float = Field(
        0.1, description="Multiplicative factor of learning rate decay"
    )
