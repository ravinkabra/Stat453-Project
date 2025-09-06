from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseLRSchedulerParams


@dataclass(config=ConfigDict(extra="allow"))
class ExponentialLRParams(BaseLRSchedulerParams):
    """ExponentialLR Scheduler Parameters"""

    _target_: str = Field(
        "torch.optim.lr_scheduler.ExponentialLR",
        description="ExponentialLR scheduler class",
    )
    gamma: float = Field(
        0.9, description="Multiplicative factor of learning rate decay"
    )
