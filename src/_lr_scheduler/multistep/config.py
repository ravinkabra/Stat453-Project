from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import List

from ..base.config import BaseLRSchedulerParams


@dataclass(config=ConfigDict(extra="allow"))
class MultiStepLRParams(BaseLRSchedulerParams):
    """MultiStepLR Scheduler Parameters"""

    _target_: str = Field(
        "torch.optim.lr_scheduler.MultiStepLR",
        description="MultiStepLR scheduler class",
    )
    milestones: List[int] = Field([30, 80], description="List of epoch indices")
    gamma: float = Field(
        0.1, description="Multiplicative factor of learning rate decay"
    )
