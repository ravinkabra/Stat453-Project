from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseLRSchedulerParams


@dataclass(config=ConfigDict(extra="allow"))
class CosineAnnealingLRParams(BaseLRSchedulerParams):
    """CosineAnnealingLR Scheduler Parameters"""

    _target_: str = Field(
        "torch.optim.lr_scheduler.CosineAnnealingLR",
        description="CosineAnnealingLR scheduler class",
    )
    T_max: int = Field(100, description="Maximum number of iterations")
    eta_min: float = Field(0.0, description="Minimum learning rate")
