from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseLRSchedulerParams


@dataclass(config=ConfigDict(extra="allow"))
class CosineAnnealingWarmRestartsParams(BaseLRSchedulerParams):
    """CosineAnnealingWarmRestarts Scheduler Parameters"""

    _target_: str = Field(
        "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts",
        description="CosineAnnealingWarmRestarts scheduler class",
    )
    T_0: int = Field(50, description="Number of iterations for the first restart")
    T_mult: int = Field(1, description="A factor increases T_i after a restart")
    eta_min: float = Field(0.0, description="Minimum learning rate")
