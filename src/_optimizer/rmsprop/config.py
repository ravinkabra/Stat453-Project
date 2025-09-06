from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseOptimizerParams


@dataclass(config=ConfigDict(extra="allow"))
class RMSpropParams(BaseOptimizerParams):
    """RMSprop Optimizer Parameters"""

    _target_: str = Field("torch.optim.RMSprop", description="RMSprop optimizer class")
    lr: float = Field(1e-2, description="Learning rate")
    alpha: float = Field(0.99, description="Smoothing constant")
    eps: float = Field(
        1e-8, description="Term added to denominator for numerical stability"
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")
    momentum: float = Field(0.0, description="Momentum factor")
    centered: bool = Field(False, description="If True, compute centered RMSProp")
