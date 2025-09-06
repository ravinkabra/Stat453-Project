from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseOptimizerParams


@dataclass(config=ConfigDict(extra="allow"))
class AdadeltaParams(BaseOptimizerParams):
    """Adadelta Optimizer Parameters"""

    _target_: str = Field(
        "torch.optim.Adadelta", description="Adadelta optimizer class"
    )
    lr: float = Field(1.0, description="Learning rate")
    rho: float = Field(0.9, description="Coefficient for computing running averages")
    eps: float = Field(
        1e-6, description="Term added to denominator for numerical stability"
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")
