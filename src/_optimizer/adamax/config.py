from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseOptimizerParams


@dataclass(config=ConfigDict(extra="allow"))
class AdamaxParams(BaseOptimizerParams):
    """Adamax Optimizer Parameters"""

    _target_: str = Field("torch.optim.Adamax", description="Adamax optimizer class")
    lr: float = Field(2e-3, description="Learning rate")
    betas: tuple[float, float] = Field(
        (0.9, 0.999), description="Betas for computing running averages"
    )
    eps: float = Field(
        1e-8, description="Term added to denominator for numerical stability"
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")
