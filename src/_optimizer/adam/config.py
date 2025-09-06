from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseOptimizerParams


@dataclass(config=ConfigDict(extra="allow"))
class AdamParams(BaseOptimizerParams):
    """Adam Optimizer Parameters"""

    _target_: str = Field("torch.optim.Adam", description="Adam optimizer class")
    lr: float = Field(1e-3, description="Learning rate")
    betas: tuple[float, float] = Field(
        (0.9, 0.999), description="Betas for computing running averages"
    )
    eps: float = Field(
        1e-8, description="Term added to denominator for numerical stability"
    )
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")
    amsgrad: bool = Field(False, description="Whether to use AMSGrad variant")
