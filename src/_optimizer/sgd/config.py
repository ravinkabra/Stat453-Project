from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseOptimizerParams


@dataclass(config=ConfigDict(extra="allow"))
class SGDParams(BaseOptimizerParams):
    """SGD Optimizer Parameters"""

    _target_: str = Field("torch.optim.SGD", description="SGD optimizer class")
    lr: float = Field(1e-2, description="Learning rate")
    momentum: float = Field(0.0, description="Momentum factor")
    dampening: float = Field(0.0, description="Dampening for momentum")
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")
    nesterov: bool = Field(False, description="Enables Nesterov momentum")
