from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseOptimizerParams


@dataclass(config=ConfigDict(extra="allow"))
class RpropParams(BaseOptimizerParams):
    """Rprop Optimizer Parameters"""

    _target_: str = Field("torch.optim.Rprop", description="Rprop optimizer class")
    lr: float = Field(1e-2, description="Learning rate")
    etas: tuple[float, float] = Field(
        (0.5, 1.2), description="Multiplicative increase and decrease factors"
    )
    step_sizes: tuple[float, float] = Field(
        (1e-6, 50.0), description="Minimum and maximum allowed step sizes"
    )
