from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field

from ..base.config import BaseOptimizerParams


@dataclass(config=ConfigDict(extra="allow"))
class AdagradParams(BaseOptimizerParams):
    """Adagrad Optimizer Parameters"""

    _target_: str = Field("torch.optim.Adagrad", description="Adagrad optimizer class")
    lr: float = Field(1e-2, description="Learning rate")
    lr_decay: float = Field(0.0, description="Learning rate decay")
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")
    initial_accumulator_value: float = Field(
        0.0, description="Initial accumulator value"
    )
    eps: float = Field(
        1e-10, description="Term added to denominator for numerical stability"
    )
