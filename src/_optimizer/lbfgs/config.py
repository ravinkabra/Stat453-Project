from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Optional

from ..base.config import BaseOptimizerParams


@dataclass(config=ConfigDict(extra="allow"))
class LBFGSParams(BaseOptimizerParams):
    """LBFGS Optimizer Parameters"""

    _target_: str = Field("torch.optim.LBFGS", description="LBFGS optimizer class")
    lr: float = Field(1.0, description="Learning rate")
    max_iter: int = Field(
        20, description="Maximum number of iterations per optimization step"
    )
    max_eval: Optional[int] = Field(
        None, description="Maximum number of function evaluations per optimization step"
    )
    tolerance_grad: float = Field(
        1e-7, description="Termination tolerance on first order optimality"
    )
    tolerance_change: float = Field(
        1e-9, description="Termination tolerance on function value/parameter changes"
    )
    history_size: int = Field(100, description="Update history size")
    line_search_fn: Optional[str] = Field(None, description="Line search function")
