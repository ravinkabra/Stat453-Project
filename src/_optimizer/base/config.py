from pydantic.dataclasses import dataclass, ConfigDict
from pydantic import Field
from typing import Optional, Union, Literal, Any


@dataclass(config=ConfigDict(extra="allow"))
class BaseOptimizerParams:
    _target_: str = Field("torch.optim.Adam", description="Optimizer class")
    params: Optional[Any] = Field(None, description="Parameters for the model.")
    lr: float = Field(1e-5, description="Learning rate")
    betas: tuple[float, float] = Field((0.9, 0.999), description="Betas")
    eps: float = Field(1e-8, description="Epsilon")
    weight_decay: float = Field(0.0, description="Weight decay")
