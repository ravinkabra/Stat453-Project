from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional, Union, Literal


@dataclass
class BaseNetworkConfig:
    _target_: Literal["src.network.base.network.BaseNetwork"] = Field(
        default="src.network.base.network.BaseNetwork"
    )
