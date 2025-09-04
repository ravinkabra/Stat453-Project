from typing import Optional, Union, Literal
from ..base.config import BaseNetworkConfig, dataclass, Field


@dataclass
class LeNetConfig(BaseNetworkConfig):
    _target_: Literal["src.network.tutorial_lenet.network.LeNet"] = Field(
        default="src.network.tutorial_lenet.network.LeNet"
    )
    input_channels: int = Field(default=1, description="Input image channels")
    num_classes: int = Field(default=10, description="Number of output classes")
