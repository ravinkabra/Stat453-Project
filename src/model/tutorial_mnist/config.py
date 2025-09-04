from ..base.config import BaseModelConfig, dataclass, Field
from ...network.tutorial_lenet.config import LeNetConfig
from typing import Literal


@dataclass
class MnistModelConfig(BaseModelConfig):
    _target_: Literal["src.model.tutorial_mnist.model.MnistModel"] = Field(
        default="src.model.tutorial_mnist.model.MnistModel"
    )
    network: LeNetConfig = Field(default_factory=LeNetConfig, description="网络配置")
