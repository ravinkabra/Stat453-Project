from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional, Union, Literal

from ..base.config import BaseModelConfig
from ...network.example import ExampleNetworkConfig


@dataclass
class ExampleModelConfig(BaseModelConfig):
    _target_: Literal["src.model.example.model.ExampleModel"] = Field(default="src.model.example.model.ExampleModel")
    example_param: Optional[Union[str, int]] = Field(
        default=None, description="An example parameter for the ExampleModel"
    )
    example_network: ExampleNetworkConfig = Field(default_factory=ExampleNetworkConfig)
