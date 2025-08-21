from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Optional, Union, Literal

@dataclass
class ExampleNetworkConfig:
    _target_: Literal["src.network.example.network.ExampleNetwork"] = Field(default="src.network.example.network.ExampleNetwork")
    input_size: int = Field(default=784, description="Input size for the network")
    output_size: int = Field(default=10, description="Output size for the network")
    hidden_layers: Optional[list[int]] = Field(default=None, description="List of hidden layer sizes")
    activation: str = Field(default="relu", description="Activation function to use")