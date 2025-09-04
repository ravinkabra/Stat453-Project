from .base.config import BaseModelConfig
from .base.model import BaseModel
from .tutorial_mnist.config import MnistModelConfig
from .tutorial_mnist.model import MnistModel
from typing import Union, Any

UnionModelConfig = Union[BaseModelConfig, MnistModelConfig, dict]
UnionModel = Union[BaseModel, MnistModel, Any]
