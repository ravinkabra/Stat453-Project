from .base.config import BaseModelConfig
from .base.model import BaseModel
from .tutorial_mnist.config import MnistModelConfig
from .tutorial_mnist.model import MnistModel
from .old_pt_m2a_transformer.config import OldPtM2ATransformerConfig
from .old_pt_m2a_transformer.model import OldPtM2ATransformer
from typing import Union, Any

UnionModelConfig = Union[
    BaseModelConfig, MnistModelConfig, OldPtM2ATransformerConfig
]
UnionModel = Union[BaseModel, MnistModel, OldPtM2ATransformer, Any]
