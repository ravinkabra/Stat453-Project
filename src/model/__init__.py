from .base.config import BaseModelConfig
from .base.model import BaseModel
from .tutorial_mnist.config import MnistModelConfig
from .tutorial_mnist.model import MnistModel
from .old_pt_m2a_transformer.config import OldPtM2ATransformerConfig
from .old_pt_m2a_transformer.model import OldPtM2ATransformer
from .old_pt_m2a_transformer_with_att.config import OldPtM2ATransformerWithAttentionConfig
from .old_pt_m2a_transformer_with_att.model import OldPtM2ATransformerWithAttention
from typing import Literal
from typing import Union, Any

UnionModelConfig = Union[
    BaseModelConfig, MnistModelConfig, OldPtM2ATransformerConfig, OldPtM2ATransformerWithAttentionConfig
]
UnionModel = Union[BaseModel, MnistModel, OldPtM2ATransformer, OldPtM2ATransformerWithAttention, Any]
