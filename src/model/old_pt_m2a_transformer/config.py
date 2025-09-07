from ..base.config import BaseModelConfig, dataclass, Field
from typing import Literal, Optional


@dataclass
class OldPtM2ATransformerConfig(BaseModelConfig):
    _target_: Literal["src.model.old_m2a_transformer.model.OldM2ATransformer"] = Field(
        default="src.model.old_m2a_transformer.model.OldM2ATransformer"
    )

    # Model hyper-parameters (defaults chosen to match original usage)
    large: bool = Field(False)
    hidden_size: Optional[int] = Field(None)
    num_layers: Optional[int] = Field(None)
    num_attention_heads: Optional[int] = Field(None)
    intermediate_size: Optional[int] = Field(None)

    local_model_num_layers: int = Field(3)
    local_model_num_attention_heads: int = Field(8)
    local_model_intermediate_size: int = Field(768)
