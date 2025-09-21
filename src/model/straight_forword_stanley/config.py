from ..base.config import BaseModelConfig, dataclass, Field
from typing import Literal, Optional

# Import the reusable network params dataclass. We go up two package levels to reach
# the top-level `src.network` package.
from transformers import RoFormerConfig


@dataclass
class OldPtSFSTransformerConfig(BaseModelConfig):
    # SFS: Straight Forword Stanley
    _target_: Literal["src.model.old_pt_m2a_transformer.model.OldPtSFSTransformer"] = Field(
        default="src.model.old_pt_m2a_transformer.model.OldPtSFSTransformer"
    )

    # Replace the scalar hyper-parameters with three network params fields so the
    # model can be constructed from prefilled network configs. This keeps
    # configuration structured and re-uses the existing `CustomizedRoFormerEncoderParams`.
    global_network: RoFormerConfig = Field(default_factory=RoFormerConfig)
    local_encoder_network: RoFormerConfig = Field(default_factory=RoFormerConfig)
    local_decoder_network: RoFormerConfig = Field(default_factory=RoFormerConfig)

    cut_point_start: int = Field(default=4, description="切分点的起始位置")
    cut_point_end: Optional[int] = Field(default=None, description="切分点的结束位置（None表示到序列末尾）")
