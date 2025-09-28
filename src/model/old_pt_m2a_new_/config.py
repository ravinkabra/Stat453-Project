from ..base.config import BaseModelConfig, dataclass, Field
from typing import Literal

# Import the reusable network params dataclass. We go up two package levels to reach
# the top-level `src.network` package.
from ...network.customized_roformer.config import CustomizedRoFormerEncoderParams


@dataclass
class OldPtM2ANewConfig(BaseModelConfig):
    _target_: Literal["src.model.old_pt_m2a_new_.model.OldPtM2ANew"] = Field(
        default="src.model.old_pt_m2a_new_.model.OldPtM2ANew"
    )


    # Replace the scalar hyper-parameters with three network params fields so the
    # model can be constructed from prefilled network configs. This keeps
    # configuration structured and re-uses the existing `CustomizedRoFormerEncoderParams`.
    global_network: CustomizedRoFormerEncoderParams = Field(default_factory=CustomizedRoFormerEncoderParams)
    local_encoder_network: CustomizedRoFormerEncoderParams = Field(default_factory=CustomizedRoFormerEncoderParams)
    local_decoder_network: CustomizedRoFormerEncoderParams = Field(default_factory=CustomizedRoFormerEncoderParams)
