from typing import Literal
from ..base.config import BaseNetworkConfig, dataclass, Field
from pydantic.dataclasses import ConfigDict
from transformers.models.roformer.configuration_roformer import RoFormerConfig


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class CustomizedRoFormerEncoderParams(BaseNetworkConfig):
    """Params-style config for RoFormer encoder (customized name).

    This Params holds a HuggingFace `RoFormerConfig` instance as `config` and
    a `_target_` so the project's factory can instantiate the encoder and pass
    the prefilled config into it.
    """

    _target_: Literal["src.network.customized_roformer.network.CustomizedRoFormerEncoder"] = Field(
        default="src.network.customized_roformer.network.CustomizedRoFormerEncoder"
    )
    # The HF config carries detailed hyper-parameters. Provide explicit
    # preset defaults here so users can create params without further steps.
    config: RoFormerConfig = Field(
        default_factory=lambda: RoFormerConfig(
            vocab_size=30522,
            embedding_size=768,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            pad_token_id=0,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            is_decoder=False,
            add_cross_attention=False,
            rotary_value=1.0,
            chunk_size_feed_forward=0,
            use_return_dict=True,
        )
    )
