# Simplified and adapted RoFormer model for project network module.
# Many detailed implementations are omitted and replaced with minimal working
# placeholders to integrate into the framework. Use upstream HuggingFace for
# production-ready behavior.

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.roformer.configuration_roformer import RoFormerConfig
from transformers.models.roformer.modeling_roformer import (
    RoFormerSinusoidalPositionalEmbedding,
    logger,
    EncoderDecoderCache,
    DynamicCache,
)
from .customized_roformer_layer import CustomizedRoFormerLayer


class CustomizedRoFormerEncoder(nn.Module):
    def __init__(self, config: RoFormerConfig):
        super().__init__()
        self.config = config
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.hidden_size // config.num_attention_heads,
        )
        self.layer = nn.ModuleList([CustomizedRoFormerLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cache_position=None,
    ):
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
        if use_cache and isinstance(past_key_values, tuple):
            logger.warning_once(
                "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.58.0. "
                "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
            )
            past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        # [sequence_length, embed_size_per_head] -> [batch_size, num_heads, sequence_length, embed_size_per_head]
        sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1], past_key_values_length)[None, None, :, :]

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                sinusoidal_pos,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values,
                output_attentions,
                cache_position,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
