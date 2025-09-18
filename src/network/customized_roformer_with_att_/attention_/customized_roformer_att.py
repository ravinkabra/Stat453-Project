from torch import nn
import torch
from typing import Optional, List, Tuple
from ..attention_.customized_roformer_self_att_ import CustomizedRoFormerSelfAttention
from transformers.models.roformer.modeling_roformer import (
    RoFormerSelfOutput,
    deprecate_kwarg,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.models.roformer.configuration_roformer import RoFormerConfig


class CustomizedRoFormerAttention(nn.Module):
    def __init__(self, config: RoFormerConfig, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.self = CustomizedRoFormerSelfAttention(config, layer_idx=layer_idx)
        self.output = RoFormerSelfOutput(config)
        self.pruned_heads = set()

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads: List[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sinusoidal_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        # Call self attention
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            sinusoidal_pos=sinusoidal_pos,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

        # Output projection
        # maybe because the hidden_states means the self.self.forward(hidden_states)
        # and the input_tensor is the hidden_states that of self.forward(hidden_states)
        attention_output = self.output.forward(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
