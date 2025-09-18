import math
import torch
from torch import nn
from typing import List, Union, Callable
from transformers.models.roformer.modeling_roformer import deprecate_kwarg, EncoderDecoderCache, RoFormerConfig


class CustomizedRoFormerSelfAttention(nn.Module):
    def __init__(self, config: RoFormerConfig, layer_idx=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.is_decoder = config.is_decoder
        self.rotary_value = config.rotary_value
        self.layer_idx = layer_idx

        # ACC Dropout configurations (moved from CustomizedRoFormerAttention)
        self.acc_positions = getattr(config, "acc_positions", [])
        self.use_acc_dropout = getattr(config, "use_acc_dropout", False)
        self.acc_dropout = nn.Dropout(getattr(config, "acc_dropout_prob", 0.1))
        self.acc_mode = getattr(config, "acc_mode", "original")

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        past_key_values=None,
        output_attentions=False,
        cache_position=None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        query_layer = (
            self.query(hidden_states)
            .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)
        )
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_layer = curr_past_key_value.layers[self.layer_idx].keys
            value_layer = curr_past_key_value.layers[self.layer_idx].values
        else:
            key_layer = (
                self.key(current_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )
            value_layer = (
                self.value(current_states)
                .view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
                .transpose(1, 2)
            )

            # Apply RoPE if self attention
            if not is_cross_attention and sinusoidal_pos is not None:
                if self.rotary_value:
                    query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer, value_layer
                    )
                else:
                    query_layer, key_layer = self.apply_rotary_position_embeddings(
                        sinusoidal_pos, query_layer, key_layer
                    )

            if past_key_values is not None:
                # save all key/value_layer to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_layer, value_layer = curr_past_key_value.update(
                    key_layer, value_layer, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_values.is_updated[self.layer_idx] = True

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RoFormerModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Apply ACC dropout (only for self-attention, not cross-attention)
        if self.use_acc_dropout and not is_cross_attention:
            attention_probs = self._apply_acc_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs

    def _apply_acc_dropout(self, attention_probs: torch.Tensor) -> torch.Tensor:
        """Apply ACC dropout to attention probabilities."""
        batch_size, num_heads, seq_len, _ = attention_probs.shape

        # Dynamically process ACC positions
        current_acc_positions = self._process_acc_positions(self.acc_positions, seq_len)

        # Select function based on self.acc_mode
        if self.acc_mode == "from_acc":
            return self._apply_from_acc_dropout(attention_probs, current_acc_positions, seq_len)
        elif self.acc_mode == "from_to_acc":
            return self._apply_from_to_acc_dropout(attention_probs, current_acc_positions, seq_len)
        elif self.acc_mode == "to_acc":
            return self._apply_to_acc_dropout(attention_probs, current_acc_positions, seq_len)
        elif self.acc_mode == "original":
            # Original mode: No dropout applied, return unmodified
            return attention_probs
        else:
            # Default to original behavior if mode is unknown or None
            return attention_probs

    def _process_acc_positions(
        self, acc_positions: Union[List[int], str, Callable[[int], List[int]], None], seq_len: int
    ) -> List[int]:
        """Process ACC positions dynamically based on current seq_len (no max_seq_len dependency).

        Args:
            acc_positions: Can be:
                - List[int]: Direct position list
                - str: "odd", "even", "all", "first_half", "second_half", "last_5"
                - callable: Lambda function that takes seq_len and returns position list
            seq_len: Current sequence length from input hidden states

        Returns:
            List[int]: Processed ACC positions filtered for current seq_len
        """
        if acc_positions is None or acc_positions == []:
            return []

        if isinstance(acc_positions, str):
            # Predefined patterns based on current seq_len
            if acc_positions == "odd":
                return [i for i in range(seq_len) if i % 2 == 1]
            elif acc_positions == "even":
                return [i for i in range(seq_len) if i % 2 == 0]
            elif acc_positions == "all":
                return list(range(seq_len))
            elif acc_positions == "first_half":
                return list(range(seq_len // 2))
            elif acc_positions == "second_half":
                return list(range(seq_len // 2, seq_len))
            elif acc_positions == "last_5":
                return [seq_len - i - 1 for i in range(min(5, seq_len))]
            else:
                raise ValueError(f"Unknown acc_positions pattern: {acc_positions}")

        elif callable(acc_positions):
            # Lambda function - pass current seq_len
            try:
                positions = acc_positions(seq_len)
                if not isinstance(positions, list):
                    positions = [positions]
                return [int(p) for p in positions if 0 <= p < seq_len]
            except Exception as e:
                raise ValueError(f"Error evaluating acc_positions lambda: {e}")

        elif isinstance(acc_positions, list):
            # Direct list - filter for current seq_len and handle negative indices
            processed_positions = []
            for pos in acc_positions:
                if isinstance(pos, int):
                    if pos < 0:
                        pos = seq_len + pos  # Convert negative to positive relative to seq_len
                    if 0 <= pos < seq_len:
                        processed_positions.append(pos)
                else:
                    raise ValueError(f"acc_positions list must contain integers, got {type(pos)}")
            return processed_positions

        else:
            raise ValueError(f"acc_positions must be list, str, or callable, got {type(acc_positions)}")

    def _apply_from_acc_dropout(
        self, attention_probs: torch.Tensor, current_acc_positions: List[int], seq_len: int
    ) -> torch.Tensor:
        """Standard ACC dropout (dropout from ACC positions to all)."""
        acc_mask = torch.zeros(seq_len, seq_len, device=attention_probs.device)
        for pos in current_acc_positions:
            if 0 <= pos < seq_len:
                acc_mask[pos, :] = 1  # Dropout from ACC to all
        acc_mask = acc_mask.unsqueeze(0).unsqueeze(0)
        acc_attention = attention_probs * acc_mask
        acc_attention = self.acc_dropout(acc_attention)
        return acc_attention + attention_probs * (1 - acc_mask)

    def _apply_from_to_acc_dropout(
        self, attention_probs: torch.Tensor, current_acc_positions: List[int], seq_len: int
    ) -> torch.Tensor:
        """Symmetric ACC dropout (dropout from and to ACC positions)."""
        acc_mask = torch.zeros(seq_len, seq_len, device=attention_probs.device)
        for pos in current_acc_positions:
            if 0 <= pos < seq_len:
                acc_mask[pos, :] = 1  # From ACC
                acc_mask[:, pos] = 1  # To ACC
        acc_mask = acc_mask.unsqueeze(0).unsqueeze(0)
        acc_attention = attention_probs * acc_mask
        acc_attention = self.acc_dropout(acc_attention)
        return acc_attention + attention_probs * (1 - acc_mask)

    def _apply_to_acc_dropout(
        self, attention_probs: torch.Tensor, current_acc_positions: List[int], seq_len: int
    ) -> torch.Tensor:
        """Target-only ACC dropout (only dropout to ACC positions, no dropout from)."""
        acc_mask = torch.zeros(seq_len, seq_len, device=attention_probs.device)
        for pos in current_acc_positions:
            if 0 <= pos < seq_len:
                acc_mask[:, pos] = 1  # Only to ACC
        acc_mask = acc_mask.unsqueeze(0).unsqueeze(0)
        acc_attention = attention_probs * acc_mask
        acc_attention = self.acc_dropout(acc_attention)
        return acc_attention + attention_probs * (1 - acc_mask)

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
            query_layer
        )
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
        # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
        rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
        if value_layer is not None:
            # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
            rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
                value_layer
            )
            value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
            return query_layer, key_layer, value_layer
        return query_layer, key_layer
