from torch import nn
import torch
from ..attention_.customized_roformer_self_att_ import CustomizedRoFormerSelfAttention
from transformers.models.roformer.modeling_roformer import (
    RoFormerSelfOutput,
    deprecate_kwarg,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)


class CustomizedRoFormerAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.self = CustomizedRoFormerSelfAttention(config, layer_idx=layer_idx)
        self.output = RoFormerSelfOutput(config)
        self.pruned_heads = set()
        
        # ACC Dropout configurations
        # Process ACC positions (support lambda, odd/even, or list)
        self.acc_positions = self._process_acc_positions(config.acc_positions, config.max_position_embeddings)
        self.use_acc_dropout = config.use_acc_dropout
        self.use_random_acc_dropout = config.use_random_acc_dropout
        self.acc_positions = config.acc_positions or []
        self.acc_dropout = nn.Dropout(config.acc_dropout_prob)
        self.random_acc_dropout = nn.Dropout(config.random_acc_dropout_prob)

    # Copied from transformers.models.bert.modeling_bert.BertAttention.prune_heads
    def prune_heads(self, heads):
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
        hidden_states,
        attention_mask=None,
        sinusoidal_pos=None,
        head_mask=None,
        encoder_hidden_states=None,
        past_key_values=None,
        output_attentions=False,
        cache_position=None,
    ):
        # Apply random ACC dropout on input features
        if self.use_random_acc_dropout:
            hidden_states = self.apply_random_acc_dropout(hidden_states)
        
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
        
        # Apply attention dropout on ACC positions
        if self.use_acc_dropout and len(self_outputs) > 1:
            self_outputs = self.apply_acc_dropout_on_attention(self_outputs)
        
        # Output projection
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
    
    def apply_random_acc_dropout(self, hidden_states):
        """Apply random dropout to ACC feature positions in input"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get current ACC positions for this sequence length
        current_acc_positions = self._get_current_acc_positions(seq_len)
        
        # Create mask for ACC positions
        acc_mask = torch.zeros(seq_len, device=hidden_states.device)
        for pos in self.acc_positions:
            if 0 <= pos < seq_len:
                acc_mask[pos] = 1
        
        # Expand mask to match hidden_states shape
        acc_mask = acc_mask.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        
        # Apply dropout only to ACC positions
        acc_features = hidden_states * acc_mask
        acc_features = self.random_acc_dropout(acc_features)
        
        # Combine with non-ACC features
        non_acc_features = hidden_states * (1 - acc_mask)
        
        return acc_features + non_acc_features
    
    def _process_acc_positions(self, acc_positions, max_seq_len):
        """Process ACC positions with support for lambda, odd/even, or list
        
        Args:
            acc_positions: Can be:
                - List[int]: Direct position list
                - str: "odd", "even", "all"
                - callable: Lambda function that takes seq_len and returns position list
            max_seq_len: Maximum sequence length for validation
            
        Returns:
            List[int]: Processed ACC positions
        """
        if acc_positions is None:
            return []
        
        if isinstance(acc_positions, str):
            # Predefined patterns
            if acc_positions == "odd":
                return [i for i in range(max_seq_len) if i % 2 == 1]
            elif acc_positions == "even":
                return [i for i in range(max_seq_len) if i % 2 == 0]
            elif acc_positions == "all":
                return list(range(max_seq_len))
            elif acc_positions == "first_half":
                return list(range(max_seq_len // 2))
            elif acc_positions == "second_half":
                return list(range(max_seq_len // 2, max_seq_len))
            elif acc_positions == "last_5":
                return [max_seq_len - i - 1 for i in range(min(5, max_seq_len))]
            else:
                raise ValueError(f"Unknown acc_positions pattern: {acc_positions}")
        
        elif callable(acc_positions):
            # Lambda function
            try:
                positions = acc_positions(max_seq_len)
                if not isinstance(positions, list):
                    positions = [positions]
                return [int(p) for p in positions if 0 <= p < max_seq_len]
            except Exception as e:
                raise ValueError(f"Error evaluating acc_positions lambda: {e}")
        
        elif isinstance(acc_positions, list):
            # Direct list - handle negative indices
            processed_positions = []
            for pos in acc_positions:
                if isinstance(pos, int):
                    if pos < 0:
                        pos = max_seq_len + pos  # Convert negative to positive
                    if 0 <= pos < max_seq_len:
                        processed_positions.append(pos)
                else:
                    raise ValueError(f"acc_positions list must contain integers, got {type(pos)}")
            return processed_positions
        
        else:
            raise ValueError(f"acc_positions must be list, str, or callable, got {type(acc_positions)}")
    
    def apply_acc_dropout_on_attention(self, self_outputs):
        """Apply dropout to attention weights from ACC positions"""
        context_layer = self_outputs[0]
        attention_probs = self_outputs[1]  # [batch_size, num_heads, seq_len, seq_len]
        
        batch_size, num_heads, seq_len, _ = attention_probs.shape
        
        # Get current ACC positions for this sequence length
        current_acc_positions = self._get_current_acc_positions(seq_len)
        
        # Create mask for ACC positions (dropout attention FROM ACC positions)
        acc_mask = torch.zeros(seq_len, seq_len, device=attention_probs.device)
        for pos in self.acc_positions:
            if 0 <= pos < seq_len:
                acc_mask[pos, :] = 1  # Dropout attention from this ACC position to all positions
        
        # Expand mask to match attention_probs shape
        acc_mask = acc_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply dropout only to ACC attention weights
        acc_attention = attention_probs * acc_mask
        acc_attention = self.acc_dropout(acc_attention)
        
        # Combine with non-ACC attention weights
        non_acc_attention = attention_probs * (1 - acc_mask)
        modified_attention_probs = acc_attention + non_acc_attention
        
        # Return modified outputs
        return (context_layer, modified_attention_probs) + self_outputs[2:]
