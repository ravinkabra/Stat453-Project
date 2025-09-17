# ACC Dropout Mechanisms in Customized RoFormer

This document describes the two ACC (Acoustic Conditioning) dropout mechanisms implemented in the Customized RoFormer attention module.

## Overview

Two different dropout strategies have been implemented to study the impact of ACC features on attention mechanisms:

### 1. Attention Dropout on ACC
- **What it does**: Applies dropout to attention weights specifically for ACC positions
- **When to use**: When you want to study how attention to ACC features affects model performance
- **Implementation**: Modifies attention probability matrix after softmax but before weighted sum

### 2. Random ACC Dropout
- **What it does**: Applies random dropout to ACC features in the input before attention computation
- **When to use**: When you want to study the impact of ACC feature noise/perturbation
- **Implementation**: Modifies input hidden states before Q/K/V projection

## Configuration Parameters

### Core Parameters
- `acc_dropout_prob`: Dropout probability for attention weights (0.0 to 1.0)
- `random_acc_dropout_prob`: Dropout probability for input features (0.0 to 1.0)
- `use_acc_dropout`: Enable/disable attention dropout on ACC (boolean)
- `use_random_acc_dropout`: Enable/disable random ACC dropout (boolean)
- `acc_positions`: List of sequence positions that contain ACC features

### Example Configurations

#### Attention Dropout on ACC Only
```python
config = RoFormerConfig(
    # ... other parameters ...
    acc_dropout_prob=0.2,
    random_acc_dropout_prob=0.0,
    use_acc_dropout=True,
    use_random_acc_dropout=False,
    acc_positions=[-1],  # Last token is ACC
)
```

#### Random ACC Dropout Only
```python
config = RoFormerConfig(
    # ... other parameters ...
    acc_dropout_prob=0.0,
    random_acc_dropout_prob=0.3,
    use_acc_dropout=False,
    use_random_acc_dropout=True,
    acc_positions=list(range(10, 20)),  # Positions 10-19 are ACC
)
```

#### Combined Dropout
```python
config = RoFormerConfig(
    # ... other parameters ...
    acc_dropout_prob=0.15,
    random_acc_dropout_prob=0.25,
    use_acc_dropout=True,
    use_random_acc_dropout=True,
    acc_positions=[-5, -4, -3, -2, -1],  # Last 5 tokens are ACC
)
```

## Usage Examples

See `acc_dropout_examples.py` for complete configuration examples:

```python
from src.network.customized_roformer_with_att_ import (
    create_config_with_acc_dropout,
    create_config_with_random_acc_dropout,
    create_config_combined_dropout
)

# Create different configurations
config1 = create_config_with_acc_dropout()
config2 = create_config_with_random_acc_dropout()
config3 = create_config_combined_dropout()
```

## Implementation Details

### Attention Dropout on ACC
```python
def apply_acc_dropout_on_attention(self, self_outputs):
    # Extract attention probabilities
    attention_probs = self_outputs[1]  # [batch_size, num_heads, seq_len, seq_len]
    
    # Create mask for ACC positions
    acc_mask = torch.zeros(seq_len, seq_len)
    for pos in self.acc_positions:
        acc_mask[pos, :] = 1  # Dropout attention from ACC positions
    
    # Apply dropout only to ACC attention weights
    acc_attention = attention_probs * acc_mask
    acc_attention = self.acc_dropout(acc_attention)
    
    # Combine with non-ACC attention weights
    return modified_outputs
```

### Random ACC Dropout
```python
def apply_random_acc_dropout(self, hidden_states):
    # Create mask for ACC positions
    acc_mask = torch.zeros(seq_len)
    for pos in self.acc_positions:
        acc_mask[pos] = 1
    
    # Apply dropout to ACC features
    acc_features = hidden_states * acc_mask
    acc_features = self.random_acc_dropout(acc_features)
    
    # Combine with non-ACC features
    return acc_features + hidden_states * (1 - acc_mask)
```

## Research Questions to Explore

1. **Impact on Performance**: How do different dropout rates affect model accuracy?
2. **Robustness**: Does ACC dropout improve model robustness to ACC noise?
3. **Attention Patterns**: How do attention patterns change with ACC dropout?
4. **Feature Importance**: Which ACC positions are most critical for performance?
5. **Combined Effects**: What happens when both mechanisms are applied together?

## Notes

- ACC positions should be carefully defined based on your specific use case
- Dropout probabilities should be tuned based on your dataset and task
- Consider the trade-off between regularization and information loss
- Monitor both training and validation performance when experimenting
