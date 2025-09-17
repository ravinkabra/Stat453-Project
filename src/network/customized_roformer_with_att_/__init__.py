"""
Customized RoFormer with Attention Module

This module provides a customized implementation of RoFormer (Rotary Position Embedding)
with enhanced attention mechanisms for the training framework.

Components:
- CustomizedRoFormerEncoder: Main encoder implementation
- CustomizedRoFormerEncoderParams: Configuration parameters
- RoFormerAttention: Attention mechanism with rotary embeddings
- RoFormerLayer: Transformer layer implementation
- RoFormerSelfAttention: Self-attention with rotary position embeddings
"""

from .config import CustomizedRoFormerEncoderParams
from .network import CustomizedRoFormerEncoder
__all__ = [
    "CustomizedRoFormerEncoderParams",
    "CustomizedRoFormerEncoder",
]
