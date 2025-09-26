"""
1D UNet Decoder for Music Token Generation
专门为音乐子序列设计，建模同时间音乐事件的并行关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Conv1DBlock(nn.Module):
    """1D UNet的基础卷积块"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Downsample1D(nn.Module):
    """1D下采样"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """1D上采样"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Music1DUNetDecoder(nn.Module):
    """
    1D UNet Decoder for Music Subsequence Generation

    设计理念：
    - 建模音乐子序列中同时间token的并行关系
    - 通过1D卷积建模局部音乐结构
    - 跳跃连接保留多尺度音乐特征
    - 全局条件融入提供上下文信息
    """

    def __init__(self, hidden_size: int, subseq_len: int, base_channels: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.subseq_len = subseq_len

        # 条件投影
        self.condition_proj = nn.Linear(hidden_size, hidden_size)

        # Encoder (下采样路径)
        self.enc1 = Conv1DBlock(hidden_size, base_channels)
        self.enc2 = Conv1DBlock(base_channels, base_channels * 2)
        self.enc3 = Conv1DBlock(base_channels * 2, base_channels * 4)

        self.down1 = Downsample1D(base_channels, base_channels)
        self.down2 = Downsample1D(base_channels * 2, base_channels * 2)

        # Bottleneck
        self.bottleneck = Conv1DBlock(base_channels * 4, base_channels * 8)

        # Decoder (上采样路径)
        self.up2 = Upsample1D(base_channels * 8, base_channels * 4)
        self.up1 = Upsample1D(base_channels * 4, base_channels * 2)

        self.dec3 = Conv1DBlock(base_channels * 8, base_channels * 4)  # bottleneck + enc3
        self.dec2 = Conv1DBlock(base_channels * 4, base_channels * 2)  # up2 + enc2
        self.dec1 = Conv1DBlock(base_channels * 2, base_channels)  # up1 + enc1

        # 输出投影
        self.out_proj = nn.Conv1d(base_channels, hidden_size, kernel_size=1)

    def forward(self, emb: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            emb: [batch*seq, subseq_len, hidden_size] - 输入embeddings
            condition: [batch*seq, hidden_size] - 全局条件（可选）

        Returns:
            [batch*seq, subseq_len, hidden_size] - 解码后的特征
        """
        batch_seq, seq_len, hidden = emb.shape
        assert seq_len == self.subseq_len, f"Expected seq_len {self.subseq_len}, got {seq_len}"

        # 转换到1D卷积格式: [B*S, H, L]
        x = emb.permute(0, 2, 1)  # [B*S, hidden, subseq_len]

        # 融入全局条件
        if condition is not None:
            cond_proj = self.condition_proj(condition)  # [B*S, hidden]
            cond_proj = cond_proj.unsqueeze(-1)  # [B*S, hidden, 1]
            x = x + cond_proj  # 广播相加

        # Encoder路径
        enc1_out = self.enc1(x)  # [B*S, base, L]
        x = self.down1(enc1_out)  # [B*S, base, L//2]

        enc2_out = self.enc2(x)  # [B*S, base*2, L//2]
        x = self.down2(enc2_out)  # [B*S, base*2, L//4]

        enc3_out = self.enc3(x)  # [B*S, base*4, L//4]

        # Bottleneck
        x = self.bottleneck(enc3_out)  # [B*S, base*8, L//4]

        # Decoder路径（带跳跃连接）
        x = self.up2(x)  # [B*S, base*4, L//2]
        x = torch.cat([x, enc3_out], dim=1)  # 跳跃连接
        x = self.dec3(x)  # [B*S, base*4, L//2]

        x = self.up1(x)  # [B*S, base*2, L]
        x = torch.cat([x, enc2_out], dim=1)  # 跳跃连接
        x = self.dec2(x)  # [B*S, base*2, L]

        # 最终上采样和跳跃连接
        x = torch.cat([x, enc1_out], dim=1)  # 跳跃连接
        x = self.dec1(x)  # [B*S, base, L]

        # 输出投影
        x = self.out_proj(x)  # [B*S, hidden, L]

        # 转换回序列格式: [B*S, L, hidden]
        output = x.permute(0, 2, 1)

        return output


# 带Causal Attention的版本（用于对比）
class CausalAttention1D(nn.Module):
    """简化的Causal Self-Attention"""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        # 生成Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)

        # Attention计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        # 应用attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)

        return self.out_proj(attn_output)


class Music1DUNetWithCausalAttention(nn.Module):
    """
    1D UNet + Causal Attention 混合架构
    结合了UNet的空间建模和Transformer的时序建模
    """

    def __init__(self, hidden_size: int, subseq_len: int, base_channels: int = 128, num_heads: int = 8):
        super().__init__()
        self.unet = Music1DUNetDecoder(hidden_size, subseq_len, base_channels)
        self.causal_attn = CausalAttention1D(hidden_size, num_heads)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, emb: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 先通过UNet建模空间关系
        x = self.unet(emb, condition)

        # 再通过Causal Attention建模时序关系
        x = self.norm(x + self.causal_attn(x))

        return x
