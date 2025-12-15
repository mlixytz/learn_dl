import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """ 前馈网络
        使用GELU激活和门控机制。
        前馈网络的作用：注意力计算各token的关系（专家会议讨论关系），前馈网络对关系进行梳理提取特征（会议纪要整理员提炼升华）
    """

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd  # FFN 中间层维度通常是4倍

        self.gate_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """ 使用SwiGLU 激活：SwiGLU(x) = swish(xW + b) ⊗ (xV + c) """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)

        # 逐元素相乘(门控)
        hidden = gate * up

        # 下投影
        output = self.down_proj(hidden)
        output = self.dropout(output)

        return output


class TransformerBlock(nn.Module):
    """ 完整的Transformer Block 
        使用Pre-LN架构（层归一化在子层前)
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-LN: 输入归一化
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)

        # 注意力层和前馈层
        self.attn = CausalSelfAttention(config)
        self.mlp = FeedForward(config)

        # 残差连接权重
        self.residual_weight = nn.Parameter(
            torch.ones(1)) if config.bias else None

    def forward(self, x, attn_mask=None, use_cache=None, past_key_value=None):
        """ 前向传播
        Args:
            x: [batch_size, seq_len, n_embd]
            attn_mask: 注意力掩码
            use_cache: 是否使用KV缓存
            past_key_value: 缓存的K和V
        Returns:
            输出张量和新的KV缓存
        """

        residual = x

        # 1. 注意力子层
        # Pre-LN: 先归一化再输入注意力层
        x_norm = self.ln_1(x)
        attn_output, present_key_value = self.attn(
            x_norm, attn_mask, use_cache, past_key_value)
        if self.residual_weight is not None:
            x = residual + self.residual_weight * attn_output
        else:
            x = residual + attn_output

        residual = x

        # 2. 前馈子层
        # Pre-LN: 先归一化再输入FFN
        x_norm = self.ln_2(x)
        mlp_output = self.mlp(x_norm)

        # 残差连接
        if self.residual_weight is not None:
            x = residual + self.residual_weight * mlp_output
        else:
            x = residual + mlp_output

        return x, present_key_value
