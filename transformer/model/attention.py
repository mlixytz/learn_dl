import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .embedding import RotaryPositionEmbedding


class CausalSelfAttention(nn.Module):
    """ 因果自注意力机制（decoder-only) 

        QKV作用: Q是问题，K是索引 V是答案。Q和K决定“关注哪儿”，V决定“关注什么”
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # QKV 投影
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # 输出投影，为多头特征融合
        self.out_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias)

        # 注意力dropout
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.rope = RotaryPositionEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=10000
        )

    def _split_heads(self, x):
        """ 将投影后的张量分割为多头

            多头的好处：并行计算，每个头关注不同的信息，最后合并输出。而单个头多个模式之间可能冲突，关注不到所有信息，且运算效率低

        Args:
            x: [batch_size, seq_len, n_embd]
            is_key: 是否是K, K需要转置用于计算主力分数
        Returns:
            [batch_size, n_head, seq_len, head_dim]
        """

        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.config.n_head,
                   self.config.head_dim)

        return x.transpose(1, 2)

    def _merge_heads(self, x):
        """合并多头输出
        Args:
            x: [batch_size, n_head, seq_len, head_dim]
        Returns:
            [batch_size, seq_len, n_embd]
        """
        batch_size, n_head, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, n_head * head_dim)

    def _scale_dot_product_attention(self, q, k, v, attn_mask=None):
        """ 缩放点积注意力计算
        Args:
            q: [batch_size, n_head, seq_len, head_dim]
            k: [batch_size, n_head, head_dim, kv_len]
            v: [batch_size, n_head, kv_len, head_dim]
            attn_mask: 注意力掩码
        Returns:
            注意力输出
        """
        batch_size, n_head, q_seq_len, head_dim = q.shape
        _, _, kv_len, _ = v.shape

        # 注意力计算分数: Q @ K^T / sqrt(d_k), sqrt(d_k) 是为了避免点积过大导致梯度消失
        # [batch, n_head, q_seq_len, kv_len]
        attn_scores = torch.matmul(q, k) / math.sqrt(head_dim)

        # 应用因果掩码（确保只能看到全面的token)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # 加权求和 attn_weights @ v
        # [batch, n_head, q_seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        return attn_output

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

        _, seq_len, _ = x.shape

        # 1. 投影QKV
        q = self.q_proj(x)  # [batch, seq_len, n_embd]
        k = self.k_proj(x)  # [batch, seq_len, n_embd]
        v = self.v_proj(x)  # [batch, seq_len, n_embd]

        # 2. 分割多头
        q = self._split_heads(q)  # [batch, n_head, seq_len, head_dim]
        k = self._split_heads(k)  # [batch, n_head, seq_len, head_dim]
        v = self._split_heads(v)  # [batch, n_head, seq_len, head_dim]

        # 3. 应用旋转位置编码
        q, k = self.rope(q, k, seq_len)

        # 4. K 转置用于注意力计算
        k = k.transpose(2, 3)  # [batch, n_head, head_dim, seq_len]

        # 4. 处理kv缓存(推理优化)
        if use_cache and past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-1)  # 在序列维度拼接
            v = torch.cat([past_v, v], dim=2)  # 在序列维度拼接

        new_kv_cache = (k, v) if use_cache else None

        # 5. 计算注意力
        attn_output = self._scale_dot_product_attention(
            q, k, v, attn_mask)

        # 6. 合并多头并输出投影
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_drop(attn_output)

        return attn_output, new_kv_cache
