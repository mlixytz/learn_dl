import torch
import torch.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """ 旋转位置编码 

        将位置信息编码为旋转角度，每个位置旋转角度不同，相对位置体现在旋转角度差

        对于位置m的查询q_m和位置n的键k_n, R为旋转矩阵
        q_m' = R_m * q_m
        k_n' = R_n * k_n
        score = (q_m')^T * k_n' = q_m^T * R_{n-m} * k_n # 注意力分数
    """

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.base = base
        self.dim = dim

        # 预计算旋转矩阵
        self._build_cos_sin_cache()

    def _build_cos_sin_cache(self):
        """ 构建旋转位置编码的cos/sin缓存 """
        # 计算 theta 值：theta_i = base^(-2i/dim)
        inv_freq = 1.0 / \
            (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))

        # 生成位置序列
        t = torch.arange(self.max_seq_len, device=inv_freq.device)

        # 计算频率外积 pos * inv_freq
        freqs = torch.einsum("i,j -> ij", t, inv_freq)

        # 扩展为复数形式 cosθ + i*sinθ
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]

        # 缓存cos和sin值，形状: [1, 1, seq_len, dim]
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        """ 将最后一半维度旋转，实现[x1, x2] -> [-x2, x1] """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, x, seq_len):
        """
        应用旋转位置编码
        Args:
            x: [batch, n_head, seq_len, head_dim]
            seq_len: 序列长度
        Returns:
            [batch, n_head, seq_len, head_dim]
        """
        cos = self.cos_cached[:, :, :seq_len, :]  # [1, 1, seq_len, head_dim]
        sin = self.sin_cached[:, :, :seq_len, :]  # [1, 1, seq_len, head_dim]
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, q, k, seq_len):
        """
        Args:
            q: [batch, n_head, seq_len, head_dim]
            k: [batch, n_head, seq_len, head_dim]
            seq_len: 序列长度
        Returns:
            q_rot: [batch, n_head, seq_len, head_dim]
            k_rot: [batch, n_head, seq_len, head_dim]
        """
        q_rot = self.apply_rotary_pos_emb(q, seq_len)
        k_rot = self.apply_rotary_pos_emb(k, seq_len)
        return q_rot, k_rot


class GPTEmbedding(nn.Module):
    """ GPT的词嵌入层（包含位置编码）"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(
            config.max_seq_len, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, n_embd]
        """
        batch_size, seq_len = input_ids.shape

        # 创建位置索引
        position_ids = torch.arange(
            0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 获取词嵌入和位置嵌入
        token_embeds = self.token_embedding(
            input_ids)  # [batch, seq_len, n_embd]
        position_embeds = self.position_embedding(
            position_ids)  # [batch, seq_len, n_embd]

        # 合并并应用dropout
        embeddings = self.dropout(token_embeds + position_embeds)

        return embeddings
