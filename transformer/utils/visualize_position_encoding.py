# 测试不同位置编码的效果

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
from model.embedding import RotaryPositionEmbedding


def visualize_position_encodings():
    """可视化不同位置编码方式"""
    d_model = 64
    max_len = 50

    # 1. 正弦位置编码
    pe_sinusoidal = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -
                         (math.log(10000.0) / d_model))

    pe_sinusoidal[:, 0::2] = torch.sin(position * div_term)
    pe_sinusoidal[:, 1::2] = torch.cos(position * div_term)

    # 2. 可学习位置编码
    pe_learnable = nn.Embedding(max_len, d_model)

    # 3. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 正弦位置编码的热图
    im1 = axes[0, 0].imshow(pe_sinusoidal.numpy().T,
                            aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Sinusoidal Positional Encoding')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Dimension')
    plt.colorbar(im1, ax=axes[0, 0])

    # 第一个维度的位置编码值
    axes[0, 1].plot(pe_sinusoidal[:, 0].numpy(), label='Dim 0 (sin)')
    axes[0, 1].plot(pe_sinusoidal[:, 1].numpy(), label='Dim 1 (cos)')
    axes[0, 1].set_title('First Two Dimensions')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Encoding Value')
    axes[0, 1].legend()

    # 余弦相似度矩阵（显示位置间的关系）
    similarity = torch.cosine_similarity(
        pe_sinusoidal.unsqueeze(1),
        pe_sinusoidal.unsqueeze(0),
        dim=2
    )
    im2 = axes[1, 0].imshow(similarity.numpy(), cmap='hot', aspect='auto')
    axes[1, 0].set_title('Position Similarity Matrix')
    axes[1, 0].set_xlabel('Position j')
    axes[1, 0].set_ylabel('Position i')
    plt.colorbar(im2, ax=axes[1, 0])

    # 相对位置距离的影响
    pos_distances = []
    for i in range(max_len):
        for j in range(max_len):
            if i != j:
                dist = torch.norm(pe_sinusoidal[i] - pe_sinusoidal[j])
                pos_distances.append((abs(i-j), dist.item()))

    pos_distances = np.array(pos_distances)
    axes[1, 1].scatter(pos_distances[:, 0], pos_distances[:, 1], alpha=0.3)
    axes[1, 1].set_title('Encoding Distance vs Position Distance')
    axes[1, 1].set_xlabel('|i - j|')
    axes[1, 1].set_ylabel('||PE(i) - PE(j)||')

    plt.tight_layout()
    plt.show()


def visualize_causal_mask():
    """可视化因果掩码"""
    seq_len = 10

    # 创建因果掩码
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))

    # 创建带未来信息的掩码对比
    full_mask = torch.ones(seq_len, seq_len)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 因果掩码
    im1 = axes[0].imshow(causal_mask.numpy(), cmap='binary')
    axes[0].set_title('Causal Mask (Lower Triangular)')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')

    # 全连接掩码
    im2 = axes[1].imshow(full_mask.numpy(), cmap='binary')
    axes[1].set_title('Full Attention Mask')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')

    # 注意力分数示例（应用掩码前后）
    # 模拟随机注意力分数
    attention_scores = torch.randn(seq_len, seq_len)

    # 应用因果掩码
    masked_scores = attention_scores.masked_fill(causal_mask == 0, -1e9)
    softmax_scores = torch.softmax(masked_scores, dim=-1)

    im3 = axes[2].imshow(softmax_scores.numpy(), cmap='viridis')
    axes[2].set_title('Attention Weights (with Causal Mask)')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()

    # 打印一个具体示例
    print("Causal Mask example (size 5x5):")
    print(torch.tril(torch.ones(5, 5)))

    print("\nAttention flow for position 3:")
    print("Can attend to positions: [0, 1, 2, 3]")
    print("Cannot attend to positions: [4] (future)")


def test_rope_properties():
    """测试RoPE的性质"""
    dim = 64
    max_len = 20
    rope = RotaryPositionEmbedding(dim=dim, max_seq_len=max_len)

    # 创建随机查询和键
    batch_size = 1
    n_head = 1
    seq_len = 5
    q = torch.randn(batch_size, n_head, seq_len, dim)
    k = torch.randn(batch_size, n_head, seq_len, dim)

    # 应用RoPE
    q_rot, k_rot = rope(q, k, seq_len)

    # 计算内积（注意力分数）
    # 位置i的查询和位置j的键的内积
    scores = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            scores[i, j] = torch.dot(q_rot[0, 0, i], k_rot[0, 0, j])

    print(f"Scores matrix (position i query, position j key):")
    print(scores.numpy())

    # 检查相对位置性质
    print(f"\nChecking RoPE properties:")
    print(f"Score(0,1) = {scores[0, 1]:.4f}")
    print(f"Score(1,2) = {scores[1, 2]:.4f}")
    print(f"Score(2,3) = {scores[2, 3]:.4f}")
    print("Note: Scores for same relative distance should be similar")

    # 可视化内积矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(scores.numpy(), cmap='viridis')
    plt.colorbar()
    plt.title('Attention Scores with RoPE')
    plt.xlabel('Key Position (j)')
    plt.ylabel('Query Position (i)')

    # 标注对角线（相同位置）
    for i in range(seq_len):
        plt.text(i, i, f'{scores[i, i]:.2f}', ha='center', va='center',
                 color='white' if scores[i, i] > 0.5 else 'black')

    plt.show()


# 运行可视化
if __name__ == "__main__":
    print("=== 位置编码与因果掩码可视化 ===\n")

    print("1. 可视化不同位置编码...")
    visualize_position_encodings()

    print("\n2. 可视化因果掩码...")
    visualize_causal_mask()

    print("\n3. 测试RoPE性质...")
    test_rope_properties()
