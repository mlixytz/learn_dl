import torch
from config.config import get_config
from model.block import TransformerBlock
from utils.helpers import create_causal_mask, count_parameters


def test_transformer_block():
    """测试Transformer Block"""
    # 使用tiny配置
    config = get_config()
    print(f"配置: {config.n_layer}层, {config.n_head}头, 维度{config.n_embd}")

    # 创建Block
    block = TransformerBlock(config, layer_idx=0)
    print(f"Block参数数量: {count_parameters(block):,}")

    # 测试输入
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, config.n_embd)

    # 创建因果掩码
    causal_mask = create_causal_mask(seq_len, x.device)

    # 前向传播（训练模式）
    block.train()
    output, kv_cache = block(x, attn_mask=causal_mask, use_cache=True)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出与输入形状相同: {output.shape == x.shape}")

    # 测试推理模式（使用KV缓存）
    block.eval()
    with torch.no_grad():
        # 第一次前向
        output1, kv_cache1 = block(
            x, attn_mask=causal_mask, use_cache=True)

        # 模拟下一个token推理（只输入最后一个token）
        next_token = torch.randn(batch_size, 1, config.n_embd)
        output2, kv_cache2 = block(
            next_token,
            attn_mask=None,  # 注意：单token时不需要掩码
            use_cache=True,
            past_key_value=kv_cache1
        )
        print(f"单token推理输出形状: {output2.shape}")

    # 梯度检查
    if x.requires_grad:
        loss = output.mean()
        loss.backward()
        print("梯度计算成功")

    print("✓ Transformer Block测试通过")


if __name__ == "__main__":
    test_transformer_block()
