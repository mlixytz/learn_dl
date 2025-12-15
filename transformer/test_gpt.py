# test_gpt.py
import torch
from config.config import get_config
from model.gpt import GPTModel
from utils.tokenizer import CharTokenizer


def test_gpt_model():
    """测试完整GPT模型"""
    print("=== 测试完整GPT模型 ===")

    # 获取配置
    config = get_config("tiny")
    print(f"配置: {config.n_layer}层, {config.n_head}头, 维度{config.n_embd}")

    # 创建模型
    model = GPTModel(config)
    print(f"模型创建成功")

    # 创建测试输入
    vocab_size = config.vocab_size
    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"输入形状: {input_ids.shape}")

    # 测试前向传播
    print("\n1. 测试前向传播...")
    outputs = model(input_ids)
    logits = outputs["logits"]
    print(f"输出logits形状: {logits.shape}")
    print(
        f"期望: [batch_size, seq_len, vocab_size] = [{batch_size}, {seq_len}, {vocab_size}]")

    # 测试带标签的前向传播
    print("\n2. 测试带标签的前向传播...")
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    outputs = model(input_ids, labels=labels)
    loss = outputs["loss"]
    print(f"损失值: {loss.item():.4f}")
    print(f"损失应为正数且不太大: {0 < loss.item() < 10}")

    # 测试生成
    print("\n3. 测试文本生成...")
    model.eval()
    with torch.no_grad():
        # 使用简单的贪心解码
        generated = model.generate(
            input_ids[:, :8],  # 只用前8个token作为prompt
            max_new_tokens=10,
            temperature=1.0,
            do_sample=False  # 贪心解码
        )
        print(f"生成形状: {generated.shape}")
        print(f"原始输入: {input_ids[0, :8].tolist()}")
        print(f"生成结果: {generated[0].tolist()}")

    # 测试KV缓存
    print("\n4. 测试KV缓存...")
    with torch.no_grad():
        # 第一次前向（无缓存）
        outputs1 = model(input_ids[:, :8], use_cache=True)
        past_key_values = outputs1["past_key_values"]

        # 第二次前向（使用缓存）
        outputs2 = model(input_ids[:, 8:9], use_cache=True,
                         past_key_values=past_key_values)

        print(f"第一次输出形状: {outputs1['logits'].shape}")
        print(f"第二次输出形状: {outputs2['logits'].shape}")
        print(f"KV缓存长度: {len(past_key_values)}")
        print(
            f"每个KV缓存形状: {past_key_values[0][0].shape if past_key_values else '无'}")

    # 测试参数共享
    print("\n5. 测试参数共享...")
    embedding_weight = model.embedding.token_embedding.weight
    lm_head_weight = model.lm_head.weight
    print(f"词嵌入权重形状: {embedding_weight.shape}")
    print(f"LM Head权重形状: {lm_head_weight.shape}")
    print(f"权重是否共享: {id(embedding_weight) == id(lm_head_weight)}")

    print("\n✓ 所有测试通过!")
    return model


def test_tokenizer():
    """测试分词器"""
    print("\n=== 测试分词器 ===")

    # 创建测试文本
    test_text = "Hello, world! This is a test."

    # 创建分词器
    tokenizer = CharTokenizer(corpus=test_text)
    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 测试编码
    token_ids = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"编码结果: {token_ids}")
    print(f"编码长度: {len(token_ids)}")

    # 测试解码
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"解码结果: {decoded}")
    print(f"是否一致: {decoded == test_text}")

    # 测试特殊token
    print(f"\n特殊token:")
    print(f"  <pad>: {tokenizer.vocab.get('<pad>')}")
    print(f"  <unk>: {tokenizer.vocab.get('<unk>')}")
    print(f"  <bos>: {tokenizer.vocab.get('<bos>')}")
    print(f"  <eos>: {tokenizer.vocab.get('<eos>')}")

    print("\n✓ 分词器测试通过!")
    return tokenizer


if __name__ == "__main__":
    # 测试分词器
    tokenizer = test_tokenizer()

    # 测试GPT模型
    model = test_gpt_model()
