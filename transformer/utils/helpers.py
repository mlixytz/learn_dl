# utils/helpers.py
import torch
import torch.nn as nn


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """创建因果注意力掩码（加性掩码，0 为可见，-inf 为屏蔽）

    Args:
        seq_len: 序列长度
        device: 设备

    Returns:
        因果掩码 [1, 1, seq_len, seq_len]，可直接加到注意力分数上
    """
    # 下三角为 1，上三角为 0
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

    # 1 -> 0（保留），0 -> -inf（屏蔽）
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)

    # 扩展为 [1, 1, seq_len, seq_len] 以便广播
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_memory_usage(model: nn.Module, batch_size: int, seq_len: int) -> float:
    """估算模型显存使用量（MB）"""
    # 参数内存
    param_memory = sum(p.numel() * p.element_size()
                       for p in model.parameters())

    # 梯度内存（训练时）
    grad_memory = param_memory

    # 激活内存（近似估算）
    # 注意：这是简化估算，实际会更复杂
    config = model.config
    attention_memory = batch_size * seq_len * config.n_embd * 2  # Q和K

    total_bytes = param_memory + grad_memory + attention_memory
    return total_bytes / (1024 ** 2)  # 转换为MB


def initialize_weights(module: nn.Module):
    """初始化模型权重"""
    if isinstance(module, nn.Linear):
        # 使用GPT-2的初始化方案
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)
