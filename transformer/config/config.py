class ModelConfig:
    def __init__(
        self,
        vocab_size=50257,  # 词汇表大小
        n_layer=12,  # transformer层数
        n_head=12,   # 注意力头数
        n_embd=768,  # 词嵌入维度
        max_seq_len=1024,  # 最大序列长度
        dropout=0.1,  # dropout率
        bias=True,    # 是否使用偏置
        use_flash_attn=False,  # 是否使用flash attention
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.bias = bias
        self.use_flash_attn = use_flash_attn

        assert n_embd % n_head == 0
        self.head_dim = n_embd // n_head


def get_config(preset: str = "tiny"):
    """
    获取模型配置。

    Args:
        preset: 预设名称，目前支持:
            - "tiny": 一个较小的配置，适合本仓库测试和示例
            - 其他值: 使用 ModelConfig 的默认超参
    """
    if preset == "tiny":
        return ModelConfig(
            vocab_size=10000,
            n_layer=4,
            n_head=4,
            n_embd=256,
            max_seq_len=512
        )
    else:
        # 回退到默认大配置
        return ModelConfig()
