import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import GPTEmbedding
from .block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 词嵌入层
        self.embedding = GPTEmbedding(config)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layer)
        ])

        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.n_embd, eps=1e-5)

        # 语言模型头（与词嵌入共享权重）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.token_embedding.weight

        # 使用_init_weights方法递归初始化所有模型的参数，包括embedding、lm_head、blocks、ln_f
        self.apply(self._init_weights)

        self._calculate_params()

    def _init_weights(self, module):
        """ 初始化权重 """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _calculate_params(self):
        """计算并打印模型参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        print(f"模型总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        # 分解参数量
        print("\n模型参数分解:")
        print(
            f"  词嵌入: {sum(p.numel() for p in self.embedding.parameters()):,}")
        print(
            f"  Transformer Blocks ({self.config.n_layer}个): {sum(p.numel() for p in self.blocks.parameters()):,}")
        print(f"  最终层归一化: {sum(p.numel() for p in self.ln_f.parameters()):,}")
        print(
            f"  LM Head: {sum(p.numel() for p in self.lm_head.parameters()):,}")

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=None, past_key_values=None):
        """ 前向传播

        Args:
            input_ids: 输入的token ids [batch_size, seq_len]
            attention_mask: 注意力掩码，用于处理变长序列
            labels: 标签序列，用于计算损失
            use_cache: 是否使用缓存
            past_key_values: 缓存的键值对
        """
        batch_size, seq_len = input_ids.shape

        # 1. 获取嵌入
        # [batch_size, seq_len, n_embd]
        hidden_states = self.embedding(input_ids)

        # 2. 创建因果注意力掩码
        past_kv_len = 0 if past_key_values is None else past_key_values[0][0].shape[-1]
        causal_mask = self._prepare_causal_mask(
            seq_len, past_key_values_length=past_kv_len
        )
        # causal_mask: [1, 1, seq_len, past_kv_len + seq_len]

        # 3. 如果有 attention_mask，合并 padding 掩码
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len] (1=保留, 0=padding)
            # 需要扩展为 [batch_size, 1, 1, seq_len] 再广播到 total_seq_len
            total_seq_len = past_kv_len + seq_len
            # 扩展 attention_mask
            # [batch, 1, 1, seq_len]
            extended_attention_mask = attention_mask[:, None, None, :]
            # 对于 past tokens，默认都是有效的
            if past_kv_len > 0:
                past_mask = torch.ones(
                    batch_size, 1, 1, past_kv_len,
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                extended_attention_mask = torch.cat(
                    [past_mask, extended_attention_mask], dim=-1)
            # 转为加性掩码：1->0, 0->-inf
            extended_attention_mask = (
                1.0 - extended_attention_mask) * torch.finfo(hidden_states.dtype).min
            # 合并因果掩码和 padding 掩码
            causal_mask = causal_mask + extended_attention_mask

        # 4. 通过Transformer Blocks
        present_key_values = [] if use_cache else None
        for block_idx, block in enumerate(self.blocks):
            # 获取当前层的past_key_values
            layer_past = None
            if past_key_values is not None:
                layer_past = past_key_values[block_idx]

            # 前向传播
            hidden_states, layer_present = block(
                hidden_states, causal_mask, use_cache=use_cache, past_key_value=layer_past
            )

            if use_cache:
                present_key_values.append(layer_present)

        # 5. 最终层归一化
        hidden_states = self.ln_f(hidden_states)

        # 6. 语言模型头
        # [batch_size, seq_len, vocab_size]
        logits = self.lm_head(hidden_states)

        # 7. 计算损失（如果有标签）
        loss = None
        if labels is not None:
            # 移位logits 和 labels 用于下一个token预测
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 计算交叉熄损失
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # 返回结果
        output = {
            'logits': logits,
            'loss': loss,
        }

        if use_cache:
            output['past_key_values'] = present_key_values
        return output

    def _prepare_causal_mask(self, seq_len, past_key_values_length=0):
        """ 创建因果注意力掩码

        Args:
            seq_len: 当前输入序列长度
            past_key_values_length: 过去的KV缓存长度

        Returns:
            causal_mask: 因果注意力掩码 [1, 1, seq_len, total_seq_len]
                         0 表示可见，-inf 表示屏蔽，可直接加到注意力分数上
        """
        device = next(self.parameters()).device
        total_seq_len = past_key_values_length + seq_len

        # key 的位置 [0, 1, ..., total_seq_len-1]
        key_positions = torch.arange(total_seq_len, device=device)
        # 当前这次前向中 query 的绝对位置：
        # past 部分已经在缓存里，所以当前 query 从 past_key_values_length 开始
        query_positions = past_key_values_length + \
            torch.arange(seq_len, device=device)

        # allowed[i, j] = True 表示第 i 个 query 可以看到第 j 个 key
        # 条件：key_position <= query_position
        allowed = key_positions.unsqueeze(
            0) <= query_positions.unsqueeze(1)  # [seq_len, total_seq_len]

        # 转成加性掩码：0 / -inf
        mask = torch.zeros(seq_len, total_seq_len, device=device)
        mask.masked_fill_(~allowed, float('-inf'))

        # 扩展 batch 和 head 维度，供广播使用
        return mask.unsqueeze(0).unsqueeze(0)

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=1.0, repetition_penalty=1.0, do_sample=False, **kwargs):
        """ 生成文本

            Args:
                input_ids: 输入的token ids [batch_size, seq_len]
                max_new_tokens: 最大生成长度
                temperature: 温度参数(越高越随机)
                top_k: top-k采样
                top_p: top-p （核）采样
                repetition_penalty: 重复惩罚参数
                do_sample: 是否使用采样

            Returns:
                generated_ids: 生成的token ids [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        batch_size = input_ids.shape[0]

        # 复制输入作为生成的基础
        generated = input_ids
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 前向传播（使用KV缓存）
                outputs = self(
                    generated if past_key_values is None else generated[:, -1:], use_cache=True, past_key_values=past_key_values)
                # 更新KV缓存
                past_key_values = outputs.get('past_key_values')

                logits = outputs['logits']
                new_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]

                # 应用重复惩罚
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            new_token_logits[i, token_id] /= repetition_penalty

                # 应用温度
                if temperature != 1.0:
                    new_token_logits /= temperature

                # 应用top-k采样
                if top_k is not None and top_k > 0:
                    # 最终shape[batch_size, vocab_size]，topk的结果是[batch_size, top_k] -> [batch_size, 1]
                    indices_to_remove = new_token_logits < torch.topk(
                        new_token_logits, top_k)[0][..., -1, None]
                    new_token_logits[indices_to_remove] = float('-inf')

                # 应用top-p(核)采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        new_token_logits, descending=True)

                    # 累加概率
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1)

                    # 移除积累概率超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保留第一个超过阈值的token
                    sorted_indices_to_remove[...,
                                             1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                        new_token_logits[indices_to_remove] = float('-inf')

                # 采样或贪心解码
                if do_sample:
                    # 采样
                    new_token_logits = F.softmax(new_token_logits, dim=-1)
                    next_token = torch.multinomial(
                        new_token_logits, num_samples=1).squeeze(1)
                else:
                    # 贪心解码
                    next_token = torch.argmax(new_token_logits, dim=-1)

                # 添加新 token 到序列
                generated = torch.cat(
                    [generated, next_token.unsqueeze(-1)], dim=-1)

            return generated
