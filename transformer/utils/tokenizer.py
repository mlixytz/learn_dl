# utils/tokenizer.py
from typing import List, Optional
import json
import os


class Tokenizer:
    """简单的分词器基类"""

    def __init__(self, vocab_file: Optional[str] = None):
        self.vocab = {}
        self.inverse_vocab = {}

        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file: str):
        """加载词汇表"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        if isinstance(vocab_data, dict):
            self.vocab = vocab_data
        else:
            self.vocab = {token: idx for idx, token in enumerate(vocab_data)}

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """编码文本为token IDs"""
        # 简单实现：按字符分割
        # 实际应用中应使用更复杂的分词算法
        return [self.vocab.get(char, 0) for char in text]

    def decode(self, token_ids: List[int]) -> str:
        """解码token IDs为文本"""
        return ''.join([self.inverse_vocab.get(token_id, '') for token_id in token_ids])

    def save_vocab(self, vocab_file: str):
        """保存词汇表"""
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)


class CharTokenizer(Tokenizer):
    """ 字符级分词器 """

    def __init__(self, corpus: Optional[str] = None):
        super().__init__()

        # 特殊token
        special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }

        self.vocab = special_tokens.copy()

        if corpus:
            self.build_vocab(corpus)

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def build_vocab(self, corpus: str, max_vocab_size: int = 10000):
        """从语料库构建词汇表"""
        # 统计字符频率
        char_freq = {}
        for char in corpus:
            char_freq[char] = char_freq.get(char, 0) + 1

        # 按频率排序
        sorted_chars = sorted(
            char_freq.items(), key=lambda x: x[1], reverse=True)

        # 添加到词汇表（保留特殊token的位置）
        next_idx = len(self.vocab)
        for char, _ in sorted_chars:
            if char not in self.vocab and next_idx < max_vocab_size:
                self.vocab[char] = next_idx
                next_idx += 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本"""
        if add_special_tokens:
            text = '<bos>' + text + '<eos>'

        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            elif char.startswith('<') and char.endswith('>'):
                # 特殊token
                tokens.append(self.vocab.get(char, 1))  # 默认为<unk>
            else:
                tokens.append(self.vocab.get(char, 1))  # <unk>

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码token IDs"""
        tokens = []
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, '')
            if skip_special_tokens and token.startswith('<') and token.endswith('>'):
                continue
            tokens.append(token)

        return ''.join(tokens)
