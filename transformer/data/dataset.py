import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional


class TextDataset(Dataset):
    """文本数据集"""

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        block_size: int = 2,
        stride: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size // 2

        # 编码所有文本
        self.examples = []

        for text in texts:
            token_ids = self.tokenizer.encode(text)

            # 使用滑动窗口创建训练样本
            for i in range(0, len(token_ids) - block_size + 1, self.stride):
                chunk = token_ids[i:i + block_size]
                if len(chunk) == block_size:
                    self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        token_ids = self.examples[idx]

        # 这里返回完整的序列，由模型内部负责 shift
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = torch.tensor(token_ids, dtype=torch.long)

        return input_ids, labels


def create_data_loader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
