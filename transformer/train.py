import torch
import torch.optim as optim
import time
import os
import random

from data.dataset import TextDataset, create_data_loader
from torch.utils.data import DataLoader
from config.config import get_config
from model.gpt import GPTModel
from utils.tokenizer import CharTokenizer
from typing import Optional, List


class Trainer:
    """GPT训练器"""

    def __init__(
        self,
        model: GPTModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        grad_clip: float = 1.0,
        log_interval: int = 10,
        save_dir: str = "checkpoints"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_clip = grad_clip
        self.log_interval = log_interval
        self.save_dir = save_dir

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 100,  # 假设训练100个epoch
            eta_min=lr * 0.1
        )

        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if "cuda" in device else None

        # 训练状态
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        print(f"训练器初始化完成，使用设备: {device}")
        print(f"优化器: AdamW (lr={lr}, weight_decay={weight_decay})")
        if self.scaler:
            print("使用混合精度训练")

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (input_ids, labels) in enumerate(self.train_loader):
            # 移动到设备
            input_ids = input_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 梯度清零
            self.optimizer.zero_grad()

            # 混合精度训练
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, labels=labels)
                    loss = outputs["loss"]

                # 缩放损失并反向传播
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )

                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 普通训练
                outputs = self.model(input_ids, labels=labels)
                loss = outputs["loss"]

                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )

                self.optimizer.step()

            # 学习率调度
            self.scheduler.step()

            # 统计损失
            total_loss += loss.item()

            # 日志记录
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * input_ids.size(0) / elapsed

                print(f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Samples/sec: {samples_per_sec:.1f}")

        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        if self.val_loader is None:
            return float('inf')

        self.model.eval()
        total_loss = 0

        for input_ids, labels in self.val_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_ids, labels=labels)
            loss = outputs["loss"]

            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.model.config.__dict__,
            'val_loss': val_loss,
        }

        # 保存常规检查点
        checkpoint_path = os.path.join(
            self.save_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path}")

    def train(self, num_epochs: int = 100, save_every: int = 10):
        """主训练循环"""
        print(f"开始训练，共{num_epochs}个epoch")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")

            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"训练损失: {train_loss:.4f}")

            # 验证
            if self.val_loader:
                val_loss = self.validate()
                print(f"验证损失: {val_loss:.4f}")

                # 检查是否是最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"新的最佳验证损失: {val_loss:.4f}")
            else:
                val_loss = float('inf')
                is_best = False

            # 保存检查点
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)

            # 打印学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")

        print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.4f}")

        # 保存最终模型
        self.save_checkpoint(num_epochs, val_loss)

        return self.train_losses, self.val_losses


def prepare_training_data(
    train_texts: List[str],
    val_texts: Optional[List[str]] = None,
    tokenizer_class=CharTokenizer,
    block_size: int = 256,
    batch_size: int = 4,
    val_split: float = 0.1
):
    """准备训练数据"""
    # 创建分词器
    all_texts = train_texts + (val_texts or [])
    tokenizer = tokenizer_class(corpus=''.join(all_texts))

    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 分割训练/验证集
    if val_texts is None and val_split > 0:
        split_idx = int(len(train_texts) * (1 - val_split))
        train_texts, val_texts = train_texts[:
                                             split_idx], train_texts[split_idx:]

    # 创建数据集
    train_dataset = TextDataset(train_texts, tokenizer, block_size)

    if val_texts:
        val_dataset = TextDataset(val_texts, tokenizer, block_size)
    else:
        val_dataset = None

    print(f"训练样本数: {len(train_dataset)}")
    if val_dataset:
        print(f"验证样本数: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = create_data_loader(train_dataset, batch_size=batch_size)
    val_loader = create_data_loader(
        val_dataset, batch_size=batch_size) if val_dataset else None

    return train_loader, val_loader, tokenizer


def main():
    """主训练函数"""
    # 配置
    config = get_config()

    # 加载数据（示例数据）
    print("准备数据...")

    # 示例数据：莎士比亚作品片段
    sample_text = """
    To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th'oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovere'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience doth make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pith and moment
With this regard their currents turn awry
And lose the name of action.
    """ * 2

    # 重复数据以增加数据集大小
    train_texts = [sample_text[:random.randint(
        config.max_seq_len, len(sample_text))]] * 100
    val_texts = [sample_text[:random.randint(
        config.max_seq_len, len(sample_text))]] * 20

    # 准备数据
    train_loader, val_loader, tokenizer = prepare_training_data(
        train_texts=train_texts,
        val_texts=val_texts,
        tokenizer_class=CharTokenizer,
        block_size=config.max_seq_len,
        batch_size=4
    )

    # 创建模型
    print("\n创建模型...")
    model = GPTModel(config)

    # 创建训练器
    print("\n创建训练器...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=6e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        log_interval=5,
        save_dir="checkpoints"
    )

    # 训练
    print("\n开始训练...")
    train_losses, val_losses = trainer.train(
        num_epochs=50,
        save_every=10
    )

    # 保存分词器
    tokenizer.save_vocab("vocab.json")
    print(f"分词器已保存: vocab.json")

    return model, tokenizer, train_losses, val_losses


if __name__ == "__main__":
    model, tokenizer, train_losses, val_losses = main()
