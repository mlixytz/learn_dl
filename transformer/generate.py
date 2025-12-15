import torch
import argparse
from typing import Optional
import sys

from config.config import get_config
from model.gpt import GPTModel
from utils.tokenizer import CharTokenizer, Tokenizer


def load_model_and_tokenizer(
    checkpoint_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """加载模型和分词器"""
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 创建配置
    config_dict = checkpoint['config']
    config = type('Config', (), config_dict)()  # 动态创建配置对象

    # 创建模型
    model = GPTModel(config)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 加载分词器
    tokenizer = CharTokenizer()
    tokenizer.load_vocab("vocab.json")

    print(f"模型加载成功: {checkpoint_path}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"使用设备: {device}")

    return model, tokenizer


def generate_text(
    model: GPTModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    device: str = "cpu"
):
    """生成文本"""
    # 编码prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # 生成
    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True
    )

    # 解码
    generated_text = tokenizer.decode(
        generated_ids[0].tolist(), skip_special_tokens=True)

    return generated_text


def interactive_generation(
    model: GPTModel,
    tokenizer: Tokenizer,
    device: str = "cpu"
):
    """交互式文本生成"""
    print("\n" + "="*50)
    print("GPT文本生成器 (输入'quit'退出)")
    print("="*50)

    while True:
        try:
            # 获取用户输入
            prompt = input("\n输入提示语: ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break

            if not prompt.strip():
                print("提示语不能为空！")
                continue

            # 生成参数
            try:
                max_tokens = int(input("最大生成长度 (默认100): ") or 100)
                temperature = float(input("温度 (默认0.8): ") or 0.8)
                top_k = input("top_k (默认50, 输入0关闭): ")
                top_k = int(top_k) if top_k.strip() else 50
                top_p = float(input("top_p (默认0.95): ") or 0.95)
                repetition_penalty = float(input("重复惩罚 (默认1.1): ") or 1.1)
            except ValueError:
                print("参数输入错误，使用默认值")
                max_tokens = 100
                temperature = 0.8
                top_k = 50
                top_p = 0.95
                repetition_penalty = 1.1

            # 生成文本
            print("\n生成中...")

            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                device=device
            )

            # 显示结果
            print("\n" + "="*50)
            print("生成结果:")
            print("="*50)
            print(generated_text)
            print("="*50)

        except KeyboardInterrupt:
            print("\n\n中断生成")
            break
        except Exception as e:
            print(f"\n生成错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="GPT文本生成")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="模型检查点路径")
    parser.add_argument("--prompt", type=str, default="To be, or not to be",
                        help="生成提示语")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="温度参数")
    parser.add_argument("--top_k", type=int, default=50,
                        help="top-k采样参数")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="top-p采样参数")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="重复惩罚参数")
    parser.add_argument("--interactive", action="store_true",
                        help="交互式生成模式")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备 (cuda/cpu)")

    args = parser.parse_args()

    # 加载模型和分词器
    try:
        model, tokenizer = load_model_and_tokenizer(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)

    # 交互式或单次生成
    if args.interactive:
        interactive_generation(model, tokenizer, device=args.device)
    else:
        print(f"提示语: {args.prompt}")
        print("生成中...")

        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device
        )

        print("\n" + "="*50)
        print("生成结果:")
        print("="*50)
        print(generated_text)


if __name__ == "__main__":
    main()
