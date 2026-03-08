# LLM-From-Scratch-0.1B

从零开始训练一个 0.1B 参数的中文大语言模型。

## 项目简介

本项目实现了一个完整的中文大语言模型训练流程，包括：

- **Tokenizer 训练**：基于 BPE 算法训练 15K 词表大小的中文分词器
- **预训练 (Pretrain)**：在大规模中文语料上进行自回归语言模型预训练
- **监督微调 (SFT)**：使用对话数据进行指令微调
- **评测 (Eval)**：交互式对话评测与 Benchmark 测试

---

## 模型权重

本仓库提供了训练好的模型权重，可通过 Git LFS 下载：

| 模型 | 参数量 | 文件路径 | 说明 |
|------|--------|----------|------|
| Pretrain | 0.1B | `checkpoints/pretrain_768.pth` | 预训练模型 |
| SFT | 0.1B | `checkpoints/sft_768.pth` | 监督微调模型 |

> 注：模型权重文件约 188MB，使用 Git LFS 存储。

---

## 模型架构

| 参数 | 默认值 |
|------|--------|
| hidden_size | 768 |
| num_hidden_layers | 12 |
| num_attention_heads | 12 |
| num_key_value_heads | 4 (GQA) |
| intermediate_size | 2048 |
| vocab_size | 15,000 |
| max_position_embeddings | 32,768 |

### 核心特性

- **Grouped Query Attention (GQA)**：减少 KV cache 显存占用
- **RoPE 位置编码**：支持长序列外推
- **SwiGLU 激活函数**：提升模型表达能力
- **RMSNorm**：更稳定的训练过程
- **Flash Attention**：加速训练推理
- **权重绑定**：Embedding 与 LM Head 共享权重

---

## 项目结构

```
LLM-From-Scratch-0.1B/
├── model/
│   ├── config.py                    # 模型配置
│   └── model_llm_from_scratch.py    # 模型实现
├── dataset/
│   ├── pretrain_dataset.py          # 预训练数据集
│   └── sft_dataset.py               # SFT 数据集
├── train/
│   ├── pretrain.py                  # 预训练脚本
│   ├── train_sft.py                 # SFT 训练脚本
│   ├── train_tokenizer.py           # Tokenizer 训练脚本
│   └── utils.py                     # 工具函数
├── benchmark/
│   ├── pretrain/                    # 预训练评测
│   └── mini_bench/                  # SFT 评测
│       ├── 100miniSponge.jsonl      # Mini 测试集 (100条)
│       └── eval.py                  # 评测脚本
├── tokenizer_15k/                   # 训练好的 Tokenizer
├── checkpoints/                     # 模型权重
│   ├── pretrain_768.pth             # 预训练权重
│   └── sft_768.pth                  # SFT 权重
└── eval.py                          # 交互式对话脚本
```

---

## 快速开始

### 环境安装

```bash
pip install torch transformers datasets swanlab
```

### 使用预训练模型

```python
from transformers import AutoTokenizer
from model.config import LLMFromScratchConfig
from model.model_llm_from_scratch import LLMFromScratchForCausalLM
import torch

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer_15k")

# 加载模型
config = LLMFromScratchConfig()
model = LLMFromScratchForCausalLM(config)
model.load_state_dict(torch.load("checkpoints/sft_768.pth", map_location="cpu"))
model.eval()

# 推理
input_text = "你好，请自我介绍一下。"
inputs = tokenizer(input_text, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### 交互式对话

```bash
python eval.py \
    --model_path checkpoints/sft_768.pth \
    --tokenizer_path tokenizer_15k \
    --model_type sft \
    --multi_turn
```

---

## 训练指南

### 预训练

```bash
cd train
python pretrain.py \
    --data_path /path/to/pretrain.bin \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --epochs 2
```

### SFT 微调

```bash
cd train
python train_sft.py \
    --data_path /path/to/sft.jsonl \
    --from_weight ../checkpoints/pretrain_768.pth \
    --tokenizer_path ../tokenizer_15k \
    --batch_size 128 \
    --learning_rate 2e-5 \
    --epochs 2
```

---

## 训练特性

- **分布式训练 (DDP)**：支持多卡多机训练
- **混合精度训练**：支持 bfloat16/float16
- **梯度累积**：支持大 batch size 等效训练
- **Warmup + Cosine Decay**：学习率调度
- **断点续训**：自动检测并恢复训练状态
- **SwanLab 集成**：训练过程可视化

---

## 评测 Benchmark

| Benchmark | 说明 |
|-----------|------|
| **C3** | 中文阅读理解 |
| **XCOPA-ZH** | 中文常识推理 |
| **Mini Bench** | 生成式对话评测 (100条测试用例) |

---

## 致谢

本项目参考了 LLaMA、Qwen 等开源模型的设计思路，感谢开源社区的贡献。

## License

MIT License
