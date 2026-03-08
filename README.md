# 🚀 LLM-From-Scratch-0.1B

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

从零开始手写代码，构建、训练并优化一个 **0.1B（1亿）参数规模的 LLaMA-style 中文大语言模型**。拒绝黑盒，全流程开源！

## 📖 项目简介

本项目旨在提供一个**极简、透明、可复现**的大模型工程实践路线图。我们不依赖庞杂的高阶黑盒 API，而是基于纯正的 PyTorch 从最底层的张量运算开始构建了一个完整的 Decoder-only Transformer 架构。项目覆盖了 LLM 生命周期的全链路：

1. **分词器 (Tokenizer)**：基于 BPE 算法自底向上训练了包含 15,000 词表的中文分词器。
2. **预训练 (Pretrain)**：在大规模中文语料上进行自回归语言建模，支持高效的多卡分布式训练。
3. **指令微调 (SFT)**：使用高质量对话数据进行有监督微调，赋予模型对话交互能力。
4. **评测体系 (Eval)**：集成 C3、XCOPA-ZH 等权威中文 Benchmark 及交互式生成评测。

---

<p align="center">
  <img width="900" alt="模型训练效果或架构图" src="https://github.com/user-attachments/assets/ccaa9df4-36a3-4d38-aee1-ee63f0493e51" />
</p>

---

## 🌟 核心技术与架构亮点

### 🧠 模型架构 (LLaMA-Style)
模型配置对标现代先进架构，包含诸多前沿的优化细节：
- **纯手写核心组件**：原生实现带有残差连接的 Pre-Norm Transformer Block。
- **位置编码与激活函数**：实现了支持最高 32,768 上下文外推的 **RoPE**（旋转位置编码）和 **SwiGLU** 门控线性单元。
- **内存优化加速**：支持 **Grouped Query Attention (GQA)** 大幅降低 KV Cache 显存占用。在 PyTorch >= 2.0 环境下自动启用 **Flash Attention** 加速计算。
- **稳定与轻量**：采用自定义的 `RMSNorm` (eps=1e-5) 稳定训练，并实现了 Embedding 与 LM Head 的权重共享（Weight Tying）以极致压缩参数量。

### 🚀 工业级训练管线
预训练脚本 (`pretrain.py`) 配备了应对大规模训练的完备功能：
- **分布式计算**：基于 `DistributedDataParallel` (DDP) 实现单机多卡/多机多卡训练。
- **优化策略**：使用 AdamW 优化器 (weight_decay=0.1) 配合 3% 步数的 Cosine Warmup 学习率调度。
- **混合精度与累积**：支持 `bfloat16`/`float16` 混合精度 (`torch.amp.autocast`) 和梯度累积，有效解决大 Batch Size 的显存瓶颈。
- **稳定续训**：通过定制 `SkipBatchSampler`，实现了断点自动检测和无损状态恢复。
- **计算图编译**：支持 `torch.compile` 将 PyTorch 代码编译为优化过的内核，显著提升吞吐。
- **可视化追踪**：原生集成 SwanLab，实时监控 Loss、学习率衰减并定期进行 Benchmark 测试打点。

---

## 🛠️ 模型参数配置

核心配置位于 `model/config.py`，默认参数对准 0.1B 黄金学习规模：

| 参数 (Hyperparameter) | 默认值 | 参数 (Hyperparameter) | 默认值 |
|:---|:---:|:---|:---:|
| `hidden_size` | 768 | `vocab_size` | 15,000 |
| `num_hidden_layers` | 12 | `intermediate_size` | 2048 |
| `num_attention_heads` | 12 | `max_position_embeddings` | 32,768 |
| `num_key_value_heads` | 4 (GQA) | `rope_theta` | 10000.0 |

> *详细定义请参考代码库中的 `LLMFromScratchConfig` 类。*

---

## 📦 模型权重下载

仓库内已包含训练好的模型权重（基于 Git LFS）。文件大小仅约 188MB，对个人消费级显卡极度友好。

| 模型版本 | 参数量 | 路径 | 说明 |
|------|--------|----------|------|
| **Pretrain Model** | 0.1B | `checkpoints/pretrain_768.pth` | 完成自回归语料学习的基础模型 |
| **SFT Model** | 0.1B | `checkpoints/sft_768.pth` | 经过对话与指令格式微调的模型 |

---

## 💻 快速开始

### 1. 环境安装
```bash
git clone [https://github.com/YuanxinLi0/LLM-From-Scratch-0.1B.git](https://github.com/YuanxinLi0/LLM-From-Scratch-0.1B.git)
cd LLM-From-Scratch-0.1B
pip install torch transformers datasets swanlab

```

### 2. 交互式对话测试

直接运行项目中提供的测试脚本，一键体验微调后的模型能力：

```bash
python eval.py \
    --model_path checkpoints/sft_768.pth \
    --tokenizer_path tokenizer_15k \
    --model_type sft \
    --multi_turn

```

*支持的交互命令：`quit` / `exit` (退出), `clear` (清空上下文), `single` (单轮问答), `multi` (多轮对话)。*

### 3. 代码中调用推理

```python
from transformers import AutoTokenizer
from model.config import LLMFromScratchConfig
from model.model_llm_from_scratch import LLMFromScratchForCausalLM
import torch

# 1. 加载定制的 BPE Tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer_15k")

# 2. 初始化模型并加载 SFT 权重
config = LLMFromScratchConfig()
model = LLMFromScratchForCausalLM(config)
model.load_state_dict(torch.load("checkpoints/sft_768.pth", map_location="cpu"))
model.eval()

# 3. 极速推理
inputs = tokenizer("你好，请详细自我介绍一下。", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))

```

---

## 🏃‍♂️ 从零开始复现训练

如果你想彻底打通训练流程，可以按照以下步骤操作：

### 阶段一：预训练 (Pretraining)

使用 `pretrain.py` 在大量无标签数据上赋予模型语言预测能力（支持多卡 `torchrun` 启动）：

```bash
cd train
python pretrain.py \
    --data_path /path/to/pretrain.bin \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --epochs 2 \
    --dtype bfloat16 \
    --use_swanlab 1

```

### 阶段二：指令微调 (SFT)

基于上一步的 Checkpoint 和指令对话数据进行对齐：

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

## 📈 评测Benchmark

项目原生集成了客观评测流水线，训练过程中会自动进行打点测试：

* **C3**：评测中文阅读理解能力。
* **XCOPA-ZH**：评测中文常识推理能力。
* **Mini Bench**：包含 100 条针对性测试用例的生成式对话评测集。

---

## 🙌 致谢与 License

本项目在架构设计和工程实现上参考了 [LLaMA](https://github.com/facebookresearch/llama)、[Qwen](https://github.com/QwenLM/Qwen) 等卓越的开源工作，深深感谢开源社区的力量。

本项目采用 **MIT License**，你可以自由地将其用于学习、修改或二次开发。

```
