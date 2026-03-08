"""
交互式模型测试脚本 - 支持多轮对话
"""
import sys
import os

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer
from model.config import LLMFromScratchConfig
from model.model_llm_from_scratch import LLMFromScratchForCausalLM

print("=" * 60)
print("LLM-From-Scratch-0.1B 交互式测试")
print("=" * 60)

# 配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 Tokenizer
print(f"\n[加载 Tokenizer] ...")
tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer_15k")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 加载模型
print(f"[加载模型] 设备: {device}")
config = LLMFromScratchConfig()
model = LLMFromScratchForCausalLM(config)

model_path = os.path.join(os.path.dirname(__file__), "checkpoints", "sft_768.pth")
state_dict = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print(f"[模型参数] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print("=" * 60)
print("输入 'quit' 或 'exit' 退出")
print("输入 'clear' 清空对话历史")
print("输入 'single' 切换为单轮模式")
print("输入 'multi' 切换为多轮模式")
print("=" * 60)

# 对话模式
multi_turn = True  # 默认多轮
conversation = []

def generate_response(prompt, conversation_history=None):
    """生成回复"""
    if conversation_history is None:
        # 单轮模式
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    else:
        # 多轮模式 - 使用 chat template
        conversation_history.append({"role": "user", "content": prompt})
        formatted = tokenizer.apply_chat_template(
            conversation=conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response

# 主循环
while True:
    try:
        mode_str = "[多轮]" if multi_turn else "[单轮]"
        user_input = input(f"\n{mode_str} 你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见!")
        break
    
    if not user_input:
        continue
    
    # 命令处理
    if user_input.lower() in ['quit', 'exit', '退出']:
        print("再见!")
        break
    elif user_input.lower() == 'clear':
        conversation = []
        print("[已清空对话历史]")
        continue
    elif user_input.lower() == 'single':
        multi_turn = False
        conversation = []
        print("[切换为单轮模式]")
        continue
    elif user_input.lower() == 'multi':
        multi_turn = True
        conversation = []
        print("[切换为多轮模式]")
        continue
    
    # 生成回复
    if multi_turn:
        response = generate_response(user_input, conversation)
        conversation.append({"role": "assistant", "content": response})
    else:
        response = generate_response(user_input)
    
    print(f"模型: {response}")