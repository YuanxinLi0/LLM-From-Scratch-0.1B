"""
璁粌15k BPE tokenizer (涓嫳鏂囧弻璇?
"""
import os
import json
import time
from datetime import datetime
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

# 璁剧疆绾跨▼鏁?
NUM_THREADS = 300
os.environ['RAYON_NUM_THREADS'] = str(NUM_THREADS)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# 路径配置
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'pretrain_data', 'merged_pretrain_data_zh_en_only_v2.jsonl')
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, 'tokenizer_15k')
VOCAB_SIZE = 15000

# 鐗规畩tokens锛?5涓級
SPECIAL_TOKENS = [
    "<|endoftext|>",   # 0
    "<|im_start|>",    # 1
    "<|im_end|>",      # 2
    "<think>",         # 3
    "</think>",        # 4
    "<pad>",           # 5
    "<unk>",           # 6
    "<|system|>",      # 7
    "<|user|>",        # 8
    "<|assistant|>",   # 9
    "<tool_call>",     # 10
    "</tool_call>",    # 11
    "<function>",      # 12
    "</function>",     # 13
    "<unused_0>",      # 14
]

def get_texts(data_path, max_lines=None):
    """璇诲彇璁粌鏁版嵁"""
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            try:
                data = json.loads(line)
                text = data.get('text', '')
                if text:
                    yield text
            except:
                continue

def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens, max_lines=None):
    """璁粌tokenizer"""
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n寮€濮嬫椂闂? {start_datetime}")
    print(f"璁粌閰嶇疆:")
    print(f"  鏁版嵁: {data_path}")
    print(f"  璇嶈〃: {vocab_size} (BPE: {vocab_size - len(special_tokens)}, 鐗规畩: {len(special_tokens)})")
    print(f"  妯″紡: {'娴嬭瘯' if max_lines else '鍏ㄩ噺'}")
    print(f"  绾跨▼: {NUM_THREADS} (鎬绘牳蹇? {os.cpu_count()})\n")
    
    # 鍒濆鍖?
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 璁粌
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,  # 缁堢浼氭樉绀猴紝閲嶅畾鍚戞椂鏃犳晥
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        limit_alphabet=6000,
        continuing_subword_prefix="",
    )
    
    print("寮€濮嬭缁?..")
    print("(娉ㄦ剰: 璁粌杩囩▼杈冮暱锛屾棩蹇楄緭鍑哄彲鑳藉欢杩燂紝璇疯€愬績绛夊緟...)")
    texts = get_texts(data_path, max_lines=max_lines)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    print("璁粌闃舵瀹屾垚锛屽紑濮嬩繚瀛?..")
    
    # 楠岃瘉鐗规畩tokens
    for i, token in enumerate(special_tokens[:5]):
        assert tokenizer.token_to_id(token) == i, f"{token} ID閿欒"
    
    # 淇濆瓨
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    
    # 閰嶇疆鏂囦欢
    added_tokens_decoder = {
        str(i): {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        } for i, token in enumerate(special_tokens)
    }
    
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": special_tokens[5:],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "model_max_length": 8192,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": "{{- '<|im_start|>' -}}{%- for message in messages -%}{%- if message.role == 'user' -%}{{- '<|user|>' + message.content + '<|im_end|>' -}}{%- elif message.role == 'assistant' -%}{{- '<|assistant|>' + message.content + '<|im_end|>' -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|assistant|>' -}}{%- endif -%}"
    }
    
    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    # 璁＄畻璁粌鏃堕棿
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    
    print(f"\n璁粌瀹屾垚! 淇濆瓨鍒? {tokenizer_dir}")
    print(f"瀹為檯璇嶈〃澶у皬: {len(tokenizer.get_vocab())}")
    print(f"缁撴潫鏃堕棿: {end_datetime}")
    print(f"鎬昏€楁椂: {hours}灏忔椂 {minutes}鍒嗛挓 {seconds}绉?({elapsed_time:.1f}绉?\n")

def eval_tokenizer(tokenizer_dir):
    """娴嬭瘯tokenizer"""
    from transformers import AutoTokenizer
    
    print("娴嬭瘯tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    # 鍩虹娴嬭瘯
    test_text = "Hello World! 浣犲ソ涓栫晫锛乀his is a test. 杩欐槸娴嬭瘯銆?
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"  璇嶈〃澶у皬: {len(tokenizer)}")
    print(f"  娴嬭瘯鏂囨湰: {test_text}")
    print(f"  Token鏁伴噺: {len(tokens)}")
    print(f"  瑙ｇ爜涓€鑷? {'鉁? if decoded == test_text else '鉁?}")
    
    # 瀵硅瘽娴嬭瘯
    messages = [
        {"role": "system", "content": "You are helpful. 浣犲緢鏈夊府鍔┿€?},
        {"role": "user", "content": "Hi! 浣犲ソ锛?},
        {"role": "assistant", "content": "Hello! 浣犲ソ锛?}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"\n瀵硅瘽妯℃澘:\n{prompt}")

if __name__ == '__main__':
    import sys
    
    total_start = time.time()
    
    # --test 鍙傛暟浣跨敤娴嬭瘯妯″紡锛堝墠10000琛岋級
    test_mode = '--test' in sys.argv
    max_lines = 10000 if test_mode else None
    
    # 璁粌
    # train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE, SPECIAL_TOKENS, max_lines)
    
    # 娴嬭瘯
    eval_tokenizer(TOKENIZER_DIR)
    
    # 鎬绘椂闂寸粺璁?
    total_elapsed = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"鎬昏繍琛屾椂闂? {total_elapsed/60:.1f} 鍒嗛挓 ({total_elapsed:.1f} 绉?")
    print(f"{'='*50}\n")


