import json
import os
import sys
import torch
from transformers import LlamaConfig, LlamaForCausalLM

if len(sys.argv) < 2:
    sys.stderr.write("usage: build_llama_model.py <output_dir>\n")
    sys.exit(1)

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

config = LlamaConfig(
    vocab_size=18,
    hidden_size=4,
    intermediate_size=16,
    num_hidden_layers=1,
    num_attention_heads=1,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
)
model = LlamaForCausalLM(config)
model.save_pretrained(out_dir)

# simple tokenizer compatible with HuggingFaceTokenizer spec
hf_tokenizer = {
    "model": {
        "type": "BPE",
        "vocab": {
            "h": 0,
            "e": 1,
            "l": 2,
            "o": 3,
            "w": 4,
            "r": 5,
            "d": 6,
            "</w>": 7,
            "he": 8,
            "hel": 9,
            "hell": 10,
            "hello": 11,
            "hello</w>": 12,
            "wo": 13,
            "wor": 14,
            "worl": 15,
            "world": 16,
            "world</w>": 17,
        },
        "merges": [
            "h e",
            "he l",
            "hel l",
            "hell o",
            "hello </w>",
            "w o",
            "wo r",
            "wor l",
            "worl d",
            "world </w>"
        ]
    }
}
with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
    json.dump(hf_tokenizer, f)
