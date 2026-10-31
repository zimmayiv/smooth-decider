"""Build the tiny random decider-shaped checkpoint the Point tests use (.cache/tiny_decider).

It has decider's real tokenizer and config (downloaded from the Hub) and a 24-layer Qwen3.5 text
model with hidden size 32, so the tests run on CPU in seconds.

    python scripts/make_tiny_decider.py
"""
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

out = Path(__file__).resolve().parents[1] / ".cache" / "tiny_decider"
out.mkdir(parents=True, exist_ok=True)
for f in ["tokenizer.json", "tokenizer_config.json", "decider_config.json", "chat_template.jinja"]:
    shutil.copy(hf_hub_download("Mapika/decider-2b", f), out / f)
tok = AutoTokenizer.from_pretrained(out)
torch.manual_seed(0)
cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=24,
                        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                        linear_key_head_dim=8, linear_value_head_dim=8, max_position_embeddings=40960,
                        tie_word_embeddings=True, pad_token_id=tok.pad_token_id)
Qwen3_5ForCausalLM(cfg).save_pretrained(out)
print("wrote", out)
