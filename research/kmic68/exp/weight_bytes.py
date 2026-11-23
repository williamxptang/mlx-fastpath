import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json, collections
import mlx.core as mx
mx.set_memory_limit(8 * 2**30); mx.set_cache_limit(512 * 2**20)
from mlx.utils import tree_flatten
from mlx_lm import load
model, tok = load(MODELS + "/Qwen3.5-9B-OptiQ-4bit")
g = collections.Counter(); bits = {}
for k, v in tree_flatten(model.parameters()):
    if "vision" in k or "visual" in k: g["vision"] += v.nbytes; continue
    if "embed_tokens" in k: g["embed (1 row read)"] += v.nbytes; continue
    if "lm_head" in k: g["lm_head"] += v.nbytes; continue
    if "mtp" in k: g["mtp"] += v.nbytes; continue
    kind = "linear_attn" if "linear_attn" in k else "self_attn" if "self_attn" in k else "mlp" if "mlp" in k else "other"
    g[kind] += v.nbytes
for name, m in model.named_modules():
    if hasattr(m, "bits") and ("lm_head" in name or "embed" in name or name.endswith("layers.3.mlp.down_proj") or name.endswith("layers.1.mlp.down_proj")):
        bits[name] = (m.bits, m.group_size)
print(json.dumps({k: round(v / 2**30, 3) for k, v in g.most_common()}, indent=0))
read = sum(v for k, v in g.items() if k not in ("vision", "embed (1 row read)", "mtp"))
print("GB read per token:", round(read / 2**30, 3), "floor at 123 GB/s:", round(read / 123e9 * 1e3, 1), "ms ->", round(123e9 / read, 1), "tok/s")
print(bits, "tie:", getattr(model.args, "tie_word_embeddings", None) or model.args.text_config.get("tie_word_embeddings") if hasattr(model.args, "text_config") else None)
print("active GB", round(mx.get_active_memory() / 2**30, 2))
