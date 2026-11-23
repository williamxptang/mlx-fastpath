"""0.8B, generate_ngram with apply(): which query widths and mask types reach the attention dispatcher."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
from collections import Counter
import mlx_fastpath as fp
import mlx_fastpath.attention as att
from mlx_fastpath.guard import MemoryGuard
M = MODELS + "/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed"
seen = Counter()
real = att.scaled_dot_product_attention
def spy(q, k, v, cache, scale, mask, sinks=None):
    seen[(q.shape[2], type(mask).__name__ if not isinstance(mask, str) else mask, q.shape[1] // k.shape[1] if hasattr(k, "shape") else "q")] += 1
    return real(q, k, v, cache, scale, mask, sinks)
with MemoryGuard(min_free_gb=6, max_footprint_gb=4.5):
    eng = fp.load(M, memory_limit_gb=2.5, n2=False)
    tok = eng.tokenizer
    code = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n\n" * 3
    for name, p in [("sky", "Explain in five sentences why the sky is blue."),
                    ("copy", "Rename the function total to sum_all in this code and repeat the whole code:\n\n" + code)]:
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
        fp.apply(); att.scaled_dot_product_attention = spy
        import mlx_lm.models.qwen3_next as qn; qn.scaled_dot_product_attention = spy
        seen.clear()
        out = fp.generate_ngram(eng, ids, max_tokens=96)
        fp.remove(); att.scaled_dot_product_attention = real
        print(name, len(out), "tokens; (qL, mask, gqa) -> calls:", dict(sorted(seen.items(), key=lambda kv: -kv[1])[:8]))
