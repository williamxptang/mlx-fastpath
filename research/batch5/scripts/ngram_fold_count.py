"""0.8B: does generate_ngram reach attention.folded_sdpa after apply()? Counted as the test counts it; and why not."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
from collections import Counter
import mlx_fastpath as fp
import mlx_fastpath.attention as att
import mlx_lm.models.qwen3_next as qn
from mlx_fastpath.guard import MemoryGuard
M = MODELS + "/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed"
calls = Counter(); why = Counter()
real_fold = att.folded_sdpa
def fold(*a, **k):
    calls["folded_sdpa"] += 1; return real_fold(*a, **k)
att.folded_sdpa = fold
real_dispatch = att.scaled_dot_product_attention
with MemoryGuard(min_free_gb=6, max_footprint_gb=4.5):
    eng = fp.load(M, memory_limit_gb=2.5, n2=False)
    tok = eng.tokenizer
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Explain in five sentences why the sky is blue."}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
    fp.remove(); fp.generate_ngram(eng, ids, max_tokens=96)          # stock first, as in the test
    print("apply:", fp.apply())
    print("qwen3_next sdpa is ours:", qn.scaled_dot_product_attention is real_dispatch)
    def spy(q, k, v, cache, scale, mask, sinks=None):
        B, Hq, qL, D = q.shape
        why[(qL, type(cache).__name__, isinstance(mask, str) and mask, B, sinks is None, getattr(k, "shape", None) and Hq // k.shape[1])] += 1
        return real_dispatch(q, k, v, cache, scale, mask, sinks)
    qn.scaled_dot_product_attention = spy
    fp.generate_ngram(eng, ids, max_tokens=96)
    fp.remove()
    print("folded_sdpa calls:", calls["folded_sdpa"])
    print("(qL, cache, mask, B, no-sinks, gqa) -> calls:", dict(why.most_common(10)))
