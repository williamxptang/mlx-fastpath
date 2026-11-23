"""At the kv4 split on the 0.8B: logit gap between the two tokens under a 4-bit KV cache, stock vs fastpath."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import sys
import mlx.core as mx
import mlx_fastpath as fp
from mlx_fastpath.guard import MemoryGuard
M = MODELS + "/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed"
with MemoryGuard(min_free_gb=6, max_footprint_gb=4.5):
    eng = fp.load(M, memory_limit_gb=2.5, n2=False)
    tok = eng.tokenizer
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Explain in five sentences why the sky is blue."}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False))
    fp.remove()
    s = fp.generate(eng, ids, max_tokens=40, kv_bits=4).text
    fp.apply()
    f = fp.generate(eng, ids, max_tokens=40, kv_bits=4).text
    fp.remove()
    a, b = tok.encode(s), tok.encode(f)
    i = next(k for k, (x, y) in enumerate(zip(a, b)) if x != y)
    print("split at", i, repr(tok.decode([a[i]])), "vs", repr(tok.decode([b[i]])))
    from mlx_lm.models.cache import make_prompt_cache
    from optiq.runtime.engine import _maybe_quantize_kv
    def gap(fast, mode):
        (fp.apply if fast else fp.remove)()
        c = make_prompt_cache(eng._model)
        eng._model(mx.array([ids]), cache=c)
        _maybe_quantize_kv(c, 4, 64)
        seq = a[:i]
        if mode == "one":                 # one token at a time (qL=1, fused kernel path when on)
            for t in seq[:-1]:
                eng._model(mx.array([[t]]), cache=c)
            lg = eng._model(mx.array([[seq[-1]]]), cache=c)[0, -1]
        else:                             # verify-style block of 3 (qL=3)
            for t in seq[:-3]:
                eng._model(mx.array([[t]]), cache=c)
            lg = eng._model(mx.array([seq[-3:]]), cache=c)[0, -1]
        lg = lg.astype(mx.float32)
        fp.remove()
        return round((lg[a[i]] - lg[b[i]]).item(), 3)
    for mode in ("one", "block3"):
        print(mode, "stock gap", gap(False, mode), "fastpath gap", gap(True, mode))
