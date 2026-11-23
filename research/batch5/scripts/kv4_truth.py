"""0.8B, 4-bit KV cache, teacher-forced along one text: stock vs fastpath vs exact fp32 attention on the same quantized cache."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import mlx.core as mx
import mlx_lm.models.qwen3_next as qn
from mlx_lm.models.cache import make_prompt_cache, QuantizedKVCache
import mlx_fastpath as fp
from mlx_fastpath.guard import MemoryGuard
from optiq.runtime.engine import _maybe_quantize_kv
M = MODELS + "/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed"
stock_sdpa = qn.scaled_dot_product_attention

def exact_sdpa(q, k, v, cache, scale, mask, sinks=None):
    if not isinstance(cache, QuantizedKVCache):
        return stock_sdpa(q, k, v, cache, scale=scale, mask=mask, sinks=sinks)
    k = mx.dequantize(*k, group_size=cache.group_size, bits=cache.bits).astype(mx.float32)
    v = mx.dequantize(*v, group_size=cache.group_size, bits=cache.bits).astype(mx.float32)
    B, Hq, qL, D = q.shape; g = Hq // k.shape[1]; N = k.shape[2]
    k = mx.repeat(k, g, axis=1); v = mx.repeat(v, g, axis=1)
    s = (q.astype(mx.float32) * scale) @ k.swapaxes(-1, -2)
    allowed = mx.arange(N)[None] <= (N - qL + mx.arange(qL))[:, None]
    s = mx.where(allowed, s, -mx.inf)
    return (mx.softmax(s, axis=-1) @ v).astype(q.dtype)

with MemoryGuard(min_free_gb=6, max_footprint_gb=4.5):
    eng = fp.load(M, memory_limit_gb=2.5, n2=False)
    tok, model = eng.tokenizer, eng._model
    res = {}
    for prompt in ["Explain in five sentences why the sky is blue.",
                   "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring."]:
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False))
        fp.remove()
        text = tok.encode(fp.generate(eng, ids, max_tokens=96, kv_bits=4).text)
        def along(mode):
            fp.remove(); qn.scaled_dot_product_attention = stock_sdpa
            if mode == "fast": fp.apply()
            if mode == "exact": qn.scaled_dot_product_attention = exact_sdpa
            c = make_prompt_cache(model); rows = [model(mx.array([ids]), cache=c)[0, -1]]
            _maybe_quantize_kv(c, 4, 64)
            for t in text[:-1]:
                rows.append(model(mx.array([[t]]), cache=c)[0, -1]); mx.eval(rows[-1])
            fp.remove(); qn.scaled_dot_product_attention = stock_sdpa
            return mx.stack(rows).astype(mx.float32)
        L = {m: along(m) for m in ("stock", "fast", "exact")}
        def cmp(x, y):
            d = mx.abs(L[x] - L[y])
            top = (L[x].argmax(-1) == L[y].argmax(-1)).mean().item()
            # error on the tokens that matter: each row's top-20 by the exact logits
            idx = mx.argpartition(-L["exact"], 20, axis=-1)[:, :20]
            dtop = mx.abs(mx.take_along_axis(L[x], idx, -1) - mx.take_along_axis(L[y], idx, -1))
            return dict(max=round(d.max().item(), 3), mean=round(d.mean().item(), 4),
                        top20_max=round(dtop.max().item(), 3), argmax_agree=round(top, 3))
        print(prompt[:30], len(text), "tokens")
        for x, y in (("stock", "exact"), ("fast", "exact"), ("fast", "stock")):
            print(f"  {x:5s} vs {y:5s}", cmp(x, y))
