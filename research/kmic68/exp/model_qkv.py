"""Real Qwen3.5-9B-OptiQ-4bit with a 4-bit KV cache (group 64, as its kv_config.json): greedy decode speed
and tokens, stock mlx-lm quantized attention vs the fused kernel (qkv_sdpa2), plus bf16-KV reference.
Same starting state for every run: KV trimmed back, GDN state restored. Capped: memory limit 8 GB + a
watchdog that kills the process above 9 GB footprint."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import sys, os, time, json, statistics, threading, subprocess, re
sys.path.insert(0, "<staging>/exp")
import mlx.core as mx
mx.set_memory_limit(8 * 2**30); mx.set_cache_limit(1 * 2**30)


def watchdog(limit_gb=9.0):
    pid = os.getpid()
    while True:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True).stdout
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*(GB|MB|KB)", out)
        if m:
            gb = float(m.group(1)) * {"GB": 1, "MB": 1 / 1024, "KB": 1 / 2**20}[m.group(2)]
            watchdog.peak = max(watchdog.peak, gb)
            if gb > limit_gb:
                print(f"WATCHDOG: footprint {gb:.1f} GB > {limit_gb} GB, exiting", flush=True); os._exit(3)
        time.sleep(2)


watchdog.peak = 0.0
threading.Thread(target=watchdog, daemon=True).start()

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache, QuantizedKVCache
import mlx_lm.models.qwen3_next as qn
import mlx_lm.models.base as base
from qkv_attn import qkv_sdpa2

_orig = base.scaled_dot_product_attention
calls = {"fused": 0, "stock": 0}
CFG = {4: dict(block=512, nsg=4), 8: dict(block=256, nsg=4)}


def fused(queries, keys, values, cache, scale, mask, sinks=None):
    B, Hq, qL, D = queries.shape
    if (isinstance(cache, QuantizedKVCache) and B == 1 and D == 256 and sinks is None and qL <= 16
            and (mask is None or (isinstance(mask, str) and mask == "causal"))):
        calls["fused"] += 1
        return qkv_sdpa2(queries, cache.keys, cache.values, scale, cache.bits, n_valid=cache.offset, **CFG[cache.bits])
    calls["stock"] += 1
    return _orig(queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks)


def use(f): qn.scaled_dot_product_attention = fused if f else _orig


model, tok = load(MODELS + "/Qwen3.5-9B-OptiQ-4bit")
GEN = 48


def decode(cache, first, n=GEN):
    saved = [list(c.state) if not hasattr(c, "trim") else None for c in cache]
    toks, t = [], first
    mx.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        logits = model(mx.array([[t]]), cache=cache)
        t = logits[0, -1].argmax().item()
        toks.append(t)
    el = time.perf_counter() - t0
    for c, s in zip(cache, saved):
        if s is None: c.trim(n)
        else: c.state = s
    return toks, n / el


text = open("<staging>/research/261118-transferable-ideas/results.md").read() * 20
all_ids = tok.encode(text)
for N in (4096, 8192, 16384):
    ids = all_ids[:N]
    cache = make_prompt_cache(model)
    use(False)
    for i in range(0, N, 512):
        model(mx.array(ids[i:i + 512])[None], cache=cache); mx.eval([c.state for c in cache])
    first = model(mx.array([[ids[-1]]]), cache=cache)[0, -1].argmax().item()
    variants = {"bf16kv": cache}
    for bits in (4, 8):
        variants[f"q{bits}"] = [c.to_quantized(group_size=64, bits=bits) if isinstance(c, KVCache) else c for c in cache]
    mx.eval([c.state for v in variants.values() for c in v])
    runs = {"bf16kv_stock": ("bf16kv", False), "q4_stock": ("q4", False), "q4_fused": ("q4", True),
            "q8_stock": ("q8", False), "q8_fused": ("q8", True)}
    res = {k: [] for k in runs}; toks = {}
    for _ in range(3):
        for name, (v, f) in runs.items():
            use(f); t, s = decode(variants[v], first); res[name].append(s); toks[name] = t
    use(False)
    med = {k: statistics.median(v) for k, v in res.items()}
    print(json.dumps({"N": N, **{f"{k}_tok_s": round(v, 2) for k, v in med.items()},
                      "q4_fused_vs_stock": round(med["q4_fused"] / med["q4_stock"], 3),
                      "q8_fused_vs_stock": round(med["q8_fused"] / med["q8_stock"], 3),
                      "q4_fused_vs_bf16kv": round(med["q4_fused"] / med["bf16kv_stock"], 3),
                      "q4_tokens_same_as_stock": sum(a == b for a, b in zip(toks["q4_fused"], toks["q4_stock"])),
                      "q8_tokens_same_as_stock": sum(a == b for a, b in zip(toks["q8_fused"], toks["q8_stock"])),
                      "q4_stock_same_as_bf16": sum(a == b for a, b in zip(toks["q4_stock"], toks["bf16kv_stock"])),
                      "of": GEN, "peak_mlx_gb": round(mx.get_peak_memory() / 2**30, 2),
                      "peak_footprint_gb": round(watchdog.peak, 2)}), flush=True)
    del cache, variants
    mx.clear_cache()
print("calls", calls)
