"""Gemma-4-12B global-layer attention (16 q heads, 1 KV head, D 512): mx.fast SDPA vs GQA fold (plain ops),
decode + verify widths, 8 distinct layers per graph. Capped 3 GB. GPU time only, interleaved, median."""
import sys, time, json, statistics
sys.path.insert(0, '.')
import mlx.core as mx
mx.set_memory_limit(3 * 2**30); mx.set_cache_limit(512 * 2**20)
from folded_ops import folded_sdpa
from smoke_ref import ref
Hq, Hkv, D, L = 16, 1, 512, 8
sc = 1.0  # Gemma 4 uses scale 1.0 with q/k norms
def T(fn, args):
    outs = [fn(*a) for a in args]
    t = time.perf_counter(); mx.eval(outs); return (time.perf_counter() - t) / len(args) * 1e6
sd = lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=sc, mask="causal" if q.shape[2] > 1 else None)
fo = lambda q, k, v: folded_sdpa(q, k, v, sc)
mx.random.seed(0)
for N in (4096, 16384, 32768):
    ks = [(mx.random.normal((1, Hkv, N, D)) * 0.05).astype(mx.bfloat16) for _ in range(L)]
    vs = [mx.random.normal((1, Hkv, N, D)).astype(mx.bfloat16) for _ in range(L)]
    mx.eval(ks, vs)
    for qL in (1, 2, 3, 4, 5, 7):
        qs = [(mx.random.normal((1, Hq, qL, D)) * 0.05).astype(mx.bfloat16) for _ in range(L)]; mx.eval(qs)
        r = ref(qs[0], ks[0], vs[0], sc)
        e = {n: round(mx.abs(f(qs[0], ks[0], vs[0]).astype(mx.float32) - r).max().item(), 4) for n, f in (("mlx", sd), ("fold", fo))}
        t = {"mlx": [], "fold": []}
        args = list(zip(qs, ks, vs))
        for _ in range(7):
            t["mlx"].append(T(sd, args)); t["fold"].append(T(fo, args))
        m = {n: statistics.median(x) for n, x in t.items()}
        print(json.dumps({"N": N, "qL": qL, "mlx_us": round(m["mlx"]), "fold_us": round(m["fold"]),
                          "speedup": round(m["mlx"] / m["fold"], 2), "err": e}), flush=True)
    del ks, vs
print("PEAK_GB", round(mx.get_peak_memory() / 2**30, 2))
