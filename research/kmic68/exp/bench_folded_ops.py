import sys, time, json, statistics
sys.path.insert(0, '.')
import mlx.core as mx
from smoke_ref import ref
from folded_ops import folded_sdpa
SHAPES = {"qwen36-27b": (24, 4, 16), "qwen35-9b": (16, 4, 8), "gemma4-12b-global": (16, 8, 8)}
sd = lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=1/16, mask="causal" if q.shape[2] > 1 else None)
def pad16(q, k, v):
    qL = q.shape[2]
    if qL >= 16: return sd(q, k, v)
    return sd(mx.concatenate([mx.zeros((1, q.shape[1], 16 - qL, 256), q.dtype), q], 2), k, v)[:, :, -qL:]
fold = lambda q, k, v: folded_sdpa(q, k, v, 1/16)
def T(fn, qs, ks, vs):
    outs = [fn(q, k, v) for q, k, v in zip(qs, ks, vs)]
    t = time.perf_counter(); mx.eval(outs); return (time.perf_counter() - t) / len(qs) * 1e6
mx.random.seed(0)
for name, (Hq, Hkv, L) in SHAPES.items():
    for N in (4096, 32768):
        ks = [mx.random.normal((1, Hkv, N, 256)).astype(mx.bfloat16) for _ in range(L)]
        vs = [mx.random.normal((1, Hkv, N, 256)).astype(mx.bfloat16) for _ in range(L)]
        mx.eval(ks, vs)
        for qL in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16):
            qs = [mx.random.normal((1, Hq, qL, 256)).astype(mx.bfloat16) for _ in range(L)]; mx.eval(qs)
            r = ref(qs[0], ks[0], vs[0], 1/16)
            err = {n: mx.abs(f(qs[0], ks[0], vs[0]).astype(mx.float32) - r).max().item() for n, f in (("plain", sd), ("pad16", pad16), ("fold", fold))}
            t = {"plain": [], "pad16": [], "fold": []}
            for _ in range(5):
                for n, f in (("plain", sd), ("pad16", pad16), ("fold", fold)): t[n].append(T(f, qs, ks, vs))
            m = {n: statistics.median(x) for n, x in t.items()}
            print(json.dumps({"shape": name, "N": N, "qL": qL, **{f"{n}_us": round(x) for n, x in m.items()},
                              "fold_vs_plain": round(m["plain"] / m["fold"], 2), "best": min(m, key=m.get),
                              **{f"err_{n}": round(e, 4) for n, e in err.items()}}), flush=True)
        del ks, vs
