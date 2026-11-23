"""E3+E5: verify-width curve. MLX sdpa vs GQA-folded MMA kernel. GPU time only (graph built before the timer),
L distinct per-layer caches per graph, methods interleaved, median of ROUNDS."""
import sys, time, json, statistics
sys.path.insert(0, '.')
import mlx.core as mx
from gqa_mma import mma_sdpa
from smoke_ref import ref
SHAPES = {"qwen36-27b": (24, 4, 16), "qwen35-9b": (16, 4, 8), "gemma4-12b-global": (16, 8, 8)}
ROUNDS = 7
def T(fn, qs, ks, vs):
    outs = [fn(q, k, v) for q, k, v in zip(qs, ks, vs)]
    t = time.perf_counter(); mx.eval(outs); return (time.perf_counter() - t) / len(qs) * 1e6
mlx = lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=1/16, mask="causal" if q.shape[2] > 1 else None)
mx.random.seed(0)
for name, (Hq, Hkv, L) in SHAPES.items():
    for N in (1024, 4096, 16384, 32768):
        ks = [mx.random.normal((1, Hkv, N, 256)).astype(mx.bfloat16) for _ in range(L)]
        vs = [mx.random.normal((1, Hkv, N, 256)).astype(mx.bfloat16) for _ in range(L)]
        mx.eval(ks, vs)
        for qL in (1, 2, 3, 4, 6, 8, 12, 16):
            qs = [mx.random.normal((1, Hq, qL, 256)).astype(mx.bfloat16) for _ in range(L)]; mx.eval(qs)
            r = ref(qs[0], ks[0], vs[0], 1/16)
            blocks = [b for b in (64, 128, 256, 512) if b <= N]
            errs = {b: mx.abs(mma_sdpa(qs[0], ks[0], vs[0], 1/16, block=b).astype(mx.float32) - r).max().item() for b in blocks}
            em = mx.abs(mlx(qs[0], ks[0], vs[0]).astype(mx.float32) - r).max().item()
            t = {"mlx": []} | {b: [] for b in blocks}
            for _ in range(ROUNDS):
                t["mlx"].append(T(mlx, qs, ks, vs))
                for b in blocks: t[b].append(T(lambda q, k, v: mma_sdpa(q, k, v, 1/16, block=b), qs, ks, vs))
            med = {m: statistics.median(x) for m, x in t.items()}
            best = min(blocks, key=med.get)
            kvb = 2 * Hkv * N * 256 * 2
            print(json.dumps({"shape": name, "N": N, "qL": qL, "mlx_us": round(med["mlx"], 1), "mma_us": round(med[best], 1),
                  "block": best, "speedup": round(med["mlx"] / med[best], 2), "mlx_GBs": round(kvb / med["mlx"] / 1e3),
                  "mma_GBs": round(kvb / med[best] / 1e3), "err_mlx": round(em, 4), "err_mma": round(errs[best], 4)}), flush=True)
        del ks, vs
