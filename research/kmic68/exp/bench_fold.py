"""E3 + E5: verify-width cost curve and GQA-folded attention vs mx.fast.scaled_dot_product_attention.
One timed graph = attention over L distinct per-layer caches (no unrealistic cache reuse); per-layer time reported.
Methods interleaved per round; median of ROUNDS."""
import sys, time, json, statistics
import mlx.core as mx
sys.path.insert(0, "<staging>/exp")
from gqa_fold import fold_sdpa

SHAPES = {  # name: (Hq, Hkv, D, full-attention layers)
    "qwen36-27b": (24, 4, 256, 16),
    "qwen35-9b": (16, 4, 256, 8),
    "gemma4-12b-global": (16, 8, 256, 8),
}
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 7
dt = mx.bfloat16


def ref(q, k, v, scale):
    _, Hq, qL, D = q.shape
    Hkv, N = k.shape[1], k.shape[2]
    qf = q.astype(mx.float32).reshape(Hkv, Hq // Hkv * qL, D)
    s = (qf * scale) @ k[0].astype(mx.float32).transpose(0, 2, 1)
    l = mx.tile(mx.arange(qL), Hq // Hkv)[:, None]
    s = mx.where(mx.arange(N)[None] <= N - qL + l, s, -mx.inf)
    return (mx.softmax(s, -1) @ v[0].astype(mx.float32)).reshape(1, Hq, qL, D)


def mlx_sdpa(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal" if q.shape[2] > 1 else None)


def timeit(fn, qs, ks, vs, scale):
    mx.eval([fn(q, k, v, scale) for q, k, v in zip(qs, ks, vs)])
    t = time.perf_counter()
    mx.eval([fn(q, k, v, scale) for q, k, v in zip(qs, ks, vs)])
    return (time.perf_counter() - t) / len(qs) * 1e6


mx.random.seed(0)
for name, (Hq, Hkv, D, L) in SHAPES.items():
    scale = D ** -0.5
    for N in (1024, 4096, 16384, 32768):
        ks = [mx.random.normal((1, Hkv, N, D)).astype(dt) for _ in range(L)]
        vs = [mx.random.normal((1, Hkv, N, D)).astype(dt) for _ in range(L)]
        mx.eval(ks, vs)
        for qL in (1, 2, 3, 4, 6, 8, 12, 16):
            qs = [mx.random.normal((1, Hq, qL, D)).astype(dt) for _ in range(L)]
            mx.eval(qs)
            r = ref(qs[0], ks[0], vs[0], scale)
            err_m = mx.abs(mlx_sdpa(qs[0], ks[0], vs[0], scale).astype(mx.float32) - r).max().item()
            blocks = [b for b in (128, 256, 512) if b <= N]
            err_f = max(mx.abs(fold_sdpa(qs[0], ks[0], vs[0], scale, block=b).astype(mx.float32) - r).max().item()
                        for b in blocks)
            t = {"mlx": []}
            t.update({f"fold{b}": [] for b in blocks})
            for _ in range(ROUNDS):
                t["mlx"].append(timeit(mlx_sdpa, qs, ks, vs, scale))
                for b in blocks:
                    t[f"fold{b}"].append(timeit(lambda q, k, v, s: fold_sdpa(q, k, v, s, block=b), qs, ks, vs, scale))
            med = {m: statistics.median(x) for m, x in t.items()}
            best = min((m for m in med if m != "mlx"), key=med.get)
            kv_bytes = 2 * Hkv * N * D * 2
            print(json.dumps({"shape": name, "N": N, "qL": qL, "mlx_us": round(med["mlx"], 1),
                              "fold_best": best, "fold_us": round(med[best], 1),
                              "speedup": round(med["mlx"] / med[best], 3),
                              "mlx_GBs": round(kv_bytes / med["mlx"] / 1e3, 1),
                              "fold_GBs": round(kv_bytes / med[best] / 1e3, 1),
                              "err_mlx": round(err_m, 4), "err_fold": round(err_f, 4)}), flush=True)
        del ks, vs
print("PEAK_GB", round(mx.get_peak_memory() / 2**30, 2))
