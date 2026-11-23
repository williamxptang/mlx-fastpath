"""Full verify-width map for mx.fast.sdpa + 'pad queries to a fast width' trick (dummy rows prepended so
causal alignment of the real rows is unchanged; output = last qL rows)."""
import sys, time, json, statistics
sys.path.insert(0, '.')
import mlx.core as mx
from smoke_ref import ref
SHAPES = {"qwen36-27b": (24, 4, 16), "qwen35-9b": (16, 4, 8), "gemma4-12b-global": (16, 8, 8)}
def sd(q, k, v): return mx.fast.scaled_dot_product_attention(q, k, v, scale=1/16, mask="causal" if q.shape[2] > 1 else None)
def padded(w):
    def f(q, k, v):
        qL = q.shape[2]
        if w <= qL: return sd(q, k, v)
        qp = mx.concatenate([mx.zeros((1, q.shape[1], w - qL, 256), q.dtype), q], axis=2)
        return sd(qp, k, v)[:, :, -qL:]
    return f
def T(fn, qs, ks, vs):
    outs = [fn(q, k, v) for q, k, v in zip(qs, ks, vs)]
    t = time.perf_counter(); mx.eval(outs); return (time.perf_counter() - t) / len(qs) * 1e6
mx.random.seed(0)
for name, (Hq, Hkv, L) in SHAPES.items():
    for N in (4096, 32768):
        ks = [mx.random.normal((1, Hkv, N, 256)).astype(mx.bfloat16) for _ in range(L)]
        vs = [mx.random.normal((1, Hkv, N, 256)).astype(mx.bfloat16) for _ in range(L)]
        mx.eval(ks, vs)
        curve = {}
        for qL in list(range(1, 17)) + [20, 24, 32]:
            qs = [mx.random.normal((1, Hq, qL, 256)).astype(mx.bfloat16) for _ in range(L)]; mx.eval(qs)
            curve[qL] = statistics.median([T(sd, qs, ks, vs) for _ in range(5)])
        # pad trick: for each qL, best width >= qL from the measured curve, then measure it for real
        for qL in range(2, 17):
            w = min((x for x in curve if x >= qL), key=curve.get)
            if w == qL: continue
            qs = [mx.random.normal((1, Hq, qL, 256)).astype(mx.bfloat16) for _ in range(L)]; mx.eval(qs)
            r = ref(qs[0], ks[0], vs[0], 1/16)
            err = mx.abs(padded(w)(qs[0], ks[0], vs[0]).astype(mx.float32) - r).max().item()
            base = statistics.median([T(sd, qs, ks, vs) for _ in range(5)])
            pad = statistics.median([T(padded(w), qs, ks, vs) for _ in range(5)])
            print(json.dumps({"shape": name, "N": N, "qL": qL, "pad_to": w, "plain_us": round(base), "padded_us": round(pad),
                              "speedup": round(base / pad, 2), "err_padded": round(err, 4)}), flush=True)
        print(json.dumps({"shape": name, "N": N, "curve_us": {k: round(v) for k, v in curve.items()}}), flush=True)
        del ks, vs
