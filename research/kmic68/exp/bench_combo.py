"""Fold x 4/8-bit KV at the Qwen3.5-9B shape (16 q heads, 4 KV heads, D 256). Capped 2 GB. GPU time, median."""
import sys, time, json, statistics
sys.path.insert(0, '.')
import mlx.core as mx
mx.set_memory_limit(2 * 2**30); mx.set_cache_limit(256 * 2**20)
from mlx_lm.models.base import quantized_scaled_dot_product_attention as mlxlm_q
from qkv_attn import qkv_sdpa2
from folded_ops import folded_qsdpa
from smoke_ref import ref
Hq, Hkv, L = 16, 4, 8
CFG = {4: dict(block=512, nsg=4), 8: dict(block=256, nsg=4)}
def T(fn, args):
    outs = [fn(*a) for a in args]
    t = time.perf_counter(); mx.eval(outs); return (time.perf_counter() - t) / len(args) * 1e6
mx.random.seed(0)
for bits in (4, 8):
    for N in (4096, 16384):
        lay = []
        for _ in range(L):
            k = mx.random.normal((1, Hkv, N, 256)).astype(mx.bfloat16); v = mx.random.normal((1, Hkv, N, 256)).astype(mx.bfloat16)
            qk, qv = mx.quantize(k, 64, bits), mx.quantize(v, 64, bits); mx.eval(qk, qv); lay.append((qk, qv)); del k, v
        kd, vd = mx.dequantize(*lay[0][0], 64, bits), mx.dequantize(*lay[0][1], 64, bits)
        for qL in (1, 2, 3, 4, 6, 8, 12, 16):
            q = [mx.random.normal((1, Hq, qL, 256)).astype(mx.bfloat16) for _ in range(L)]; mx.eval(q)
            mask = "causal" if qL > 1 else None
            fs = {"mlxlm": lambda qq, la: mlxlm_q(mx.array(qq), la[0], la[1], scale=1/16, mask=mask, group_size=64, bits=bits),
                  "foldq": lambda qq, la: folded_qsdpa(qq, la[0], la[1], 1/16, bits)}
            R = 4 * qL
            for rg in sorted({4, 8, 16} & set(range(1, R + 1)) | ({R} if R <= 16 else set())):
                fs[f"kern_rg{rg}"] = (lambda rg: lambda qq, la: qkv_sdpa2(qq, la[0], la[1], 1/16, bits, rg=rg, **CFG[bits]))(rg)
            r = ref(q[0], kd, vd, 1/16)
            err = {n: round(mx.abs(f(q[0], lay[0]).astype(mx.float32) - r).max().item(), 4) for n, f in fs.items()}
            t = {n: [] for n in fs}
            args = list(zip(q, lay))
            for _ in range(5):
                for n, f in fs.items(): t[n].append(T(f, args))
            m = {n: round(statistics.median(x)) for n, x in t.items()}
            best = min(m, key=m.get)
            print(json.dumps({"bits": bits, "N": N, "qL": qL, "us": m, "best": best,
                              "best_vs_mlxlm": round(m["mlxlm"] / m[best], 2), "err": err}), flush=True)
        del lay, kd, vd
print("PEAK_GB", round(mx.get_peak_memory() / 2**30, 2))
