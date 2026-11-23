"""E6: fused attention on 4/8-bit KV vs mlx-lm's quantized SDPA (unfused) vs bf16 mx.fast SDPA.
Memory capped at 2 GB. Correctness vs fp32 reference on the dequantized cache. GPU time only, median."""
import sys, time, json, statistics, itertools
sys.path.insert(0, '.')
import mlx.core as mx
mx.set_memory_limit(2 * 2**30); mx.set_cache_limit(256 * 2**20)
from mlx_lm.models.base import quantized_scaled_dot_product_attention as mlxlm_q
from qkv_attn import qkv_sdpa, qkv_sdpa2
from smoke_ref import ref

MODE = sys.argv[1] if len(sys.argv) > 1 else "bench"
Hq, Hkv, L = 24, 4, 6  # Qwen3.6-27B attention shape; 6 distinct layers per graph


def T(fn, args):
    outs = [fn(*a) for a in args]
    t = time.perf_counter(); mx.eval(outs); return (time.perf_counter() - t) / len(args) * 1e6


def make(N, bits):
    layers = []
    for _ in range(L):
        k = mx.random.normal((1, Hkv, N, 256)).astype(mx.bfloat16)
        v = mx.random.normal((1, Hkv, N, 256)).astype(mx.bfloat16)
        qk, qv = mx.quantize(k, 64, bits), mx.quantize(v, 64, bits)
        kd, vd = mx.dequantize(*qk, 64, bits), mx.dequantize(*qv, 64, bits)  # bf16 baseline = same values
        mx.eval(qk, qv, kd, vd)
        layers.append((qk, qv, kd, vd))
        del k, v
    return layers


mx.random.seed(0)
if MODE == "sweep":  # tune block / nsg / u at one point
    for bits in (4, 8):
        lay = make(16384, bits)
        q = [mx.random.normal((1, Hq, 1, 256)).astype(mx.bfloat16) for _ in range(L)]; mx.eval(q)
        res = []
        for b, n, u in itertools.product((128, 256, 512, 1024), (1, 2, 4, 8), (0,)):
            if b < n * 32: continue
            res.append((statistics.median([T(lambda qq, la: qkv_sdpa2(qq, la[0], la[1], 1/16, bits, block=b, nsg=n), list(zip(q, lay))) for _ in range(5)]), b, n, u))
        res.sort(); print(bits, "bit best", [(round(r[0]), *r[1:]) for r in res[:4]], flush=True)
        del lay
    sys.exit()

CFG = {4: dict(block=int(sys.argv[2]), nsg=int(sys.argv[3]), u=int(sys.argv[4])),
       8: dict(block=int(sys.argv[5]), nsg=int(sys.argv[6]), u=int(sys.argv[7]))}
for bits in (4, 8):
    for N in (4096, 16384, 32768):
        lay = make(N, bits)
        for qL in (1, 2, 4, 8):
            q = [mx.random.normal((1, Hq, qL, 256)).astype(mx.bfloat16) for _ in range(L)]; mx.eval(q)
            mask = "causal" if qL > 1 else None
            f_ours = lambda qq, la: qkv_sdpa2(qq, la[0], la[1], 1/16, bits, **CFG[bits])
            f_mlxlm = lambda qq, la: mlxlm_q(mx.array(qq), la[0], la[1], scale=1/16, mask=mask, group_size=64, bits=bits)
            f_bf16 = lambda qq, la: mx.fast.scaled_dot_product_attention(qq, la[2], la[3], scale=1/16, mask=mask)
            r = ref(q[0], lay[0][2], lay[0][3], 1/16)
            err = {n: mx.abs(f(q[0], lay[0]).astype(mx.float32) - r).max().item() for n, f in (("ours", f_ours), ("mlxlm", f_mlxlm), ("bf16", f_bf16))}
            t = {"ours": [], "mlxlm": [], "bf16": []}
            args = list(zip(q, lay))
            for _ in range(5):
                for n, f in (("ours", f_ours), ("mlxlm", f_mlxlm), ("bf16", f_bf16)): t[n].append(T(f, args))
            m = {n: statistics.median(x) for n, x in t.items()}
            qbytes = sum(a.nbytes for a in lay[0][0]) + sum(a.nbytes for a in lay[0][1])
            print(json.dumps({"bits": bits, "N": N, "qL": qL, **{f"{n}_us": round(x) for n, x in m.items()},
                              "vs_mlxlm": round(m["mlxlm"] / m["ours"], 2), "vs_bf16": round(m["bf16"] / m["ours"], 2),
                              "ours_GBs_packed": round(qbytes / m["ours"] / 1e3),
                              **{f"err_{n}": round(e, 4) for n, e in err.items()}}), flush=True)
        del lay
print("PEAK_GB", round(mx.get_peak_memory() / 2**30, 2))
