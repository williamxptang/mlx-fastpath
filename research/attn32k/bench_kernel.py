"""Kernel-level bench for item 2: verify attention at qL 1-4 on a bf16 cache, GPU time per layer (8 distinct layers
per graph so caches can't hide DRAM reads), median of 5 interleaved repeats. Error = max abs vs an fp32 reference.
argv: kernel module names (research/attn32k/<name>.py with sdpa(q, k, v, scale, n_valid=None, block=...)).
Guarded: MemoryGuard(14, 14). Appends to bench_kernel.jsonl."""
import importlib, json, statistics, sys, time
import mlx_fastpath as fp
from mlx_fastpath.folded import folded_sdpa

KERNELS = sys.argv[1:] or ["mma2"]
SHAPES = {"qwen35-9b": (16, 4), "qwen36-27b": (24, 4)}
L = 8
out = open("bench_kernel.jsonl", "a")
with fp.MemoryGuard(min_free_gb=14, max_footprint_gb=14) as g:
    import mlx.core as mx
    mods = {k: importlib.import_module(k) for k in KERNELS}

    def ref(q, k, v, n):
        q, k, v = (a.astype(mx.float32) for a in (q, k[:, :, :n], v[:, :, :n]))
        Hq, Hkv, qL = q.shape[1], k.shape[1], q.shape[2]
        k, v = (mx.repeat(a, Hq // Hkv, axis=1) for a in (k, v))
        s = (q / 16) @ k.swapaxes(-1, -2)
        mask = mx.arange(n)[None] <= (n - qL + mx.arange(qL))[:, None]
        s = mx.where(mask, s, -mx.inf)
        return mx.softmax(s, axis=-1) @ v

    def methods(n):
        m = {"stock": lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k[:, :, :n], v[:, :, :n], scale=1 / 16,
                                                                       mask="causal" if q.shape[2] > 1 else None),
             "fold": lambda q, k, v: folded_sdpa(q, k[:, :, :n], v[:, :, :n], 1 / 16)}
        for name, mod in mods.items():
            for b in (256, 512, 1024):
                m[f"{name}_b{b}"] = (lambda mod, b: lambda q, k, v: mod.sdpa(q, k, v, 1 / 16, n_valid=n, block=b))(mod, b)
        return m

    def T(fn, qs, ks, vs):
        outs = [fn(q, k, v) for q, k, v in zip(qs, ks, vs)]
        t = time.perf_counter(); mx.eval(outs); return (time.perf_counter() - t) / len(qs) * 1e6

    mx.random.seed(0)
    for shape, (Hq, Hkv) in SHAPES.items():
        for N in (8192, 32768):
            alloc = N + 256  # cache buffers are longer than the valid length
            ks = [mx.random.normal((1, Hkv, alloc, 256)).astype(mx.bfloat16) for _ in range(L)]
            vs = [mx.random.normal((1, Hkv, alloc, 256)).astype(mx.bfloat16) for _ in range(L)]
            mx.eval(ks, vs)
            M = methods(N)
            for qL in (1, 2, 3, 4):
                qs = [mx.random.normal((1, Hq, qL, 256)).astype(mx.bfloat16) for _ in range(L)]; mx.eval(qs)
                r = ref(qs[0], ks[0], vs[0], N)
                err = {n: round(mx.abs(f(qs[0], ks[0], vs[0]).astype(mx.float32) - r).max().item(), 4) for n, f in M.items()}
                t = {n: [] for n in M}
                for _ in range(5):
                    for n, f in M.items(): t[n].append(T(f, qs, ks, vs))
                med = {n: round(statistics.median(x)) for n, x in t.items()}
                row = {"shape": shape, "N": N, "qL": qL, "us": med, "err": err, **g.report()}
                print(json.dumps(row), flush=True); out.write(json.dumps(row) + "\n"); out.flush()
            del ks, vs
