"""B1: re-measure the M5's read-bandwidth ceiling with several kernels and take the best (the old 123 GB/s came from
one grid-stride kernel). Reads 1 GB (4 x 256 MB uint32 buffers); each timed mx.eval runs R passes over all buffers
(4 GB per eval), median of 9 after 2 warm-ups. Every Metal kernel XORs all it read into out[0] with one atomic per
thread, checked against numpy, so a kernel that skips bytes fails instead of looking fast.
Also: mx.sum / mx.max, and 4-bit gs64 quantized_matmul at M=1 on the 9B's MLP shapes (40 distinct matrices, ~1.1 GB,
so nothing stays in cache). Rows go to peakbw2.jsonl. Guarded: MemoryGuard(14, 14), MLX limit 10 GB."""
import json
import statistics
import time
from pathlib import Path

import numpy as np

import mlx_fastpath as fp

HERE = Path(__file__).parent
NBUF, BUF_U32, R, REPS = 4, 64 * 2**20, 4, 9
GIB_PER_TOKEN, AR_PIPELINED = 5.61, 19.66  # §9 floor (GiB of weights per token), §14.3 pipelined AR tok/s
out = open(HERE / "peakbw2.jsonl", "a")


def emit(row):
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n"); out.flush()


def timed(make_calls, nbytes):
    """make_calls() -> list of arrays for one eval; returns (GB/s median, min, max) over REPS evals."""
    for _ in range(2):
        mx.eval(make_calls())
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        mx.eval(make_calls())
        ts.append(time.perf_counter() - t0)
    gbs = sorted(nbytes / t / 1e9 for t in ts)
    return round(statistics.median(gbs), 1), round(gbs[0], 1), round(gbs[-1], 1)


HDR = "uint i = thread_position_in_grid.x; uint n = threads_per_grid.x; const device uint4* a4 = (const device uint4*)a; uint4 acc = 0;\n"
FTR = "\natomic_fetch_xor_explicit(&out[0], acc.x ^ acc.y ^ acc.z ^ acc.w, memory_order_relaxed);"
SRC = {
    # the old peakread.py kernel
    "stride": "for (uint j = i; j < N; j += n) acc ^= a4[j];",
    # grid-stride, U independent loads in flight per iteration (coalesced across the SIMD group)
    "stride_unroll": """uint j = i;
for (; j + (U - 1) * n < N; j += U * n) {
  #pragma unroll
  for (uint u = 0; u < U; ++u) acc ^= a4[j + u * n];
}
for (; j < N; j += n) acc ^= a4[j];""",
    # each thread reads its own contiguous chunk, U loads per iteration
    "chunk": """uint c = N / n; uint j = i * c, e = j + c;
for (; j + U <= e; j += U) {
  #pragma unroll
  for (uint u = 0; u < U; ++u) acc ^= a4[j + u];
}
for (; j < e; ++j) acc ^= a4[j];""",
}

with fp.MemoryGuard(min_free_gb=14, max_footprint_gb=14) as g:
    import mlx.core as mx
    mx.set_memory_limit(10 * 2**30)
    mx.set_cache_limit(1 * 2**30)
    K = {k: mx.fast.metal_kernel(name=f"rd_{k}", input_names=["a"], output_names=["out"], source=HDR + s + FTR,
                                 atomic_outputs=True) for k, s in SRC.items()}
    bufs = [mx.random.randint(0, 2**31 - 1, (BUF_U32,), dtype=mx.int32, key=mx.random.key(s)).view(mx.uint32)
            for s in range(NBUF)]
    mx.eval(bufs)
    n4 = BUF_U32 // 4
    want = [int(np.bitwise_xor.reduce(np.asarray(b))) for b in bufs]
    nbytes = R * sum(b.nbytes for b in bufs)
    base = {"device": mx.device_info()["device_name"], "mlx": mx.__version__, "read_gb_per_eval": nbytes / 1e9}

    def run(kind, threads, tg, U):
        c = lambda b: K[kind](inputs=[b], template=[("N", n4), ("U", U)], grid=(threads, 1, 1), threadgroup=(tg, 1, 1),
                              output_shapes=[(1,)], output_dtypes=[mx.uint32], init_value=0)[0]
        ok = all(int(c(b).item()) == w for b, w in zip(bufs, want))
        med, lo, hi = timed(lambda: [c(b) for _ in range(R) for b in bufs], nbytes)
        emit({**base, "kernel": kind, "threads": threads, "tg": tg, "unroll": U, "xor_ok": ok,
              "gb_s": med, "range": [lo, hi], **g.report()})

    for th in (16384, 65536, 262144):
        run("stride", th, 256, 1)
    for kind in ("stride_unroll", "chunk"):
        for U in (4, 8):
            for th in (8192, 16384, 32768, 65536, 131072, 262144):
                for tg in (64, 256, 1024):
                    run(kind, th, tg, U)

    for name, f in (("mx.sum", mx.sum), ("mx.max", mx.max)):
        med, lo, hi = timed(lambda: [f(b) for _ in range(R) for b in bufs], nbytes)
        emit({**base, "kernel": name, "gb_s": med, "range": [lo, hi], **g.report()})
    del bufs

    # 4-bit gs64 qmv at M=1, 9B MLP shapes (out, in): gate/up 12288x4096, down 4096x12288
    for shape in ((12288, 4096), (4096, 12288)):
        W = []
        for s in range(40):
            W.append(mx.quantize(mx.random.normal(shape, key=mx.random.key(100 + s)).astype(mx.bfloat16) * 0.02,
                                 group_size=64, bits=4))
            mx.eval(W[-1])
        x = mx.random.normal((1, shape[1]), key=mx.random.key(7)).astype(mx.bfloat16)
        mx.eval(x)
        wb = sum(a.nbytes for w in W for a in w)
        med, lo, hi = timed(lambda: [mx.quantized_matmul(x, *w, transpose=True, group_size=64, bits=4)
                                     for _ in range(2) for w in W], 2 * wb)
        emit({**base, "kernel": "qmv_4bit_gs64", "shape": list(shape), "matrices": len(W), "bytes_per_eval": 2 * wb,
              "gb_s": med, "range": [lo, hi], **g.report()})
        del W

out.close()
rows = [json.loads(l) for l in open(HERE / "peakbw2.jsonl")]
best = max((r for r in rows if r.get("xor_ok", True)), key=lambda r: r["gb_s"])
floor = best["gb_s"] * 1e9 / (GIB_PER_TOKEN * 2**30)
print(json.dumps({"summary": True, "best_gb_s": best["gb_s"], "best": {k: best.get(k) for k in ("kernel", "threads", "tg", "unroll", "shape")},
                  "floor_tok_s": round(floor, 2), "pipelined_ar_pct_of_floor": round(100 * AR_PIPELINED / floor, 1),
                  "reopen_fusion_and_E8": best["gb_s"] >= 135, "all_xor_ok": all(r.get("xor_ok", True) for r in rows)}))
