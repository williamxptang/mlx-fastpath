"""Quantized-matmul cost vs rows M (verify width) for the 9B's linear shapes, 4- and 8-bit; plus 'split M into
chunks of <=s rows' alternative. L distinct weight sets per graph (no cache reuse). GPU time only."""
import time, json, statistics
import mlx.core as mx
SHAPES = [(4096, 12288), (12288, 4096), (4096, 4096 * 2)]  # mlp up/gate, mlp down, attn-ish proj
L = 12
def T(fn, ws, x):
    outs = [fn(x, w) for w in ws]
    t = time.perf_counter(); mx.eval(outs); return (time.perf_counter() - t) / len(ws) * 1e6
for bits in (4, 8):
    for K, Nout in SHAPES:
        ws = []
        for _ in range(L):
            w = mx.random.normal((Nout, K)).astype(mx.bfloat16)
            ws.append(mx.quantize(w, group_size=64, bits=bits))
        mx.eval(ws)
        qm = lambda x, w: mx.quantized_matmul(x, *w, transpose=True, group_size=64, bits=bits)
        def split(s):
            return lambda x, w: mx.concatenate([qm(x[i:i + s], w) for i in range(0, x.shape[0], s)], 0)
        curve = {}
        for M in range(1, 17):
            x = mx.random.normal((M, K)).astype(mx.bfloat16); mx.eval(x)
            curve[M] = statistics.median([T(qm, ws, x) for _ in range(7)])
        best_split = {}
        for M in range(2, 17):
            x = mx.random.normal((M, K)).astype(mx.bfloat16); mx.eval(x)
            opts = {s: statistics.median([T(split(s), ws, x) for _ in range(5)]) for s in (1, 2, 3, 4, 5, 8) if s < M}
            s = min(opts, key=opts.get)
            best_split[M] = (s, round(opts[s]), round(curve[M] / opts[s], 2))
        wbytes = sum(a.nbytes for a in ws[0])
        print(json.dumps({"bits": bits, "K": K, "N": Nout, "M1_GBs": round(wbytes / curve[1] / 1e3),
                          "curve_us": {m: round(v) for m, v in curve.items()},
                          "split_best(M: chunk, us, speedup)": best_split}), flush=True)
