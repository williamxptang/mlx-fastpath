"""Which matmuls make verifying 3 tokens cost so much more than 2 (Gemma 4 E2B int4 on M1: 32 -> 52 ms)?

1. Every distinct quantized weight in the model (QuantizedLinear / QuantizedEmbedding as head): shape, bits, group,
   count. Each is timed alone on a (1, n, in) bf16 input for n = 1..6 (median of REPS, after warm-up).
2. The whole target step for n = 1..6 tokens on a cache prefilled with CTX tokens (fresh cache per repeat; prefill not
   timed).
3. Predicted step(n) = sum over shapes of count x time(shape, n); compared with the measured step, so the cliff can be
   pinned on shapes (or on something else).

    python cliff_bench.py OUT_JSONL MODEL_DIR
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlx_fastpath.guard import MemoryGuard  # noqa: E402

out_path, mdir = sys.argv[1:3]
WIDTHS, REPS, CTX = range(1, 7), int(os.environ.get("REPS", 15)), int(os.environ.get("CTX", 512))
mx.set_memory_limit(6 * 2**30)
mx.set_cache_limit(1 * 2**30)


def med_ms(fn, reps=REPS):
    fn(); fn()
    ts = []
    for _ in range(reps):
        mx.synchronize(); t = time.perf_counter(); fn(); mx.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


with MemoryGuard(min_free_gb=7.5, max_footprint_gb=7.0) as guard:
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tok = load(mdir)
    shapes = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            out_f = m.weight.shape[0]
            in_f = m.weight.shape[1] * 32 // m.bits
            key = (type(m).__name__, out_f, in_f, m.bits, m.group_size, getattr(m, "mode", "affine"))
            shapes.setdefault(key, {"count": 0, "example": name, "module": m})
            shapes[key]["count"] += 1
    rows = []
    for key, d in sorted(shapes.items(), key=lambda kv: -kv[1]["count"] * kv[0][1] * kv[0][2]):
        kind, out_f, in_f, bits, gs, mode = key
        m = d["module"]
        t = {}
        for n in WIDTHS:
            x = mx.random.normal((1, n, in_f)).astype(mx.bfloat16)
            f = (lambda m=m, x=x: mx.eval(m(x))) if kind == "QuantizedLinear" else (lambda m=m, x=x: mx.eval(m.as_linear(x)))
            t[n] = round(med_ms(f), 4)
        rows.append({"kind": kind, "out": out_f, "in": in_f, "bits": bits, "group": gs, "mode": mode,
                     "count": d["count"], "example": d["example"], "ms": t,
                     "jump_2_to_3": round(t[3] / t[2], 3)})
    ids = tok.encode("The quick brown fox jumps over the lazy dog. " * 200)[:CTX]

    def step(n):
        c = make_prompt_cache(model)
        model(mx.array([ids])[:, :CTX], cache=c)
        mx.eval([x.state for x in c])

        def run():
            mx.eval(model(mx.array([[ids[0]] * n]), cache=c))
            for x in c:
                if hasattr(x, "trim") and x.is_trimmable():
                    x.trim(n)
        return run

    measured = {}
    for n in WIDTHS:
        ts = []
        for _ in range(5):
            r = step(n)
            ts.append(med_ms(r, reps=1))
        measured[n] = round(statistics.median(ts), 3)
    predicted = {n: round(sum(r["count"] * r["ms"][n] for r in rows), 3) for n in WIDTHS}
    rec = {"model": os.path.basename(mdir.rstrip("/")), "ctx": CTX, "step_ms": measured, "matmul_sum_ms": predicted,
           "shapes": rows, "guard": guard.report(), "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2)}
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "shapes"}), flush=True)
    for r in rows[:12]:
        print(r["count"], r["kind"], r["out"], r["in"], r["bits"], r["group"], r["mode"], r["ms"], "x3/2", r["jump_2_to_3"])
