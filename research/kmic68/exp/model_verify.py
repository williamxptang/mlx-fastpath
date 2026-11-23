"""(A) Real Qwen3.5-9B-OptiQ-4bit: cost of one verify forward pass of qL tokens at context N, before/after the
GQA-fold swap. Same cache state for both (KV trimmed, GDN state restored after every pass). Interleaved, median."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import sys, time, json, statistics
sys.path.insert(0, "<staging>/exp")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, KVCache
import patch_fold

MODEL = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
model, tok = load(MODEL)
mx.random.seed(0)
REPS = 9


def prefill(cache, ids, step=2048):
    for i in range(0, len(ids), step):
        model(mx.array(ids[i:i + step])[None], cache=cache)
        mx.eval([c.state for c in cache])


def verify(cache, toks):
    saved = [None if isinstance(c, KVCache) else list(c.state) for c in cache]  # state is the live list: copy it
    logits = model(toks, cache=cache)
    mx.eval(logits)
    for c, s in zip(cache, saved):
        if isinstance(c, KVCache):
            c.trim(toks.shape[1])
        else:
            c.state = s
    return logits


for N in (4096, 16384, 32768):
    ids = mx.random.randint(1000, 60000, (N,)).tolist()
    cache = make_prompt_cache(model)
    t = time.perf_counter(); prefill(cache, ids); pf = time.perf_counter() - t
    for qL in (1, 3, 4, 6, 8, 12, 16):
        toks = mx.random.randint(1000, 60000, (1, qL))
        patch_fold.enable(False); a = verify(cache, toks); a2 = verify(cache, toks)
        control = mx.abs(a.astype(mx.float32) - a2.astype(mx.float32)).max().item()
        assert control == 0.0, f"rollback not exact: {control}"
        patch_fold.enable(True); b = verify(cache, toks)
        diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()
        same = (a.argmax(-1) == b.argmax(-1)).all().item()
        ta, tb = [], []
        for _ in range(REPS):
            patch_fold.enable(False); t = time.perf_counter(); verify(cache, toks); ta.append(time.perf_counter() - t)
            patch_fold.enable(True); t = time.perf_counter(); verify(cache, toks); tb.append(time.perf_counter() - t)
        ma, mb = statistics.median(ta) * 1e3, statistics.median(tb) * 1e3
        print(json.dumps({"N": N, "qL": qL, "before_ms": round(ma, 2), "after_ms": round(mb, 2),
                          "speedup": round(ma / mb, 3), "argmax_same": same, "max_logit_diff": round(diff, 4), "control_diff": control,
                          "prefill_s": round(pf, 1)}), flush=True)
    patch_fold.enable(False)
    del cache
print("calls", patch_fold.stats, "PEAK_GB", round(mx.get_peak_memory() / 2**30, 2))
