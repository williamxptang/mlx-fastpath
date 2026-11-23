"""Isolate each MTP phase (synced before/after) at short and 16k context on Qwen3.5-9B-OptiQ-4bit."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import sys, time, json, statistics
sys.path.insert(0, "<staging>/exp")
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, KVCache
from optiq.runtime.engine import OptiqEngine
from optiq.runtime.mtp.gdn_capture import forward_with_gdn_capture

eng = OptiqEngine(MODELS + "/Qwen3.5-9B-OptiQ-4bit")
model = eng._model
R = 9


def snap(cache):
    return [None if isinstance(c, KVCache) else list(c.state) for c in cache]


def restore(cache, s, n):
    for c, x in zip(cache, s):
        if isinstance(c, KVCache): c.trim(n)
        else: c.state = x


def t_sync(fn):
    mx.synchronize(); t = time.perf_counter(); r = fn(); mx.eval(r); mx.synchronize()
    return time.perf_counter() - t


for N in (200, 16000):
    ids = mx.random.randint(1000, 60000, (N,)).tolist()
    cache = make_prompt_cache(model)
    for i in range(0, N, 2048):
        out = model(mx.array(ids[i:i + 2048])[None], cache=cache); mx.eval([c.state for c in cache])
    _, hid, _ = forward_with_gdn_capture(model, mx.array([[ids[-1]]]), cache=cache, return_hidden=True)
    mx.eval(hid)
    hid = hid[:, -1:, :]
    res = {}
    # draft step (fresh mtp cache each time, as optiq does at cycle start)
    res["draft_ms"] = statistics.median([t_sync(lambda: eng._mtp_draft(hid, 1234, [None])[0]) for _ in range(R)]) * 1e3
    for qL in (2, 3, 4):
        toks = mx.random.randint(1000, 60000, (1, qL))
        s = snap(cache)
        def plain():
            r = model(toks, cache=cache); return r
        def capt():
            r = forward_with_gdn_capture(model, toks, cache=cache, return_hidden=True); return [r[0], r[1]] + [v for d in r[2].values() for v in d.values()]
        tp, tc = [], []
        for _ in range(R):
            tp.append(t_sync(plain)); restore(cache, s, qL)
            tc.append(t_sync(capt)); restore(cache, s, qL)
        res[f"plain_q{qL}_ms"] = statistics.median(tp) * 1e3
        res[f"capture_q{qL}_ms"] = statistics.median(tc) * 1e3
    res["ar_q1_ms"] = statistics.median([t_sync(lambda: model(mx.array([[5]]), cache=cache)) or 0 for _ in range(1)]) * 1e3
    print(json.dumps({"N": N, **{k: round(v, 2) for k, v in res.items()}}), flush=True)
    del cache
