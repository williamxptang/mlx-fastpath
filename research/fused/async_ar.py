"""Item 3, step 2: pipelined plain decoding (graph for token t+1 built on the lazy argmax and queued with async_eval
while the GPU runs token t) vs optiq's synchronous AR loop. Greedy, 300 tokens, chat / needle_4k / needle_16k.
Audit: identical tokens, wall tok/s over the full token count, codes found. Guarded: MemoryGuard(14, 14)."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
import time
from pathlib import Path

import mlx_fastpath as fp

HERE = Path(__file__).parent
TARGET = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
D = json.load(open(HERE.parent / "dflash/prompts.json"))
P = {k: D["prompts"][k] for k in ("chat", "needle_4k", "needle_16k")}
MAX = 300
out = open(HERE / "async_ar.jsonl", "a")

with fp.MemoryGuard(min_free_gb=14, max_footprint_gb=14) as g:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    eng = fp.load(TARGET, memory_limit_gb=12)
    model, eos = eng._model, eng._eos_ids()

    def prefill(ids):
        c = make_prompt_cache(model)
        _, lo = eng._prefill_chunked(ids, c, step=1024)
        return c, int(mx.argmax(lo[0, -1]).item())

    def sync(ids):
        c, t = prefill(ids)
        toks = [t]
        t0 = time.perf_counter()
        while len(toks) < MAX and toks[-1] not in eos:
            lo = model(mx.array([[toks[-1]]]), cache=c)
            mx.eval(lo)
            toks.append(int(mx.argmax(lo[0, -1]).item()))
        return toks, time.perf_counter() - t0

    def pipelined(ids):
        c, t = prefill(ids)
        toks = [t]
        t0 = time.perf_counter()
        nxt = mx.argmax(model(mx.array([[t]]), cache=c)[0, -1], keepdims=True)
        mx.async_eval(nxt)
        while len(toks) < MAX and toks[-1] not in eos:
            cur = nxt
            if len(toks) + 1 < MAX:
                nxt = mx.argmax(model(cur[None], cache=c)[0, -1], keepdims=True)
                mx.async_eval(nxt)
            toks.append(int(cur.item()))  # may run one step past EOS; that step's token is not kept
        return toks, time.perf_counter() - t0

    for name, ids in P.items():
        r = {}
        for rep in range(2):
            for m, fn in (("sync", sync), ("pipelined", pipelined)):
                toks, el = fn(ids)
                r.setdefault(m, []).append((len(toks) - 1) / el)
                r[m + "_toks"] = toks
        text = eng.tokenizer.decode(r["pipelined_toks"])
        row = {"prompt": name, "prompt_tokens": len(ids), "tokens": len(r["pipelined_toks"]),
               "sync_tok_s": round(max(r["sync"]), 2), "pipelined_tok_s": round(max(r["pipelined"]), 2),
               "speedup": round(max(r["pipelined"]) / max(r["sync"]), 3),
               "same_tokens": r["sync_toks"] == r["pipelined_toks"],
               **({"codes": sum(c in text for c in D["codes"])} if name.startswith("needle") else {}), **g.report()}
        print(json.dumps(row), flush=True); out.write(json.dumps(row) + "\n"); out.flush()
