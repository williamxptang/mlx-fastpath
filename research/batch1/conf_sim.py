"""B3: draft confidence log for setup A's drafter (32k-vocab MTP head, depth 2), Qwen3.5-9B-OptiQ-4bit, greedy.
Based on research/branch/sim.py. Per prompt:
1. Plain greedy decoding (AR, 300 tokens) with the main model's hidden state at every position.
2. At every position j, a chain cycle starting with last_token = x[j]: step 1 top-1 d1 with its probability p1,
   step 2 from (step-1 hidden, d1) top-1 d2 with p2 (softmax over the draft head's 32k ids, as setup A drafts).
   a1 = d1 == x[j+1]; a2 = d2 == x[j+2] (counts only after a1). Logged per position.
3. Verify cost at this context: forward_with_gdn_capture at widths 1-4, rolled back each time, median of 7
   interleaved repeats; and the draft step cost.
Prompts: chat, math, code_edit, needle_8k, needle_16k, then needle_32k last (rows already written survive a guard stop).
Writes conf.jsonl. Guarded: MemoryGuard(14, 14), MLX limit 10 GB."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
import statistics
import time
from pathlib import Path

import mlx_fastpath as fp

HERE = Path(__file__).parent
DF = HERE.parent / "dflash"
TARGET = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
GEN = 300
P = json.load(open(DF / "prompts.json"))["prompts"]
L = json.load(open(DF / "prompts_long.json"))["prompts"]
P = {k: P[k] for k in ("chat", "math", "code_edit")} | {"needle_8k": L["needle_8k"], "needle_16k": P["needle_16k"],
                                                         "needle_32k": L["needle_32k"]}
out = open(HERE / "conf.jsonl", "a")


def emit(row, show=None):
    print(json.dumps(show or row), flush=True)
    out.write(json.dumps(row) + "\n"); out.flush()


with fp.MemoryGuard(min_free_gb=14, max_footprint_gb=14) as g:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    eng = fp.load(TARGET, memory_limit_gb=10)
    fp.apply()
    from optiq.runtime.mtp.cache_state import rollback_after_verify, snapshot_untrimmable_cache
    from optiq.runtime.mtp.gdn_capture import forward_with_gdn_capture
    from mlx_fastpath.draft import DEFAULT_VOCAB, _head_of, sub_head, sub_logits
    lm = eng._lm
    sub = sub_head(_head_of(lm), DEFAULT_VOCAB)

    def mtp_step(h, toks):
        """h [N,1,D], toks [N] -> (top-1 ids [N], top-1 probs [N], hidden [N,1,D]); one draft step per row."""
        _, post = lm._mtp_core(h, mx.array(toks, mx.uint32)[:, None], mtp_cache=[None], emit_logits=False)
        pr = mx.softmax(sub_logits(sub, post)[:, -1].astype(mx.float32), axis=-1)
        top = mx.argmax(pr, axis=-1)
        p = mx.take_along_axis(pr, top[:, None], axis=-1)[:, 0]
        mx.eval(top, p, post)
        return top.tolist(), p.tolist(), post

    for name, ids in P.items():
        cache = make_prompt_cache(eng._model)
        hidden, logits = eng._prefill_chunked(ids, cache, step=1024)
        x, H = [int(mx.argmax(logits[0, -1]).item())], [hidden[:, -1:, :]]  # H[i] predicts x[i]
        for _ in range(GEN):
            hidden, logits = eng._forward(mx.array([[x[-1]]]), cache)
            mx.eval(logits, hidden)
            x.append(int(mx.argmax(logits[0, -1]).item())); H.append(hidden)
        J = list(range(len(x) - 2))
        d1, p1, h1 = mtp_step(mx.concatenate([H[j] for j in J], axis=0), [x[j] for j in J])
        d2, p2, _ = mtp_step(h1, d1)
        a1 = [d1[i] == x[j + 1] for i, j in enumerate(J)]
        a2 = [d2[i] == x[j + 2] for i, j in enumerate(J)]
        row = {"prompt": name, "context": len(ids) + GEN, "positions": len(J), "p1": [round(v, 4) for v in p1],
               "p2": [round(v, 4) for v in p2], "a1": a1, "a2": a2, "ar_text": eng.tokenizer.decode(x)}
        emit(row, {"prompt": name, "positions": len(J), "p_a1": round(sum(a1) / len(J), 3),
                   "p_a2_given_a1": round(sum(a and b for a, b in zip(a1, a2)) / max(1, sum(a1)), 3), **g.report()})

        times = {w: [] for w in range(1, 5)}
        for rep in range(7):
            for w in range(1, 5):
                snap = snapshot_untrimmable_cache(cache)
                mx.eval([c.state for c in cache])
                t0 = time.perf_counter()
                lo, hd, cap = forward_with_gdn_capture(eng._model, mx.array([x[-w:]]), cache=cache, return_hidden=True)
                mx.eval(lo, hd)
                times[w].append((time.perf_counter() - t0) * 1e3)
                rollback_after_verify(cache, snap, verified_tokens=w)
                del lo, hd, cap
        ts = []
        for _ in range(15):
            t0 = time.perf_counter(); mtp_step(H[-1], [x[-1]]); ts.append((time.perf_counter() - t0) * 1e3)
        emit({"prompt": name, "context": len(ids) + GEN,
              "verify_ms": {w: round(statistics.median(v[1:]), 2) for w, v in times.items()},
              "verify_range": {w: [round(min(v[1:]), 2), round(max(v[1:]), 2)] for w, v in times.items()},
              "draft_step_ms": round(statistics.median(ts[3:]), 2), **g.report()})
        del cache, H, h1
