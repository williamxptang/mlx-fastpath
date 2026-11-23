"""Item 1 go/no-go, before building tree verify: how many tokens per cycle would branch drafting add, and what does
a wider verify pass cost? Qwen3.5-9B-OptiQ-4bit, greedy, 32k-vocab MTP draft head (setup A's drafter).

1. Plain greedy decoding (AR) with the main model's hidden state at every position.
2. At every position j, the MTP head drafts as in a cycle that starts with last_token = x[j+1]:
   step 1 from (h_j, x[j+1]) -> top-2 (d1a, d1b); step 2 from each step-1 hidden -> top-2.
   Acceptance is judged against the AR tokens (= the target's greedy choice).
3. Verify cost: forward_with_gdn_capture at widths 1-6, rolled back after each pass, median of interleaved repeats.
Writes sim.jsonl. Guarded: MemoryGuard(14, 14), MLX limit 12 GB."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
import statistics
import sys
import time
from pathlib import Path

import mlx_fastpath as fp

HERE = Path(__file__).parent
DF = HERE.parent / "dflash"
TARGET = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
GEN = 300
P = json.load(open(DF / "prompts.json"))["prompts"]
P = {k: P[k] for k in ("chat", "math", "code_edit")} | {"needle_8k": json.load(open(DF / "prompts_long.json"))["prompts"]["needle_8k"]}
out = open(HERE / "sim.jsonl", "a")


def emit(row):
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n"); out.flush()


with fp.MemoryGuard(min_free_gb=14, max_footprint_gb=14) as g:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    eng = fp.load(TARGET, memory_limit_gb=12)
    fp.apply()
    from optiq.runtime.mtp.cache_state import rollback_after_verify, snapshot_untrimmable_cache
    from optiq.runtime.mtp.gdn_capture import forward_with_gdn_capture
    from mlx_fastpath.draft import DEFAULT_VOCAB, _head_of, sub_head, sub_logits
    lm = eng._lm
    sub = sub_head(_head_of(lm), DEFAULT_VOCAB)

    def mtp_step(h, toks):
        """h [N,1,D], toks [N] -> (top-2 ids [N,2], hidden [N,1,D]); one draft step per row, no MTP cache."""
        _, post = lm._mtp_core(h, mx.array(toks, mx.uint32)[:, None], mtp_cache=[None], emit_logits=False)
        lo = sub_logits(sub, post)[:, -1]
        top = mx.argsort(-lo, axis=-1)[:, :2]
        mx.eval(top, post)
        return top.tolist(), post

    for name, ids in P.items():
        cache = make_prompt_cache(eng._model)
        hidden, logits = eng._prefill_chunked(ids, cache, step=1024)
        x, H = [int(mx.argmax(logits[0, -1]).item())], [hidden[:, -1:, :]]  # H[i] predicts x[i]
        for _ in range(GEN):
            hidden, logits = eng._forward(mx.array([[x[-1]]]), cache)
            mx.eval(logits, hidden)
            x.append(int(mx.argmax(logits[0, -1]).item())); H.append(hidden)
        # cycle at j: last_hidden = H[j] (it predicted x[j]), last_token = x[j]; truth x[j+1], x[j+2]
        J = list(range(len(x) - 2))
        hs = mx.concatenate([H[j] for j in J], axis=0)
        top1, h1 = mtp_step(hs, [x[j] for j in J])
        a2, ha = mtp_step(h1, [t[0] for t in top1])
        b2, _ = mtp_step(h1, [t[1] for t in top1])
        rows = [{"a1": top1[i][0] == x[j + 1], "b1": top1[i][1] == x[j + 1],
                 "a2": a2[i][0] == x[j + 2], "a2b": a2[i][1] == x[j + 2], "b2": b2[i][0] == x[j + 2]}
                for i, j in enumerate(J)]
        n = len(rows)
        f = lambda c: sum(1 for r in rows if c(r)) / n
        # expected tokens committed per cycle (accepted drafts + the target's own token)
        E = {"chain1": 1 + f(lambda r: r["a1"]),
             "chain2": 1 + f(lambda r: r["a1"]) + f(lambda r: r["a1"] and r["a2"]),
             "T4_top2_d1": 1 + f(lambda r: r["a1"] or r["b1"]) + f(lambda r: r["a1"] and r["a2"]),
             "T4_top2_d2": 1 + f(lambda r: r["a1"]) + f(lambda r: r["a1"] and (r["a2"] or r["a2b"])),
             "T5_two_chains": 1 + f(lambda r: r["a1"] or r["b1"]) + f(lambda r: (r["a1"] and r["a2"]) or (not r["a1"] and r["b1"] and r["b2"])),
             "T5_top2_both": 1 + f(lambda r: r["a1"] or r["b1"]) + f(lambda r: r["a1"] and (r["a2"] or r["a2b"]))}
        emit({"prompt": name, "positions": n, "p_a1": round(f(lambda r: r["a1"]), 3),
              "p_b1_given_miss": round(f(lambda r: r["b1"]) / max(1e-9, 1 - f(lambda r: r["a1"])), 3),
              "p_a2_given_a1": round(f(lambda r: r["a1"] and r["a2"]) / max(1e-9, f(lambda r: r["a1"])), 3),
              "E": {k: round(v, 3) for k, v in E.items()}, **g.report()})

        if name in ("chat", "needle_8k"):
            # verify cost by width at this context, after the AR run (cache holds prompt + 300 tokens)
            times = {w: [] for w in range(1, 7)}
            for rep in range(7):
                for w in range(1, 7):
                    snap = snapshot_untrimmable_cache(cache)
                    mx.eval([c.state for c in cache])
                    t0 = time.perf_counter()
                    lo, hd, cap = forward_with_gdn_capture(eng._model, mx.array([x[-w:]]), cache=cache, return_hidden=True)
                    mx.eval(lo, hd)
                    times[w].append((time.perf_counter() - t0) * 1e3)
                    rollback_after_verify(cache, snap, verified_tokens=w)
                    del lo, hd, cap
            # draft step cost: 1 row (chain) vs 2 rows (two branches in one call)
            dt = {}
            for nrow in (1, 2):
                ts = []
                for _ in range(15):
                    t0 = time.perf_counter(); mtp_step(mx.concatenate([H[-1]] * nrow, axis=0) if nrow > 1 else H[-1], [x[-1]] * nrow)
                    ts.append((time.perf_counter() - t0) * 1e3)
                dt[nrow] = round(statistics.median(ts[3:]), 2)
            emit({"prompt": name, "context": len(ids) + GEN,
                  "verify_ms": {w: round(statistics.median(v[1:]), 2) for w, v in times.items()},
                  "draft_step_ms": dt, **g.report()})
        del cache, H, hs, h1, ha
