"""KL quality gate (papers-scan N3): how far does a candidate model's next-token distribution move from a reference?

ref:   the reference model decodes greedily, one token at a time (as plain decoding does), and stores the full
       next-token log-probs (float32) at every generated position, plus the token ids.
score: a candidate model is teacher-forced on the same prompt + ids, in chunks (as a speculative verify pass is),
       and compared position by position with the stored log-probs:
         kl       KL(ref || cand) in nats (mean, p99, max)
         flips    candidate argmax != reference argmax (= the reference's greedy token)
         sure_flips  flips where the reference's top-1 beat its top-2 by > 1 nat (not a near-tie)
Control: score the reference model itself. Its numbers are the floor (width-1 decode vs chunked forward, different
reduction order); a candidate is judged against that floor, not against 0.

Usage (from research/batch2/):
  python kl_gate.py ref   MODEL REFDIR
  python kl_gate.py score MODEL REFDIR LABEL [CHUNK=64]      -> appends to kl.jsonl
Env: PROMPTS (default ../dflash/prompts.json), ONLY (comma list), MAX (tokens, default 300),
     GUARD_FREE_GB / GUARD_CAP_GB (MemoryGuard; default 14 / 14, the run rule: change only with the user's OK),
     MLX_LIMIT_GB / MLX_CACHE_GB (defaults 10 / 1).
REFDIR holds ~1 MB per scored position (300 tokens x 5 prompts = 1.5 GB): keep it outside the repo.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SURE = 1.0   # nats between the reference's top-1 and top-2 for a flip to count as a real disagreement


# ---------- model-independent pieces ----------
def compare(ref_logp, cand_logp):
    """Per-position metrics, all [n]: kl, flip, sure_flip, ref_margin. Inputs: [n, V] float32 log-probs."""
    import mlx.core as mx
    kl = mx.sum(mx.exp(ref_logp) * (ref_logp - cand_logp), axis=-1)
    top2 = mx.topk(ref_logp, 2, axis=-1)
    margin = mx.max(top2, axis=-1) - mx.min(top2, axis=-1)
    flip = mx.argmax(cand_logp, axis=-1) != mx.argmax(ref_logp, axis=-1)
    return {"kl": kl, "flip": flip, "sure_flip": mx.logical_and(flip, margin > SURE), "ref_margin": margin}


def summarize(m):
    kl = np.asarray(m["kl"], dtype=np.float64)
    assert kl.size, "no positions scored"
    return {"positions": int(kl.size), "kl_mean": round(float(kl.mean()), 5), "kl_p99": round(float(np.quantile(kl, 0.99)), 4),
            "kl_max": round(float(kl.max()), 4), "flip_rate": round(float(np.mean(m["flip"])), 4),
            "sure_flip_rate": round(float(np.mean(m["sure_flip"])), 4)}


def _logp(logits):
    import mlx.core as mx
    lo = logits.astype(mx.float32)
    return lo - mx.logsumexp(lo, axis=-1, keepdims=True)


# ---------- forward passes ----------
def prefill(lm, cache, ids, step=2048):
    """Run ids through the model into the cache without computing logits (lazy: only cache state is evaluated)."""
    import mlx.core as mx
    for i in range(0, len(ids), step):
        lm(mx.array(ids[i:i + step])[None], cache=cache)
        mx.eval([c.state for c in cache])


def greedy_reference(lm, prompt, n, eos, on_row):
    """Plain greedy decoding, one token per step; on_row(i, logp[V]) gets each step's log-probs. Returns the ids."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    prefill(lm, cache, prompt[:-1])
    tok, ids = prompt[-1], []
    for i in range(n):
        logp = _logp(lm(mx.array([[tok]]), cache=cache)[0, -1])
        tok = int(mx.argmax(logp))
        on_row(i, logp)
        ids.append(tok)
        if tok in eos:
            break
    return ids


def teacher_forced(lm, prompt, ids, chunk):
    """Yields (start, logp[c, V]): the model's log-probs for ids[start:start+c], given prompt + ids[:start]."""
    from mlx_lm.models.cache import make_prompt_cache
    import mlx.core as mx
    cache = make_prompt_cache(lm)
    prefill(lm, cache, prompt[:-1])
    inp = [prompt[-1]] + list(ids[:-1])
    for i in range(0, len(inp), chunk):
        yield i, _logp(lm(mx.array(inp[i:i + chunk])[None], cache=cache)[0])


def score_prompt(lm, prompt, ids, ref_rows, chunk):
    """ref_rows(start, end) -> [end-start, V] float32 reference log-probs. Returns per-position metric arrays."""
    import mlx.core as mx
    parts = []
    for start, cand in teacher_forced(lm, prompt, ids, chunk):
        ref = mx.array(ref_rows(start, start + cand.shape[0]))
        m = compare(ref, cand)
        mx.eval(m)
        parts.append({k: np.array(v) for k, v in m.items()})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------- CLI ----------
def _prompts():
    P = json.load(open(os.environ.get("PROMPTS", HERE.parent / "dflash" / "prompts.json")))["prompts"]
    if os.environ.get("ONLY"):
        P = {k: P[k] for k in os.environ["ONLY"].split(",")}
    return P


def _load(model):
    import mlx_fastpath as fp
    eng = fp.load(model, memory_limit_gb=float(os.environ.get("MLX_LIMIT_GB", 10)),
                  cache_limit_gb=float(os.environ.get("MLX_CACHE_GB", 1)))
    return eng._lm, eng


def cmd_ref(model, refdir):
    import mlx.core as mx
    refdir = Path(refdir)
    refdir.mkdir(parents=True, exist_ok=True)
    n = int(os.environ.get("MAX", 300))
    lm, eng = _load(model)
    tok = eng.tokenizer if hasattr(eng, "tokenizer") else eng._tokenizer
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    meta = {"model": str(model), "date": time.strftime("%Y-%m-%d %H:%M"), "prompts": {}}
    for name, prompt in _prompts().items():
        V = None
        arr = {}

        def on_row(i, logp):
            nonlocal V
            if V is None:
                V = logp.shape[-1]
                arr["a"] = np.lib.format.open_memmap(refdir / f"{name}.npy", mode="w+", dtype=np.float32, shape=(n, V))
            arr["a"][i] = np.array(logp)

        t0 = time.time()
        ids = greedy_reference(lm, list(prompt), n, eos, on_row)
        arr["a"].flush()
        meta["prompts"][name] = {"prompt_tokens": len(prompt), "ids": ids, "text": tok.decode(ids)}
        print(json.dumps({"ref": name, "tokens": len(ids), "seconds": round(time.time() - t0, 1)}), flush=True)
        mx.clear_cache()
    json.dump(meta, open(refdir / "ref.json", "w"))


def cmd_score(model, refdir, label, chunk=64):
    import mlx.core as mx
    refdir = Path(refdir)
    meta = json.load(open(refdir / "ref.json"))
    lm, _ = _load(model)
    rows, out = [], open(HERE / "kl.jsonl", "a")
    allm = []
    for name, prompt in _prompts().items():
        ids = meta["prompts"][name]["ids"]
        ref = np.load(refdir / f"{name}.npy", mmap_mode="r")
        t0 = time.time()
        m = score_prompt(lm, list(prompt), ids, lambda a, b: ref[a:b], int(chunk))
        np.savez(refdir / f"score_{label}_{name}.npz", **m)
        allm.append(m)
        row = {"label": label, "model": str(model), "ref": meta["model"], "prompt": name, "chunk": int(chunk),
               **summarize(m), "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        mx.clear_cache()
    tot = {k: np.concatenate([m[k] for m in allm]) for k in allm[0]}
    rows.append({"label": label, "model": str(model), "ref": meta["model"], "prompt": "ALL", "chunk": int(chunk), **summarize(tot)})
    print(json.dumps(rows[-1]), flush=True)
    for r in rows:
        out.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    import mlx_fastpath as fp
    free, cap = float(os.environ.get("GUARD_FREE_GB", 14)), float(os.environ.get("GUARD_CAP_GB", 14))
    with fp.MemoryGuard(min_free_gb=free, max_footprint_gb=cap, swap_tolerance_gb=float(os.environ.get("GUARD_SWAP_GB", 0.0625))) as g:
        if sys.argv[1] == "ref":
            cmd_ref(sys.argv[2], sys.argv[3])
        elif sys.argv[1] == "score":
            cmd_score(sys.argv[2], sys.argv[3], sys.argv[4], *(sys.argv[5:6]))
        else:
            raise SystemExit(__doc__)
        print(json.dumps({"guard": g.report()}))
