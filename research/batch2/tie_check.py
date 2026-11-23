"""Where two lossless greedy runs of the same model split (plain DFlash vs DFlash with the id-set draft head), was the
target's choice a near-tie? Teacher-forces the source model on prompt + the common prefix (cut back to the last space,
re-tokenized) and prints, at the split, the log-probs of each run's next token and the top-3.
argv: results jsonl (default r1_dflash.jsonl). ONLY: comma list of prompts. Long prompts are prefilled in chunks of 512
(one full-prompt pass would build 16k x 248k logits). Guard via GUARD_FREE_GB / GUARD_CAP_GB / GUARD_SWAP_GB
(default 14 / 14 / 0.0625)."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
import os
import sys
from pathlib import Path

import mlx_fastpath as fp

HERE = Path(__file__).parent
SRC = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
rows = [json.loads(l) for l in open(HERE / (sys.argv[1] if len(sys.argv) > 1 else "r1_dflash.jsonl"))]
P = json.load(open(HERE.parent / "dflash" / "prompts.json"))["prompts"]
if os.environ.get("ONLY"):
    rows = [r for r in rows if r["prompt"] in os.environ["ONLY"].split(",")]
CHUNK = 512

with fp.MemoryGuard(min_free_gb=float(os.environ.get("GUARD_FREE_GB", 14)),
                    max_footprint_gb=float(os.environ.get("GUARD_CAP_GB", 14)),
                    swap_tolerance_gb=float(os.environ.get("GUARD_SWAP_GB", 0.0625))) as g:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    eng = fp.load(SRC, memory_limit_gb=float(os.environ.get("MLX_LIMIT_GB", 10)),
                  cache_limit_gb=float(os.environ.get("MLX_CACHE_GB", 1)))
    lm, tok = eng._lm, eng._tokenizer
    for p in sorted({r["prompt"] for r in rows}):
        a = next(r["text"] for r in rows if r["prompt"] == p and r["variant"] == "plain")
        b = next(r["text"] for r in rows if r["prompt"] == p and r["variant"] == "idset")
        if a == b:
            print(json.dumps({"prompt": p, "split": None}), flush=True)
            continue
        i = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        cut = a.rfind(" ", 0, i)
        prefix, na, nb = a[:cut], a[cut:i + 12], b[cut:i + 12]
        ids = list(P[p]) + tok.encode(prefix, add_special_tokens=False)
        cache = make_prompt_cache(lm)
        for k in range(0, len(ids) - 1, CHUNK):
            lm(mx.array(ids[k:min(k + CHUNK, len(ids) - 1)])[None], cache=cache)
            mx.eval([c.state for c in cache])
        lo = lm(mx.array(ids[-1:])[None], cache=cache)[0, -1].astype(mx.float32)
        lp = lo - mx.logsumexp(lo)
        ta, tb = tok.encode(na, add_special_tokens=False)[0], tok.encode(nb, add_special_tokens=False)[0]
        top = mx.argsort(-lp)[:3].tolist()
        print(json.dumps({"prompt": p, "split_char": i, "context_cut_at": cut, "plain_next": tok.decode([ta]),
                          "idset_next": tok.decode([tb]), "logp_plain": round(float(lp[ta]), 3),
                          "logp_idset": round(float(lp[tb]), 3), "same_first_token": ta == tb,
                          "top3": [(tok.decode([t]), round(float(lp[t]), 3)) for t in top]}), flush=True)
    print(json.dumps({"guard": g.report()}))
