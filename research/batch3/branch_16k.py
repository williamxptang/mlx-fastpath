"""Where does a model leave the source's needle_16k answer, and was it a near-tie there?

The source (AR, greedy) answers needle_16k with "Based on the provided document, here are the three secret codes...";
the middle variant and both N2-all models start "Based on the research handoff..." or "Here is ..." and never list
the codes. first_token.py showed the first token decides nothing (the source's own top1 is "Based" too). This
teacher-forces the source's first 40 answer tokens (from dflash/baseline.jsonl, method AR) through MODEL and reports,
at every position, whether MODEL's argmax is the source's token. At the first position where it is not: MODEL's top 5
and the gap in nats between its top1 and the source's token. The source itself is the control (its gaps should be 0
except at chunk-width near-ties, §16.4).
argv: MODEL. Guard via GUARD_FREE_GB / GUARD_CAP_GB, MLX_LIMIT_GB / MLX_CACHE_GB (run.py sets them)."""
import json
import os
import sys
from pathlib import Path

import mlx_fastpath as fp

HERE = Path(__file__).parent
R = HERE.parent
IDS = json.load(open(R / "dflash" / "prompts.json"))["prompts"]["needle_16k"]
SRC_TEXT = next(json.loads(l)["text"] for l in open(R / "dflash" / "baseline.jsonl")
                if json.loads(l).get("method") == "AR" and json.loads(l).get("prompt") == "needle_16k")
N = 40
MODEL = sys.argv[1]

with fp.MemoryGuard(min_free_gb=float(os.environ.get("GUARD_FREE_GB", 14)),
                    max_footprint_gb=float(os.environ.get("GUARD_CAP_GB", 14))) as g:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    sys.path.insert(0, str(R / "batch2"))
    from kl_gate import _logp, prefill
    eng = fp.load(MODEL, memory_limit_gb=float(os.environ.get("MLX_LIMIT_GB", 10)),
                  cache_limit_gb=float(os.environ.get("MLX_CACHE_GB", 1)))
    lm, tok = eng._lm, eng._tokenizer
    ans = tok.encode(SRC_TEXT, add_special_tokens=False)[:N]
    cache = make_prompt_cache(lm)
    prefill(lm, cache, list(IDS[:-1]))
    lp = _logp(lm(mx.array([IDS[-1]] + ans[:-1])[None], cache=cache)[0])        # row i predicts ans[i]
    arg = mx.argmax(lp, axis=-1).tolist()
    gaps = [round(float(lp[i, arg[i]] - lp[i, ans[i]]), 3) for i in range(len(ans))]
    first = next((i for i in range(len(ans)) if arg[i] != ans[i]), None)
    row = {"prompt": "needle_16k", "model": MODEL, "positions": len(ans), "agree": sum(a == b for a, b in zip(arg, ans)),
           "first_diff": first, "peak_footprint_gb": g.report()["peak_footprint_gb"]}
    if first is not None:
        top = mx.argsort(-lp[first])[:5].tolist()
        row.update({"prefix": tok.decode(ans[:first]), "source_token": tok.decode([ans[first]]),
                    "model_token": tok.decode([arg[first]]), "gap_nats": gaps[first],
                    "top5": [(tok.decode([t]), round(float(lp[first, t]), 3)) for t in top]})
    row["gaps_over_0.5"] = [(i, tok.decode([ans[i]]), gaps[i]) for i in range(len(ans)) if gaps[i] > 0.5]
    print(json.dumps(row), flush=True)
    print(json.dumps({"guard": g.report()}))
