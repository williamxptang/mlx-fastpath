"""Was the needle prompts' first-token choice a near-tie? (§17.5: the middle variant and both N2-all models open the
16k needle answer with a summary instead of the codes; every model does so on needle_24k_v3.)

For one model and each prompt: prefill the prompt but its last token, then print the next-token distribution's top 10
(token, log-prob) and the margin top1 - top2 in nats. Greedy decoding takes top1, so a margin near 0 means the
pass/fail on that prompt was a coin flip for this model.
argv: MODEL. Prompts: needle_4k and needle_16k (dflash/prompts.json), needle_24k_v3 (batch2/prompts_var.json).
Guard via GUARD_FREE_GB / GUARD_CAP_GB, MLX_LIMIT_GB / MLX_CACHE_GB (run.py sets them)."""
import json
import os
import sys
from pathlib import Path

import mlx_fastpath as fp

HERE = Path(__file__).parent
R = HERE.parent
PROMPTS = {**{k: v for k, v in json.load(open(R / "dflash" / "prompts.json"))["prompts"].items()
              if k in ("needle_4k", "needle_16k")},
           "needle_24k_v3": json.load(open(R / "batch2" / "prompts_var.json"))["prompts"]["needle_24k_v3"]}
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
    for name, ids in PROMPTS.items():
        cache = make_prompt_cache(lm)
        prefill(lm, cache, list(ids[:-1]))
        lp = _logp(lm(mx.array(ids[-1:])[None], cache=cache)[0, -1])
        top = mx.argsort(-lp)[:10].tolist()
        vals = [round(float(lp[t]), 3) for t in top]
        print(json.dumps({"prompt": name, "model": MODEL, "prompt_tokens": len(ids),
                          "top10": [(tok.decode([t]), v) for t, v in zip(top, vals)],
                          "margin_nats": round(vals[0] - vals[1], 3),
                          "peak_footprint_gb": g.report()["peak_footprint_gb"]}), flush=True)
        del cache
        mx.clear_cache()
    print(json.dumps({"guard": g.report()}))
