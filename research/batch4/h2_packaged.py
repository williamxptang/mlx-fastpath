"""H2: the packaged wins on the 9B, one configuration per process (the runner interleaves them ABBA).
argv: CFG = ar | A | A-idset.   env N2 = 0 | 1 (the load() switch).
  ar       fp.generate_ar: pipelined plain decoding (the baseline is optiq's own AR loop, baseline_child.py AR)
  A        fp.apply() + fp.generate: MTP depth 2 with the package's defaults
  A-idset  as A, with apply(draft_prompt_ids=True)
Prompts chat, math, code_edit, needle_4k (research/dflash/prompts.json), greedy, 300 tokens. One JSON row per prompt.
Bars are in research/batch4/plan.md (H2). Env: GUARD_FREE_GB / GUARD_CAP_GB / MLX_LIMIT_GB / MLX_CACHE_GB."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
import os
import sys
import time
from pathlib import Path

import mlx_fastpath as fp

HERE = Path(__file__).parent
MODEL = os.environ.get("MODEL", MODELS + "/Qwen3.5-9B-OptiQ-4bit")
CFG = sys.argv[1]
N2 = os.environ.get("N2", "1") == "1"
D = json.load(open(HERE.parent / "dflash" / "prompts.json"))
ONLY = os.environ.get("ONLY", "chat,math,code_edit,needle_4k").split(",")
MAX = 300
assert CFG in ("ar", "A", "A-idset"), CFG

with fp.MemoryGuard(min_free_gb=float(os.environ.get("GUARD_FREE_GB", 14)),
                    max_footprint_gb=float(os.environ.get("GUARD_CAP_GB", 14))) as g:
    from mlx_fastpath.n2 import resolve_model
    used = str(resolve_model(MODEL, N2))
    eng = fp.load(MODEL, memory_limit_gb=float(os.environ.get("MLX_LIMIT_GB", 10)),
                  cache_limit_gb=float(os.environ.get("MLX_CACHE_GB", 0.25)), n2=N2)
    applied = fp.apply(draft_prompt_ids=(CFG == "A-idset")) if CFG != "ar" else {}
    tok = eng.tokenizer
    for name in ONLY:
        ids = D["prompts"][name]
        t0 = time.perf_counter()
        if CFG == "ar":
            r = fp.generate_ar(eng, ids, max_tokens=MAX)
            toks, tok_s, accept = r["tokens"], r["tok_s"], None
            text = tok.decode(toks)
        else:
            st = fp.generate(eng, ids, max_tokens=MAX)
            text, tok_s, accept = st.text, st.decode_tok_s, st.acceptance_rate
            toks = range(st.generated_tokens)                 # the engine's own count
        total = time.perf_counter() - t0
        print(json.dumps({"prompt": name, "cfg": CFG, "n2": N2, "model": used, "prompt_tokens": len(ids),
                          "tokens": len(toks), "tok_s": round(tok_s, 2), "total_s": round(total, 2),
                          "accept": None if accept is None else round(accept, 3), "applied": sorted(applied),
                          **({"codes": sum(c in text for c in D["codes"])} if name.startswith("needle") else {}),
                          "text": text, "peak_footprint_gb": g.report()["peak_footprint_gb"]}), flush=True)
    print(json.dumps({"guard": g.report()}))
