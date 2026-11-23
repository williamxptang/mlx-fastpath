"""Gemma 4 E2B int4 / Qwen 2B OptiQ, needle_4k prompt: first-token logits after prefill done (a) in one pass,
(b) as mlx-lm stream_generate does (2048-token chunks, then the last token), (c) as fastpath generate_ar does
(1024-token chunks of ids[:-1], then the last token). Reports top-2 tokens and the logit gap between stock's and
fastpath's first tokens under each scheme, and max |logit difference| vs the one-pass reference."""
import json
import sys
from pathlib import Path

import mlx.core as mx

from mlx_fastpath.guard import MemoryGuard

M = sys.argv[1]
D = json.load(open(Path(__file__).resolve().parents[1] / "ab" / "prompts_text.json"))
with MemoryGuard(min_free_gb=7, max_footprint_gb=6):
    mx.set_memory_limit(5 * 2**30)
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tok = load(M)
    r = tok.apply_chat_template([{"role": "user", "content": D["prompts"]["needle_4k"]}], add_generation_prompt=True,
                                tokenize=True, enable_thinking=False)
    ids = list(r["input_ids"] if hasattr(r, "keys") else r)

    def last_logits(chunks):
        c = make_prompt_cache(model)
        out = None
        for a, b in chunks:
            out = model(mx.array([ids[a:b]]), cache=c)
            mx.eval(out)
        return out[0, -1].astype(mx.float32)

    n = len(ids)
    schemes = {"one pass": [(0, n)],
               "mlx-lm (2048 chunks + last)": [(i, min(i + 2048, n - 1)) for i in range(0, n - 1, 2048)] + [(n - 1, n)],
               "fastpath (1024 chunks + last)": [(i, min(i + 1024, n - 1)) for i in range(0, n - 1, 1024)] + [(n - 1, n)]}
    L = {k: last_logits(v) for k, v in schemes.items()}
    ref = L["one pass"]
    for k, lg in L.items():
        top = mx.argsort(-lg)[:2].tolist()
        print(f"{k:32s} top2 {[tok.decode([t]) for t in top]} gap {(lg[top[0]] - lg[top[1]]).item():.3f} "
              f"| max |diff| vs one pass {mx.abs(lg - ref).max().item():.3f}")
    print("prompt tokens", n)
