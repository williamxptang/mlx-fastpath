"""Audit for runs on any prompt file: argv results.jsonl prompts.json [methods...]. Per row: codes found, distinct
4-grams / longest repeat, wall vs reported tok/s, and a teacher-forced re-score by the plain target (argmax agreement,
mean NLL; outputs re-encoded from text). Guarded: MemoryGuard(14, 14). Prints JSON lines."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
import sys

from transformers import AutoTokenizer



def repetition(g):
    """Same as audit.py: distinct 4-gram ratio, longest repeated span (tokens)."""
    grams = [tuple(g[i:i + 4]) for i in range(len(g) - 3)]
    longest = 0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            k = 0
            while j + k < len(g) and g[i + k] == g[j + k]:
                k += 1
            longest = max(longest, k)
    return round(len(set(grams)) / max(1, len(grams)), 3), longest

TARGET = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
res, pf = sys.argv[1], sys.argv[2]
D = json.load(open(pf))
rows = [json.loads(l) for l in open(res)]
if sys.argv[3:]:
    rows = [r for r in rows if r["method"] in sys.argv[3:]]
assert rows, "no rows to audit"
tok = AutoTokenizer.from_pretrained(TARGET)
import mlx_fastpath as fp
with fp.MemoryGuard(min_free_gb=14, max_footprint_gb=14) as g:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    model = fp.load(TARGET, memory_limit_gb=12)._model
    for r in rows:
        prompt, gen = D["prompts"][r["prompt"]], tok.encode(r["text"], add_special_tokens=False)
        cache = make_prompt_cache(model)
        for i in range(0, len(prompt) - 1, 1024):
            model(mx.array(prompt[i:min(i + 1024, len(prompt) - 1)])[None], cache=cache)
            mx.eval([c.state for c in cache])
        logits = model(mx.array(prompt[-1:] + gen[:-1])[None], cache=cache)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = mx.array(gen)
        d4, rep = repetition(gen)
        print(json.dumps({"method": r["method"], "prompt": r["prompt"], "tokens_reencoded": len(gen),
                          "agree": round((logits.argmax(-1) == tgt).astype(mx.float32).mean().item(), 4),
                          "nll": round(-mx.take_along_axis(lp, tgt[:, None], axis=-1).mean().item(), 4),
                          "codes": sum(c in r["text"] for c in D["codes"]), "distinct4": d4, "longest_repeat": rep,
                          "wall_vs_reported": round(r["wall_tok_s"] / r["tok_s"], 3), **g.report()}), flush=True)
        del cache, logits, lp
