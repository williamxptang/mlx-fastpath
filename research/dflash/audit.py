"""Audit for "fast but rubbish" on baseline.jsonl + dflash.jsonl. argv: text | rescore <method>...
text:    hidden codes found, distinct 4-grams / longest repeat, code_edit edits, tokens shared with AR before divergence,
         wall tok/s vs reported tok/s. No model.
rescore: teacher-forced re-score of each method's outputs by the plain target (argmax agreement, mean NLL). Guarded.
Outputs are re-encoded from text (the children keep text, not ids); the token counts are printed to show the round trip."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
import sys

from transformers import AutoTokenizer

TARGET = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
D = json.load(open("prompts.json"))
P, CODES = D["prompts"], D["codes"]
rows = [json.loads(l) for f in ("baseline.jsonl", "dflash.jsonl") for l in open(f)]
tok = AutoTokenizer.from_pretrained(TARGET)
ids = {(r["method"], r["prompt"]): tok.encode(r["text"], add_special_tokens=False) for r in rows}


def repetition(g):
    grams = [tuple(g[i:i + 4]) for i in range(len(g) - 3)]
    longest = 0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            k = 0
            while j + k < len(g) and g[i + k] == g[j + k]:
                k += 1
            longest = max(longest, k)
    return round(len(set(grams)) / max(1, len(grams)), 3), longest


def shared_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def text_audit():
    src = open("../../mlx_fastpath/attention.py").read()
    want = src.replace('MODULES = ("mlx_lm.models.qwen3_next",)',
                       'MODULES = ("mlx_lm.models.qwen3_next", "mlx_lm.models.qwen3_5_moe")').replace(
        "MAX_VERIFY = 16", "MAX_VERIFY = 12")
    assert want != src
    for r in rows:
        m, p, t = r["method"], r["prompt"], r["text"]
        out = {"method": m, "prompt": p, "tokens_reencoded": len(ids[(m, p)]),
               "wall_vs_reported": round(r["wall_tok_s"] / r["tok_s"], 3)}
        out["distinct4"], out["longest_repeat"] = repetition(ids[(m, p)])
        out["same_as_AR_tokens"] = shared_prefix(ids[(m, p)], ids[("AR", p)])
        if p.startswith("needle"):
            out["codes"] = f"{sum(c in t for c in CODES)}/{len(CODES)}"
        if p == "code_edit":
            body = t.split("```python\n", 1)[-1]
            out["moe_added"] = "mlx_lm.models.qwen3_5_moe" in body
            out["max_verify_12"] = "MAX_VERIFY = 12" in body
            out["verbatim_chars_of_want"] = shared_prefix(body, want)
            out["want_chars"] = len(want)
        print(json.dumps(out), flush=True)


def rescore(methods):
    import mlx_fastpath as fp
    with fp.MemoryGuard(min_free_gb=11, max_footprint_gb=11) as g:
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        model = fp.load(TARGET, memory_limit_gb=9)._model
        for m in methods:
            for p, prompt in P.items():
                gen = ids[(m, p)]
                cache = make_prompt_cache(model)
                seq = prompt + gen
                for i in range(0, len(prompt) - 1, 256):
                    model(mx.array(seq[i:min(i + 256, len(prompt) - 1)])[None], cache=cache)
                    mx.eval([c.state for c in cache])
                logits = model(mx.array(seq[len(prompt) - 1:-1])[None], cache=cache)[0].astype(mx.float32)
                lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                tgt = mx.array(gen)
                nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).mean().item()
                agree = (logits.argmax(-1) == tgt).astype(mx.float32).mean().item()
                del cache, logits, lp
                print(json.dumps({"method": m, "prompt": p, "agree": round(agree, 4), "nll": round(nll, 4),
                                  **g.report()}), flush=True)


if sys.argv[1] == "text":
    text_audit()
else:
    rescore(sys.argv[2:])
