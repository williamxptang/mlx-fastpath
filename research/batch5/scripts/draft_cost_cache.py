"""0.8B: does optiq's measure_draft_cost leave the live cache changed, and does our drafter's draft() raise inside it?"""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import mlx.core as mx
import mlx_fastpath as fp
from mlx_fastpath.guard import MemoryGuard
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache
from optiq.runtime import ngram_spec
M = MODELS + "/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed"
with MemoryGuard(min_free_gb=6, max_footprint_gb=4.5):
    eng = fp.load(M, memory_limit_gb=2.5, n2=False)
    tok, model = eng.tokenizer, eng._model
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Explain in five sentences why the sky is blue."}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
    def first_token_after(cost_with):
        c = make_prompt_cache(model)
        model(mx.array([ids[:-1]]), cache=c)
        print("  cache types:", sorted({type(x).__name__ for x in c}), "| can trim:", can_trim_prompt_cache(c))
        if cost_with is not None:
            cost = ngram_spec.measure_draft_cost(cost_with, model, cache=c)
            print("  draft cost:", round(cost, 3), "| drafter.ok after:", cost_with.ok)
        lg = model(mx.array([ids[-1:]]), cache=c)[0, -1]
        return tok.decode([int(mx.argmax(lg).item())])
    print("no costing        ->", first_token_after(None))
    fp.remove()
    print("stock MtpDrafter  ->", first_token_after(ngram_spec.MtpDrafter(model, depth=2)))
    fp.apply()
    d = ngram_spec.MtpDrafter(model, depth=2)
    print("drafter class:", type(d).__name__, "| ok:", d.ok)
    try:
        print("  our draft() on zeros:", d.draft(mx.zeros((1, 1, 1024), mx.bfloat16), 0))
    except Exception as e:
        print("  our draft() RAISES:", type(e).__name__, e)
    print("our FastMtpDrafter->", first_token_after(ngram_spec.MtpDrafter(model, depth=2)))
    fp.remove()
