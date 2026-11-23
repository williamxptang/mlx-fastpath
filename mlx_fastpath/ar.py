"""Pipelined plain (autoregressive) greedy decoding.

optiq's AR loop builds the graph for token t+1 only after token t is read back (`.item()`), so the GPU idles while
Python builds the next graph. Here the next graph is built on the lazy argmax and queued with `async_eval` while the
GPU still runs the current token, as mlx-lm's `generate_step` does. No kernel change. Greedy, but NOT always the same
text as optiq's loop: on the 9B, 3 of 4 prompts split mid-text (results.md §18.2). The likely cause, not yet
verified, is bf16 near-ties tipped by the different prefill (chunks of `prefill_step`, then the last id alone).
Measured on Qwen3.5-9B-OptiQ-4bit (M5): +10-11% tok/s at short, 4k and 16k context (§14.3), +13-15% in §18.2.
Use it where MTP is off; with MTP, `generate()` (setup A) is faster.
"""
import time

import mlx.core as mx

DEFAULT_PREFILL_STEP = 1024


def _prefill(model, cache, ids, step):
    """Run ids through the model into the cache; logits are never evaluated (lazy), so the LM head is skipped."""
    for i in range(0, len(ids), step):
        model(mx.array(ids[i:i + step])[None], cache=cache)
        mx.eval([c.state for c in cache])


def generate_ar(engine, prompt_ids, max_tokens=256, prefill_step=DEFAULT_PREFILL_STEP):
    """Greedy decoding with the next step queued while the current one runs. Returns
    {"tokens": generated ids (EOS not included), "seconds": decode time, "tok_s": tokens / decode time}."""
    from mlx_lm.models.cache import make_prompt_cache
    model, eos = engine._model, set(engine._eos_ids())
    ids = list(prompt_ids)
    cache = make_prompt_cache(model)
    _prefill(model, cache, ids[:-1], prefill_step)
    t0 = time.perf_counter()
    nxt = mx.argmax(model(mx.array([ids[-1:]]), cache=cache)[0, -1], keepdims=True)
    mx.async_eval(nxt)
    out = []
    while len(out) < max_tokens:
        cur = nxt
        if len(out) + 1 < max_tokens:         # queue the step after this one before reading this one back
            nxt = mx.argmax(model(cur[None], cache=cache)[0, -1], keepdims=True)
            mx.async_eval(nxt)
        t = int(cur.item())
        if t in eos:                          # the queued step past EOS is dropped
            break
        out.append(t)
    el = time.perf_counter() - t0
    return {"tokens": out, "seconds": el, "tok_s": len(out) / el if el > 0 else 0.0}
