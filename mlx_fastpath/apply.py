"""One entry point: apply the measured speedups, or refuse on untested package versions."""
import warnings
from importlib.metadata import PackageNotFoundError, version

from . import attention, draft, prefill

# Versions every speedup here was measured and audited on (M5, 2026-11-19; re-validated on Qwen3.5 0.8B/2B + Gemma 4 E2B, 2026-11-22).
TESTED = {"mlx": "0.32.4", "mlx-lm": "0.32.2", "mlx-optiq": "0.8.5"}
LONG_CONTEXT = 24_000  # above this, the 4-bit KV path beats bf16 MTP (16k: bf16 wins, 32k: 4-bit wins)


def installed_versions():
    out = {}
    for pkg in TESTED:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def apply(*, attention_dispatch=True, draft_vocab=draft.DEFAULT_VOCAB, draft_prompt_ids=False, fused_capture=True,
          prefill_step=prefill.DEFAULT_STEP, force=False):
    """Swap the speedups in at runtime. Returns what was enabled. Nothing on disk is changed.
    draft_prompt_ids: MTP drafts may also pick the prompt's own ids above draft_vocab. Default off: +5-12% on DFlash
    drafts, but only -1.5 to +1.9% on MTP (results.md §18.2, under its +3% bar). Lossless either way."""
    found = installed_versions()
    mismatch = {p: (found[p], want) for p, want in TESTED.items() if found[p] != want}
    if mismatch and not force:
        warnings.warn(f"mlx_fastpath: untested versions {mismatch}; running stock. Use force=True to override.",
                      stacklevel=2)
        return {}
    done = {}
    if attention_dispatch:
        attention.enable(); done["attention_dispatch"] = list(attention.MODULES)
    if draft_vocab:
        draft.enable_reduced_vocab(draft_vocab, prompt_ids=draft_prompt_ids); done["draft_vocab"] = draft_vocab
        if draft_prompt_ids:
            done["draft_prompt_ids"] = True
        draft.enable_fast_mtp_drafter(draft_vocab); done["ngram_mtp_drafter"] = draft_vocab
    if fused_capture:
        draft.enable_fused_capture(); done["fused_capture"] = True
    if prefill_step:
        prefill.enable(prefill_step); done["prefill_step"] = prefill_step
    return done


def remove():
    attention.disable()
    draft.disable_reduced_vocab()
    draft.disable_fast_mtp_drafter()
    draft.disable_fused_capture()
    prefill.disable()


def recommended(prompt_tokens):
    """generate() settings by context length, from the measured policy."""
    return {"depth": 2, "kv_bits": 4 if prompt_tokens > LONG_CONTEXT else None}


def generate(engine, prompt_ids, max_tokens=256, temperature=0.0, **overrides):
    """optiq engine generate (MTP) with the recommended depth / KV mode for this prompt length."""
    kw = recommended(len(prompt_ids)) | overrides
    draft.set_prompt(prompt_ids)                 # used only when apply(draft_prompt_ids=True)
    try:
        return engine.generate("", prompt_ids=list(prompt_ids), max_tokens=max_tokens, temperature=temperature,
                               depth=kw["depth"], kv_bits=kw["kv_bits"])
    finally:
        draft.set_prompt(None)


def generate_ngram(engine, prompt_ids, max_tokens=256, depth=2, num_draft=8):
    """Greedy n-gram (prompt-lookup) + MTP speculation; best when the answer copies from the prompt
    (code edits 4.1x, copying 3.5x plain decoding). Measured only on prompts up to ~1.2k tokens so far.
    Returns the generated token ids."""
    from optiq.runtime import ngram_spec
    import mlx.core as mx
    drafter = ngram_spec.MtpDrafter(engine._model, depth=depth)
    eos = engine._eos_ids()
    out = []
    for t, _ in ngram_spec.ngram_generate_step(mx.array(list(prompt_ids)), engine._model,
                                               context_tokens=list(prompt_ids), max_tokens=max_tokens,
                                               num_draft_tokens=num_draft, mtp_drafter=drafter,
                                               persist_costs=False):
        t = int(t)
        if t in eos or len(out) >= max_tokens:
            break
        out.append(t)
    return out
