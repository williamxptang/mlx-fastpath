"""DFlash on the shared prompts. argv: label draft_quant(none|w4:gs64|w8:gs64) verify_mode verify_len_cap(0=block).
DFLASH_VERIFY_QMM is set by the parent. Env: PROMPTS (json file, default prompts.json), ONLY (comma list of prompt
names), MAX (tokens, default 300), VOCAB (draft only ids < VOCAB with mlx_fastpath's reduced-vocab head, runtime swap of
the draft backend's head; the target still verifies with the full head), VOCAB_PROMPT=1 (with VOCAB: the draft also
scores every id in the prompt, rebuilt per prompt, see subhead_ids.py), DRAFT_WINDOW (draft_window_size; dflash-mlx
raises it to the drafter's sliding_window, 4096, results.md §15.4), PROFILE=1 (dflash-mlx's own per-cycle trace; adds a "cycles" summary to each row). Guarded:
MemoryGuard(14, 14) (next round rule); GUARD_FREE_GB / GUARD_CAP_GB / MLX_LIMIT_GB / MLX_CACHE_GB override it
and the MLX limits (defaults 14 / 14 / 10 / 1)."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import os
import json
import sys
import time

import mlx_fastpath as fp

label, dq, vmode, vcap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
MAX = int(os.environ.get("MAX", 300))
VOCAB = int(os.environ.get("VOCAB", 0))
VOCAB_PROMPT = os.environ.get("VOCAB_PROMPT") == "1"
assert VOCAB or not VOCAB_PROMPT, "VOCAB_PROMPT needs VOCAB"
WINDOW = int(os.environ.get("DRAFT_WINDOW", 0)) or None
PROFILE = os.environ.get("PROFILE") == "1"
TARGET = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
DRAFT = MODELS + "/Qwen3.5-9B-DFlash"
P = json.load(open(os.environ.get("PROMPTS", "prompts.json")))["prompts"]
if os.environ.get("ONLY"):
    P = {k: P[k] for k in os.environ["ONLY"].split(",")}

with fp.MemoryGuard(min_free_gb=float(os.environ.get("GUARD_FREE_GB", 14)), max_footprint_gb=float(os.environ.get("GUARD_CAP_GB", 14)),
                    swap_tolerance_gb=float(os.environ.get("GUARD_SWAP_GB", 0.0625))) as g:
    import mlx.core as mx
    mx.set_memory_limit(int(float(os.environ.get("MLX_LIMIT_GB", 10)) * 2**30))
    mx.set_cache_limit(int(float(os.environ.get("MLX_CACHE_GB", 1)) * 2**30))
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    from dflash_mlx.model import DFlashDraftModelArgs

    # The drafter's config (transformers 5.x) nests rope_theta and block_size; dflash-mlx 0.1.8 wants them top-level.
    _from_dict = DFlashDraftModelArgs.from_dict.__func__

    def _lift(cls, params):
        params = dict(params)
        params.setdefault("rope_theta", (params.get("rope_parameters") or {}).get("rope_theta"))
        params.setdefault("block_size", (params.get("dflash_config") or {}).get("block_size"))
        return _from_dict(cls, params)

    DFlashDraftModelArgs.from_dict = classmethod(_lift)
    from dflash_mlx.generate import generation_tps_from_summary
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context

    ctx = build_offline_runtime_context(verify_mode=vmode, verify_len_cap=vcap or None, draft_window_size=WINDOW)
    if PROFILE:
        from dataclasses import replace
        from dflash_mlx.diagnostics import DiagnosticsConfig, TraceConfig
        ctx = replace(ctx, diagnostics=DiagnosticsConfig(trace=TraceConfig(cycle_events=True)))
    b = load_runtime_bundle(model_ref=TARGET, draft_ref=DRAFT, draft_quant=None if dq == "none" else dq,
                            verify_config=ctx.verify)
    stop = get_stop_token_ids(b.tokenizer)
    if VOCAB:
        import dflash_mlx.draft_backend as db
        from mlx_fastpath.draft import sub_head, sub_logits
        assert type(b.draft_backend) is db.EagerDraftBackend, type(b.draft_backend)
        wrapper = b.target_ops.text_wrapper(b.target_model)
        assert not getattr(wrapper.args, "tie_word_embeddings", True)
        from subhead_ids import draft_ids, rows_head, subset_logits
        full_vocab = int(wrapper.lm_head.weight.shape[0])
        cur = {"sub": sub_head(wrapper.lm_head, VOCAB), "idx": None, "rows": VOCAB}

        class SubHeadOps:
            """target_ops for the draft call only. VOCAB alone: logits over ids < VOCAB (index = id, so argmax is
            unchanged). With VOCAB_PROMPT: the id set's logits scattered back to a full-vocab row."""
            def __init__(self, ops): self._ops = ops
            def __getattr__(self, n): return getattr(self._ops, n)

            def logits_from_hidden(self, tm, h):
                if cur["idx"] is None:
                    return sub_logits(cur["sub"], h)
                return subset_logits(cur["sub"], cur["idx"], h, full_vocab)

        _orig_draft, _orig_greedy = db.EagerDraftBackend.draft_greedy, db.greedy_tokens_with_mask
        db.EagerDraftBackend.draft_greedy = lambda self, *a, **k: _orig_draft(self, *a, **(k | {"target_ops": SubHeadOps(k["target_ops"])}))
        db.greedy_tokens_with_mask = lambda lo, m=None: _orig_greedy(lo, None if m is None else m[..., :lo.shape[-1]])
    for name, ids in P.items():
        toks, summary = [], None
        if VOCAB_PROMPT:   # built before the clock starts: a one-off gather of ~33k head rows per prompt
            idx = draft_ids(ids, VOCAB)
            cur.update(idx=idx, sub=rows_head(wrapper.lm_head, idx), rows=int(idx.shape[0]))
            mx.eval([a for a in cur["sub"] if isinstance(a, mx.array)])
        t0 = time.perf_counter()
        stream = stream_dflash_generate(target_model=b.target_model, target_ops=b.target_ops, tokenizer=b.tokenizer,
                                        draft_model=b.draft_model, draft_backend=b.draft_backend, prompt="",
                                        prompt_tokens_override=list(ids), max_new_tokens=MAX, use_chat_template=False,
                                        stop_token_ids=stop, runtime_context=ctx)
        try:
            for ev in stream:
                if isinstance(ev, TokenEvent):
                    toks.append(int(ev.token_id))
                elif isinstance(ev, SummaryEvent):
                    summary = ev
        finally:
            getattr(stream, "close", lambda: None)()
        wall = time.perf_counter() - t0
        pre = float(dict(summary.phase_timings_us).get("prefill", 0.0)) / 1e6
        extra = {"draft_window": summary.draft_window_size, "cycles_n": int(summary.cycles_completed),
                 "tokens_per_cycle": round(float(summary.tokens_per_cycle), 3),
                 "adaptive_reductions": int(summary.adaptive_block_reductions),
                 "adaptive_cycles": int(summary.adaptive_block_cycles), "adaptive_block_min": summary.adaptive_block_min}
        if VOCAB:
            extra["draft_rows"] = cur["rows"]
        if PROFILE:
            import statistics as st
            cy = list(summary.cycle_profile_us or [])
            med = lambda f: round(st.median(f(c) for c in cy), 3) if cy else None
            extra["cycles"] = {"n": len(cy), "draft_ms_med": med(lambda c: c.draft_us / 1e3),
                               "verify_ms_med": med(lambda c: c.verify_us / 1e3), "cycle_ms_med": med(lambda c: c.cycle_total_us / 1e3),
                               "commit_mean": round(sum(c.commit_count for c in cy) / max(1, len(cy)), 3),
                               "block_len_hist": {str(k): sum(c.block_len == k for c in cy) for k in sorted({c.block_len for c in cy})},
                               "verify_tokens_hist": {str(k): sum(c.verify_token_count == k for c in cy) for k in sorted({c.verify_token_count for c in cy}, key=str)},
                               "share_ms": {k: round(sum(getattr(c, k) for c in cy) / max(1e-9, sum(c.cycle_total_us for c in cy)), 3)
                                            for k in ("draft_us", "verify_us", "acceptance_us", "hidden_extraction_us", "rollback_us", "commit_us", "other_us")}}
        print(json.dumps({"method": label, "prompt": name, "prompt_tokens": len(ids), "tokens": int(summary.generation_tokens),
                          "tok_s": round(generation_tps_from_summary(summary), 2),
                          "wall_tok_s": round(len(toks) / max(1e-9, wall - pre), 2),
                          "accept": round(float(summary.acceptance_ratio), 3), "prefill_s": round(pre, 2),
                          **extra, "text": b.tokenizer.decode(toks), **g.report()}), flush=True)
