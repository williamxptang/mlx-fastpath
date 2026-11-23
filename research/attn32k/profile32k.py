"""Item 2, step 1: where does a 32k bf16 MTP cycle spend its time on Qwen3.5-9B-OptiQ-4bit?
One prefill of needle_32k, then on the same cache (rolled back after each measurement):
  - verify pass (forward_with_gdn_capture) at widths 1-4 for each attention dispatch,
  - MTP2 (setup A drafter + fused capture) for GEN (env, default 200) tokens per dispatch: tok/s, acceptance, ms per cycle.
Dispatch modes: stock (mx.fast SDPA), pkg (mlx_fastpath rule: fold when gqa*qL >= 16), fold (fold whenever qL > 1),
and any extra kernels passed by name on the command line (research/attn32k/<name>.py providing hook(q, k, v, cache, scale), used for qL 2-4).
Guarded: MemoryGuard(14, 14), MLX limit 12 GB. Appends to profile32k.jsonl."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import mlx_fastpath as fp

HERE = Path(__file__).parent
TARGET = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "needle_32k"
EXTRA = sys.argv[2:]
GEN = int(__import__("os").environ.get("GEN", 200))
PL = json.load(open(HERE.parent / "dflash/prompts_long.json"))
ids, CODES = PL["prompts"][PROMPT], PL["codes"]
out = open(HERE / "profile32k.jsonl", "a")


def emit(row):
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n"); out.flush()


with fp.MemoryGuard(min_free_gb=14, max_footprint_gb=14) as g:
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    eng = fp.load(TARGET, memory_limit_gb=12)
    fp.apply()
    from mlx_fastpath import attention
    from optiq.runtime.engine import GenStats
    from optiq.runtime.mtp.cache_state import restore_cache, rollback_after_verify, snapshot_untrimmable_cache
    from optiq.runtime.mtp.gdn_capture import forward_with_gdn_capture

    def mode_fn(mode):
        if mode == "stock":
            return attention._stock
        if mode == "pkg":
            return attention.scaled_dot_product_attention
        if mode == "fold":
            def f(q, k, v, cache, scale, mask, sinks=None):
                if q.shape[0] == 1 and q.shape[2] > 1 and sinks is None and isinstance(mask, str):
                    return attention.folded_sdpa(q, k, v, scale)
                return attention._stock(q, k, v, cache, scale=scale, mask=mask, sinks=sinks)
            return f
        kern = importlib.import_module(mode).hook

        def f(q, k, v, cache, scale, mask, sinks=None):
            if q.shape[0] == 1 and 1 < q.shape[2] <= 4 and sinks is None and isinstance(mask, str) \
                    and getattr(cache, "keys", None) is not None and cache.keys.dtype == mx.bfloat16:
                return kern(q, k, v, cache, scale)
            return attention._stock(q, k, v, cache, scale=scale, mask=mask, sinks=sinks)
        return f

    import mlx_lm.models.qwen3_next as qn
    sys.path.insert(0, str(HERE))
    t0 = time.perf_counter()
    cache = make_prompt_cache(eng._model)
    hidden, logits = eng._prefill_chunked(ids, cache, step=1024)
    tok = int(mx.argmax(logits[0, -1]).item()); hid = hidden[:, -1:, :]
    emit({"prompt": PROMPT, "prompt_tokens": len(ids), "prefill_s": round(time.perf_counter() - t0, 1), **g.report()})
    x = [tok]
    for _ in range(8):  # a few AR tokens to have real verify inputs
        hidden, logits = eng._forward(mx.array([[x[-1]]]), cache)
        x.append(int(mx.argmax(logits[0, -1]).item())); hid = hidden
    base_offset = cache[eng._inner.fa_idx].offset
    import os
    modes = (os.environ.get("MODES") or "stock,pkg,fold").split(",") + EXTRA

    times = {(m, w): [] for m in modes for w in range(1, 5)}
    for rep in range(6):
        for m in modes:
            qn.scaled_dot_product_attention = mode_fn(m)
            for w in range(1, 5):
                snap = snapshot_untrimmable_cache(cache)
                mx.eval([c.state for c in cache])
                t0 = time.perf_counter()
                lo, hd, cap = forward_with_gdn_capture(eng._model, mx.array([x[-w:]]), cache=cache, return_hidden=True)
                mx.eval(lo, hd)
                times[(m, w)].append((time.perf_counter() - t0) * 1e3)
                rollback_after_verify(cache, snap, verified_tokens=w)
                del lo, hd, cap
    ref_lo = None
    for m in ["stock"] + modes:
        qn.scaled_dot_product_attention = mode_fn(m)
        snap = snapshot_untrimmable_cache(cache)
        lo, _, _ = forward_with_gdn_capture(eng._model, mx.array([x[-3:]]), cache=cache, return_hidden=True)
        lo = lo.astype(mx.float32); mx.eval(lo)
        rollback_after_verify(cache, snap, verified_tokens=3)
        if ref_lo is None:
            ref_lo = lo
        else:
            emit({"prompt": PROMPT, "mode": m, "w3_logit_maxdiff_vs_stock": round(mx.abs(lo - ref_lo).max().item(), 3),
                  "w3_argmax_same": bool((lo.argmax(-1) == ref_lo.argmax(-1)).all().item())})
    for m in modes:
        emit({"prompt": PROMPT, "mode": m, "context": base_offset,
              "verify_ms": {w: round(statistics.median(times[(m, w)][1:]), 2) for w in range(1, 5)}, **g.report()})

    texts = {}
    for m in modes:
        qn.scaled_dot_product_attention = mode_fn(m)
        snap = snapshot_untrimmable_cache(cache)
        mx.eval([c.state for c in cache])
        st = GenStats(prompt_tokens=len(ids), mode="MTP", depth=2)
        gen = []
        t0 = time.perf_counter()
        for t, _ in eng._mtp_loop_iter(cache, hid, x[-1], GEN, 2, 0.0, eng._eos_ids(), st):
            gen.append(t)
        el = time.perf_counter() - t0
        n_new = cache[eng._inner.fa_idx].offset - base_offset
        for c in cache:
            if c.is_trimmable():
                c.trim(c.offset - base_offset)
        restore_cache(cache, snap)
        qn.scaled_dot_product_attention = attention._stock
        snap = snapshot_untrimmable_cache(cache)
        rl = eng._forward(mx.array([[x[-1]] + gen[:-1]]), cache)[1][0].astype(mx.float32)
        lp = rl - mx.logsumexp(rl, axis=-1, keepdims=True)
        tg = mx.array(gen)
        agree = (rl.argmax(-1) == tg).astype(mx.float32).mean().item()
        nll = -mx.take_along_axis(lp, tg[:, None], axis=-1).mean().item()
        for c in cache:
            if c.is_trimmable():
                c.trim(c.offset - base_offset)
        restore_cache(cache, snap)
        del rl, lp
        texts[m] = gen
        emit({"prompt": PROMPT, "mode": m, "mtp2_tok_s": round(len(gen) / el, 2), "tokens": len(gen),
              "accept": round(st.drafts_accepted / max(1, st.drafts_attempted), 3),
              "cycles": st.speculation_cycles, "ms_per_cycle": round(el * 1e3 / max(1, st.speculation_cycles), 1),
              "same_as_first_mode": gen == texts.get(modes[0], gen), "rescore_agree": round(agree, 4), "rescore_nll": round(nll, 4),
              "codes": sum(c in eng.tokenizer.decode(x + gen) for c in CODES), "cache_added": n_new, **g.report()})
    qn.scaled_dot_product_attention = attention.scaled_dot_product_attention
    json.dump({m: eng.tokenizer.decode(v) for m, v in texts.items()}, open(HERE / f"profile32k_{PROMPT}_{GEN}_texts.json", "w"),
              ensure_ascii=False, indent=1)
