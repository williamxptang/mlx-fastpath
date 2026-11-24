"""Where does a Gemma 4 assistant-drafter pass spend its time (optiq 0.8.5's assistant_generate_step)?

Per model (one model per process: optiq is imported only after the load) and prompt:
  A. Real loop: optiq's assistant_generate_step for GEN tokens at gamma 1..3 -> tok/s, target calls (passes),
     acceptance; per-pass wall time = decode time / passes.
  B. Phase costs on a warm cache at the same context length, each a median over REPS:
     v(n)        target verify step with n tokens (forward + logits + read-back), rolled back after
     d_sync(g)   g drafter steps, each draft read back to Python before the next (what optiq does)
     d_lazy(g)   the same g drafter steps chained on the GPU, one read-back at the end
  Model: pass ~= d_sync(g) + v(g+1) + rest. "rest" = per-pass wall - that sum (Python, logsumexp, yields, trims).
  Possible saving from keeping drafts on the GPU = d_sync(g) - d_lazy(g).

    python loop_profile.py PROMPTS_JSON DRAFTER_DIR OUT_DIR MODEL_DIR
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # this repo's mlx_fastpath
import mlx.core as mx

import mlx_fastpath as fp
from mlx_fastpath.guard import MemoryGuard

prompts_json, drafter_dir, out_dir, mdir = sys.argv[1:5]
os.makedirs(out_dir, exist_ok=True)
GEN, REPS, GAMMAS = 128, 15, (1, 2, 3)
PROMPTS = {k: v for k, v in json.load(open(prompts_json))["prompts"].items() if k in ("chat", "code_edit", "needle_4k")}
if os.environ.get("SMOKE"):
    GEN, REPS, GAMMAS, PROMPTS = 24, 3, (2,), {"chat": PROMPTS["chat"]}
assert PROMPTS


def timed(fn):
    mx.synchronize()
    t = time.perf_counter()
    fn()
    mx.synchronize()
    return time.perf_counter() - t


with MemoryGuard(min_free_gb=8.2, max_footprint_gb=7.2) as guard, open(os.path.join(out_dir, "loop_profile.jsonl"), "a") as f:
    eng = fp.load(mdir, memory_limit_gb=5.2, n2=False)          # load BEFORE importing optiq
    fp.remove()
    from mlx_lm.models.cache import make_prompt_cache
    from optiq.runtime.spec import GemmaAssistantDrafter, SpecStats, assistant_generate_step
    from optiq.runtime.spec.kv_view import extract_typed_kv
    from optiq.runtime.spec.runtime import _target_embed, _target_step, _trim_cache

    model, tok = eng._model, eng.tokenizer
    drafter = GemmaAssistantDrafter.from_pretrained(drafter_dir)
    lm = model.language_model if hasattr(model, "language_model") else model

    for pname, ptext in PROMPTS.items():
        ids = tok.apply_chat_template([{"role": "user", "content": ptext}], add_generation_prompt=True, tokenize=True)
        rec = {"model": os.path.basename(mdir.rstrip("/")), "prompt": pname, "prompt_tokens": len(ids), "loop": {}}

        # A. the real loop
        for g in GAMMAS:
            cache, stats = make_prompt_cache(model), SpecStats()
            n, t_first = 0, None
            for _ in assistant_generate_step(mx.array(ids), model, drafter=drafter, gamma=g, max_tokens=GEN,
                                             prompt_cache=cache, stats=stats):
                n += 1
                if t_first is None:
                    t_first, calls0 = time.perf_counter(), stats.n_target_calls
            dt = time.perf_counter() - t_first
            passes = stats.n_target_calls - calls0
            rec["loop"][g] = {"tok_s": round((n - 1) / dt, 2), "passes": passes,
                              "ms_per_pass": round(1000 * dt / max(passes, 1), 2),
                              "tokens_per_pass": round((n - 1) / max(passes, 1), 3),
                              "accept": round(stats.acceptance_rate, 3)}
            del cache
            mx.clear_cache()

        # B. phase costs on a warm cache: prefill the prompt, one target step for a real hidden state
        cache = make_prompt_cache(model)
        for i in range(0, len(ids) - 1, 2048):
            model(mx.array(ids[i:min(i + 2048, len(ids) - 1)])[None], cache=cache)
            mx.eval([c.state for c in cache])
        hidden, logits = _target_step(model, mx.array([[ids[-1]]]), cache)
        mx.eval(hidden, logits)
        cur, h_prev = int(mx.argmax(logits[0, -1])), hidden[:, -1:, :]

        def verify(nn):
            h, lg = _target_step(model, mx.array([[cur] * nn]), cache)
            mx.argmax(lg[0], axis=-1).tolist()
            _trim_cache(cache, nn)

        def drafts(g, lazy):
            shared_kv, pos = extract_typed_kv(model, cache), cache[0].offset
            tok_in, h_in, out = cur, h_prev, []
            for k in range(g):
                emb = (_target_embed(model, tok_in) if not lazy
                       else lm.model.embed_tokens(tok_in.reshape(1, 1)) * lm.model.embed_scale)
                d_logits, h_in = drafter.forward(last_token_emb=emb, target_hidden=h_in, shared_kv=shared_kv,
                                                 position=pos + k)
                d = mx.argmax(d_logits.reshape(-1))
                if lazy:
                    out.append(d); tok_in = d
                else:
                    tok_in = int(d); out.append(tok_in)
            if lazy:
                mx.stack(out).tolist()

        if True:
            tok_in0 = cur
        costs = {"v": {}, "d_sync": {}, "d_lazy": {}}
        for nn in range(1, max(GAMMAS) + 2):
            verify(nn)                                              # warm-up
            costs["v"][nn] = round(1000 * statistics.median(timed(lambda: verify(nn)) for _ in range(REPS)), 3)
        for g in GAMMAS:
            for mode, lazy in (("d_sync", False), ("d_lazy", True)):
                if lazy:
                    cur = mx.array(tok_in0)
                drafts(g, lazy)                                     # warm-up
                costs[mode][g] = round(1000 * statistics.median(timed(lambda: drafts(g, lazy)) for _ in range(REPS)), 3)
                cur = tok_in0
        rec["phase_ms"] = costs
        for g in GAMMAS:
            L = rec["loop"][g]
            model_ms = costs["d_sync"][g] + costs["v"][g + 1]
            L["model_ms"] = round(model_ms, 2)
            L["rest_ms"] = round(L["ms_per_pass"] - model_ms, 2)
            L["draft_sync_saving_ms"] = round(costs["d_sync"][g] - costs["d_lazy"][g], 2)
        rec["plain_step_ms"] = costs["v"][1]
        del cache
        mx.clear_cache()
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(json.dumps(rec), flush=True)
    print(json.dumps({"guard": guard.report(), "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2)}), flush=True)
