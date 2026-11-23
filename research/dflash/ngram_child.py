"""N+ (optiq n-gram prompt lookup + 32k-vocab MTP2 drafter, as mlx_fastpath.generate_ngram after fp.apply()) on the
shared prompts, same output rows as the other children. Env PROMPTS / ONLY / MAX as in dflash_child.py; KV_BITS=4 for
a 4-bit KV cache (method "N+q4"). tok_s counts tokens after the first one over the time after it (prefill excluded);
wall_tok_s includes prefill. Guarded: MemoryGuard(14, 14)."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json, os, time
import mlx_fastpath as fp
MAX = int(os.environ.get("MAX", 300))
KV = int(os.environ.get("KV_BITS", 0)) or None
P = json.load(open(os.environ.get("PROMPTS", "prompts.json")))["prompts"]
if os.environ.get("ONLY"):
    P = {k: P[k] for k in os.environ["ONLY"].split(",")}
with fp.MemoryGuard(min_free_gb=14, max_footprint_gb=14) as g:
    eng = fp.load(MODELS + "/Qwen3.5-9B-OptiQ-4bit", memory_limit_gb=12)
    fp.apply()
    import mlx.core as mx
    from optiq.runtime import ngram_spec
    eos = eng._eos_ids()
    for name, ids in P.items():
        drafter = ngram_spec.MtpDrafter(eng._model, depth=2)  # = mlx_fastpath's 32k-vocab drafter after apply()
        out, t_first = [], None
        t0 = time.perf_counter()
        for t, _ in ngram_spec.ngram_generate_step(mx.array(ids), eng._model, context_tokens=list(ids), max_tokens=MAX,
                                                   num_draft_tokens=8, mtp_drafter=drafter, persist_costs=False,
                                                   prefill_step_size=1024, kv_bits=KV):
            if t_first is None:
                t_first = time.perf_counter()
            t = int(t)
            if t in eos or len(out) >= MAX:
                break
            out.append(t)
        end = time.perf_counter()
        print(json.dumps({"method": "N+q4" if KV else "N+", "prompt": name, "prompt_tokens": len(ids), "tokens": len(out),
                          "tok_s": round((len(out) - 1) / (end - t_first), 2), "wall_tok_s": round((len(out) - 1) / (end - t_first), 2),
                          "accept": None, "prefill_s": round(t_first - t0, 2),
                          "text": eng.tokenizer.decode(out), **g.report()}), flush=True)
