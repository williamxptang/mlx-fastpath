"""Plain decoding (AR) and our setup A (mlx-fastpath MTP depth 2) on the shared prompts. Env PROMPTS / ONLY / MAX as in
dflash_child.py. cfg AR | A | Aq4 (A with a 4-bit KV cache, mlx_fastpath's fused 4-bit attention). MODEL (model folder,
default the OptiQ 4-bit 9B), TEMP (sampling temperature, default 0 = greedy; optiq samples by temperature only, no
top-p/top-k), SEED (mx.random seed before each prompt, default 0). Guarded: MemoryGuard(14, 14); GUARD_FREE_GB / GUARD_CAP_GB /
GUARD_SWAP_GB / MLX_LIMIT_GB / MLX_CACHE_GB override it and the MLX limits (defaults 14 / 14 / 0.0625 / 9 / 1)."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json, os, sys, time
import mlx_fastpath as fp
MAX = int(os.environ.get("MAX", 300))
MODEL = os.environ.get("MODEL", MODELS + "/Qwen3.5-9B-OptiQ-4bit")
TEMP = float(os.environ.get("TEMP", 0.0))
SEED = int(os.environ.get("SEED", 0))
cfg = sys.argv[1]
P = json.load(open(os.environ.get("PROMPTS", "prompts.json")))["prompts"]
if os.environ.get("ONLY"):
    P = {k: P[k] for k in os.environ["ONLY"].split(",")}
with fp.MemoryGuard(min_free_gb=float(os.environ.get("GUARD_FREE_GB", 14)), max_footprint_gb=float(os.environ.get("GUARD_CAP_GB", 14)),
                    swap_tolerance_gb=float(os.environ.get("GUARD_SWAP_GB", 0.0625))) as g:
    import mlx.core as mx
    eng = fp.load(MODEL, memory_limit_gb=float(os.environ.get("MLX_LIMIT_GB", 9)),
                  cache_limit_gb=float(os.environ.get("MLX_CACHE_GB", 1)))
    if cfg in ("A", "Aq4"):
        fp.apply()
    for name, ids in P.items():
        mx.random.seed(SEED)
        t0 = time.perf_counter()
        st = eng.generate("", prompt_ids=ids, max_tokens=MAX, temperature=TEMP, depth=0 if cfg == "AR" else 2,
                          kv_bits=4 if cfg == "Aq4" else None)
        wall = time.perf_counter() - t0
        print(json.dumps({"method": cfg, "prompt": name, "prompt_tokens": len(ids), "tokens": st.generated_tokens,
                          "tok_s": round(st.decode_tok_s, 2), "wall_tok_s": round(st.generated_tokens / (wall - st.prefill_time_s), 2),
                          "accept": round(st.acceptance_rate, 3) if cfg != "AR" else None, "prefill_s": round(st.prefill_time_s, 2),
                          "model": os.path.basename(MODEL), "temp": TEMP, "seed": SEED,
                          "text": st.text, **g.report()}), flush=True)
