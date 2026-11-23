"""Gemma 4 E2B OptiQ: does it run, and say something sensible, under mlx-lm stock, optiq's engine (AR), and fastpath generate_ar?"""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import time
import mlx.core as mx
import mlx_fastpath as fp
from mlx_fastpath.guard import MemoryGuard
M = MODELS + "/gemma-4-e2b-it-OptiQ-4bit"
PROMPTS = ["Explain in three sentences why the sky is blue.", "What is 17 * 23? Answer with the number only."]
with MemoryGuard(min_free_gb=8.2, max_footprint_gb=7.2) as g:
    eng = fp.load(M, memory_limit_gb=5.2, n2=False)
    tok = eng.tokenizer
    import mlx_lm
    for p in PROMPTS:
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
        print(f"\n### {p}  ({len(ids)} prompt tokens)")
        fp.remove()
        t0 = time.perf_counter()
        txt = mlx_lm.generate(eng._model, tok, prompt=ids, max_tokens=60, verbose=False)
        print(f"[mlx-lm generate]  {time.perf_counter() - t0:.2f}s  {txt.strip()[:220]!r}")
        st = eng.generate("", prompt_ids=ids, max_tokens=60, depth=0)
        print(f"[optiq engine AR]  {st.decode_tok_s:.1f} tok/s  {st.text.strip()[:220]!r}")
        print("apply:", sorted(fp.apply()))
        r = fp.generate_ar(eng, ids, max_tokens=60)
        print(f"[fastpath gen_ar]  {r['tok_s']:.1f} tok/s  {tok.decode(r['tokens']).strip()[:220]!r}")
        fp.remove()
    print("\n[memory]", g.report(), "MLX peak GB", round(mx.get_peak_memory() / 2**30, 2))
