"""Gemma 4 E2B (OptiQ) + its MTP assistant drafter through mlx-vlm: does it load, is greedy output the same as without
the drafter, and what acceptance / speed does it show? One process, MemoryGuard."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import time
import mlx.core as mx
from mlx_fastpath.guard import MemoryGuard
T = MODELS + "/gemma-4-e2b-it-OptiQ-4bit"
D = MODELS + "/gemma-4-E2B-it-assistant-bf16"
PROMPTS = ["Explain in five sentences why the sky is blue.",
           "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
           "A train leaves at 9:40 and arrives at 13:15. How long is the trip? Show the steps."]
with MemoryGuard(min_free_gb=8.2, max_footprint_gb=7.2) as g:
    mx.set_memory_limit(int(5.2 * 2**30)); mx.set_cache_limit(2**30)
    from mlx_vlm.utils import load
    from mlx_vlm.generate import generate
    from mlx_vlm.speculative.drafters import load_drafter
    try:
        model, processor = load(T, strict=False)   # text only: the OptiQ file holds no audio/vision towers
    except Exception as e:
        print("TARGET LOAD FAILED:", type(e).__name__, str(e)[:500]); raise SystemExit(2)
    print("target loaded:", type(model).__name__)
    drafter, kind = load_drafter(D, kind="mtp")
    print("drafter loaded:", type(drafter).__name__, "kind", kind)
    try:
        from mlx_vlm.speculative.drafters import validate_drafter_compatibility
        validate_drafter_compatibility(model, drafter, kind); print("compatibility: OK")
    except ImportError:
        print("compatibility: no validator exported")
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    for p in PROMPTS:
        prompt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        r0 = generate(model, processor, prompt, max_tokens=160, temperature=0.0)
        r1 = generate(model, processor, prompt, max_tokens=160, temperature=0.0, draft_model=drafter, draft_kind=kind,
                      draft_block_size=6)
        same = r0.text == r1.text
        split = next((i for i, (a, b) in enumerate(zip(r0.text, r1.text)) if a != b), None)
        print(f"\n### {p[:40]}")
        print(f"  plain   {r0.generation_tps:6.1f} tok/s, {r0.generation_tokens} tokens")
        print(f"  +assist {r1.generation_tps:6.1f} tok/s, {r1.generation_tokens} tokens | same text: {same}"
              + ("" if same else f" (first differing char {split})"))
        print("  stats:", {k: getattr(r1, k) for k in dir(r1) if any(s in k for s in ("accept", "draft", "spec"))})
        print("  text:", repr(r1.text[:160]))
    print("\n[memory]", g.report(), "MLX peak GB", round(mx.get_peak_memory() / 2**30, 2))
