"""Builds the A/B weights that don't exist yet, one per process (argv: int4-q2b | int4-gemma), under MemoryGuard:
plain mlx-lm 4-bit (group 64) copies of Qwen3.5-2B (+ optiq's MTP sidecar from the bf16 source) and Gemma 4 E2B.
N2 copies are built with the package CLI (`mlx-fastpath n2 MODEL`), not here."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import sys

import mlx.core as mx

from mlx_fastpath.guard import MemoryGuard

LLM = MODELS + ""
JOBS = {"int4-q2b": (f"{LLM}/Qwen3.5-2B-bf16", f"{LLM}/Qwen3.5-2B-4bit", True),
        "int4-gemma": (f"{LLM}/gemma-4-E2B-it-bf16", f"{LLM}/gemma-4-e2b-it-4bit", False)}

src, dst, mtp = JOBS[sys.argv[1]]
with MemoryGuard(min_free_gb=float(sys.argv[2]), max_footprint_gb=float(sys.argv[3])) as g:
    mx.set_memory_limit(int((float(sys.argv[3]) - 1) * 2**30))
    mx.set_cache_limit(2**30)
    from mlx_lm import convert
    convert(src, dst, quantize=True, q_bits=4, q_group_size=64)
    if mtp:
        from optiq.runtime.mtp_convert import preserve_mtp
        assert preserve_mtp(src, dst), "no MTP tensors written"
    print("[guard]", g.report(), "MLX peak GB", round(mx.get_peak_memory() / 2**30, 2), flush=True)
