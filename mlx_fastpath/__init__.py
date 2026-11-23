"""mlx_fastpath: measured, audited speedups for MLX / mlx-lm / mlx-optiq on Apple Silicon, applied at runtime.

    import mlx_fastpath as fp
    engine = fp.load("…/Qwen3.5-9B-OptiQ-4bit")   # load before optiq is imported
    fp.apply()                                    # fold + fused 4-bit KV attention, cheap drafts, fused capture
    stats = fp.generate(engine, prompt_ids, max_tokens=256)

Nothing inside mlx, mlx-lm or optiq is edited; apply() refuses on untested versions unless force=True.
"""
from ._version import __version__
from .ar import generate_ar
from .apply import apply, generate, generate_ngram, recommended, remove
from .guard import MemoryGuard
from .load import load

__all__ = ["__version__", "MemoryGuard", "apply", "generate", "generate_ar", "generate_ngram", "load", "recommended",
           "remove"]
