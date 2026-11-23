"""Load a model for optiq's MTP engine without the extra memory.

Measured on M5 with Qwen3.5-9B-OptiQ-4bit:
  - importing optiq BEFORE mlx_lm.load lifts the load peak from 7.2 GB to 10.9 GB (same 7.6 GB after);
  - OptiqEngine(path) keeps a duplicate model copy; OptiqEngine.from_loaded does not.
n2 (default on, MLX_FASTPATH_N2=0 or n2=False turns it off): load the model's N2-gdnhead copy when one exists next to
it (`mlx-fastpath n2 MODEL` builds it; +14-22% decode, results.md §18.2; see n2.py).
"""
import sys
import warnings

import mlx.core as mx


def load(model_path, *, memory_limit_gb=None, cache_limit_gb=1.0, n2=None):
    if memory_limit_gb:
        mx.set_memory_limit(int(memory_limit_gb * 2**30))
    if cache_limit_gb is not None:
        mx.set_cache_limit(int(cache_limit_gb * 2**30))
    if "optiq" in sys.modules:
        warnings.warn("optiq was imported before loading: expect a ~3.7 GB higher load peak (9B). "
                      "Import mlx_fastpath and load first.", stacklevel=2)
    from .n2 import resolve_model
    model_path = str(resolve_model(model_path, n2))
    import mlx_lm
    model, tokenizer = mlx_lm.load(model_path)
    from optiq.runtime.engine import OptiqEngine
    return OptiqEngine.from_loaded(model, tokenizer, model_path)
