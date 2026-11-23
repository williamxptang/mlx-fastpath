"""Runtime swap (no installed file edited): route multi-token verify attention through the GQA fold."""
import mlx_lm.models.qwen3_next as qn
import mlx_lm.models.base as base
from folded_ops import folded_sdpa

_orig = base.scaled_dot_product_attention
MIN_ROWS = 16  # fold when gqa * qL >= 16 (measured break-even on M5)
stats = {"fold": 0, "orig": 0}


def patched(queries, keys, values, cache, scale, mask, sinks=None):
    B, Hq, qL, _ = queries.shape
    g = Hq // keys.shape[1]
    plain_mask = mask is None if qL == 1 else (isinstance(mask, str) and mask == "causal")
    if (B == 1 and sinks is None and not hasattr(cache, "bits") and plain_mask and g * qL >= MIN_ROWS):
        stats["fold"] += 1
        return folded_sdpa(queries, keys, values, scale)
    stats["orig"] += 1
    return _orig(queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks)


def enable(on=True):
    qn.scaled_dot_product_attention = patched if on else _orig
