"""Attention dispatch for verify/decode, swapped in at runtime into mlx-lm model modules.

Rules (measured on M5, Qwen3.5-9B / Qwen3.6-27B shapes):
  bf16/fp16 cache : GQA fold when qL > 1 and gqa*qL >= 16, else stock mx.fast SDPA.
  quantized cache : fused kernel for qL <= 3, folded quantized matmuls for qL 4..16 (head dim 256, group 64).
Anything else (array masks, sinks, batch > 1, prefill widths > 16) goes to the stock path.
"""
import importlib

from mlx_lm.models import base
from mlx_lm.models.cache import QuantizedKVCache

from .folded import folded_qsdpa, folded_sdpa
from .qkv_kernel import fused_qkv_sdpa

# Model modules verified with these rules. Others keep stock attention.
MODULES = ("mlx_lm.models.qwen3_next",)
FOLD_MIN_ROWS = 16
MAX_VERIFY = 16

_stock = base.scaled_dot_product_attention


def scaled_dot_product_attention(queries, keys, values, cache, scale, mask, sinks=None):
    B, Hq, qL, D = queries.shape
    plain_mask = mask is None if qL == 1 else (isinstance(mask, str) and mask == "causal")
    if B == 1 and sinks is None and plain_mask and qL <= MAX_VERIFY:
        if isinstance(cache, QuantizedKVCache):
            if D == 256 and cache.group_size == 64:
                if qL <= 3:
                    return fused_qkv_sdpa(queries, cache.keys, cache.values, scale, cache.bits,
                                          n_valid=cache.offset)
                return folded_qsdpa(queries, keys, values, scale, cache.bits)
        else:
            g = Hq // keys.shape[1]
            if qL > 1 and g * qL >= FOLD_MIN_ROWS:
                return folded_sdpa(queries, keys, values, scale)
    return _stock(queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks)


def enable(modules=MODULES):
    for name in modules:
        importlib.import_module(name).scaled_dot_product_attention = scaled_dot_product_attention


def disable(modules=MODULES):
    for name in modules:
        importlib.import_module(name).scaled_dot_product_attention = _stock


def enabled(module=MODULES[0]):
    return importlib.import_module(module).scaled_dot_product_attention is scaled_dot_product_attention
