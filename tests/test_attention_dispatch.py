"""The swapped-in attention gives the same result as mlx-lm's stock attention on real cache objects."""
import mlx.core as mx
import pytest
from mlx_lm.models import base
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from conftest import max_err
from mlx_fastpath import attention

TOL = 5e-3


def _fill(cache, Hkv, N, rng):
    k = rng.normal((1, Hkv, N, 256)).astype(mx.bfloat16)
    v = rng.normal((1, Hkv, N, 256)).astype(mx.bfloat16)
    return cache.update_and_fetch(k, v)


@pytest.mark.parametrize("cache_kind", ["bf16", "q4", "q8"])
@pytest.mark.parametrize("Hq,Hkv,prefix,qL", [(24, 4, 900, 1), (24, 4, 900, 3), (24, 4, 900, 4),
                                              (16, 4, 700, 8), (16, 4, 700, 16)])
def test_swapped_attention_equals_stock(rng, cache_kind, Hq, Hkv, prefix, qL):
    cache = KVCache() if cache_kind == "bf16" else QuantizedKVCache(group_size=64, bits=int(cache_kind[1]))
    _fill(cache, Hkv, prefix, rng)
    keys, values = _fill(cache, Hkv, qL, rng)  # the verify tokens' own K/V, as a model step does
    q = rng.normal((1, Hq, qL, 256)).astype(mx.bfloat16)
    mask = "causal" if qL > 1 else None
    # mlx-lm's quantized path scales its queries argument in place: give it its own copy
    stock = base.scaled_dot_product_attention(mx.array(q), keys, values, cache, scale=1 / 16, mask=mask)
    fast = attention.scaled_dot_product_attention(q, keys, values, cache, scale=1 / 16, mask=mask)
    assert max_err(fast, stock) < TOL


def test_enable_and_disable_swap_the_model_module_name():
    import mlx_lm.models.qwen3_next as qn
    try:
        attention.enable()
        assert qn.scaled_dot_product_attention is attention.scaled_dot_product_attention
    finally:
        attention.disable()
    assert qn.scaled_dot_product_attention is base.scaled_dot_product_attention
