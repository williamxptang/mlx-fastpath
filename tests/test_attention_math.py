"""Folded and fused attention compute the same attention as an fp32 reference, at bf16 accuracy."""
import mlx.core as mx
import pytest

from conftest import max_err, reference_attention
from mlx_fastpath.folded import folded_qsdpa, folded_sdpa
from mlx_fastpath.qkv_kernel import fused_qkv_sdpa

# Tolerance: bf16 output rounding. mx.fast SDPA's own error on these inputs is <= ~3e-3.
TOL = 5e-3
SHAPES = [  # Hq, Hkv, N, qL  (Qwen3.6-27B and Qwen3.5-9B attention shapes; verify widths 1..16)
    (24, 4, 600, 1), (24, 4, 600, 3), (24, 4, 1000, 8), (16, 4, 777, 4), (16, 4, 512, 16), (16, 8, 300, 2),
]


def _qkv(rng, Hq, Hkv, N, qL):
    q = rng.normal((1, Hq, qL, 256)).astype(mx.bfloat16)
    k = rng.normal((1, Hkv, N, 256)).astype(mx.bfloat16)
    v = rng.normal((1, Hkv, N, 256)).astype(mx.bfloat16)
    return q, k, v


@pytest.mark.parametrize("Hq,Hkv,N,qL", SHAPES)
def test_fold_matches_reference_on_bf16_cache(rng, Hq, Hkv, N, qL):
    q, k, v = _qkv(rng, Hq, Hkv, N, qL)
    assert max_err(folded_sdpa(q, k, v, 1 / 16), reference_attention(q, k, v, 1 / 16)) < TOL


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("Hq,Hkv,N,qL", SHAPES)
def test_quantized_paths_match_reference_on_dequantized_cache(rng, bits, Hq, Hkv, N, qL):
    q, k, v = _qkv(rng, Hq, Hkv, N, qL)
    qk, qv = mx.quantize(k, 64, bits), mx.quantize(v, 64, bits)
    ref = reference_attention(q, mx.dequantize(*qk, 64, bits), mx.dequantize(*qv, 64, bits), 1 / 16)
    assert max_err(folded_qsdpa(q, qk, qv, 1 / 16, bits), ref) < TOL
    assert max_err(fused_qkv_sdpa(q, qk, qv, 1 / 16, bits), ref) < TOL


def test_fused_kernel_ignores_cache_rows_past_valid_length(rng):
    """A real cache buffer is longer than its valid length; rows past it must not leak into the result."""
    q, k, v = _qkv(rng, 24, 4, 1024, 2)
    qk, qv = mx.quantize(k, 64, 4), mx.quantize(v, 64, 4)
    n = 700
    ref = reference_attention(q, mx.dequantize(*qk, 64, 4)[:, :, :n], mx.dequantize(*qv, 64, 4)[:, :, :n], 1 / 16)
    assert max_err(fused_qkv_sdpa(q, qk, qv, 1 / 16, 4, n_valid=n), ref) < TOL
