"""GQA fold for multi-token verify attention, in plain MLX ops (no kernel).

All query rows that share one KV head (gqa x qL) become one matmul, so K/V are read once per KV head and
MLX's slow paths for verify widths 6-15 at head dim 256 are avoided. Measured on M5: up to 4.0x faster
verify attention at the Qwen3.6-27B shape, error equal to mx.fast SDPA (research/kmic68 results.md section 2c).
Causal over the last qL keys: query row l may see keys n <= N - qL + l.
"""
import mlx.core as mx


def _causal_rows(g, qL, N):
    l = mx.arange(g * qL) % qL
    return mx.arange(N)[None] <= (N - qL + l)[:, None]


def folded_sdpa(q, k, v, scale):
    """q [B,Hq,qL,D]; k/v [B,Hkv,N,D] (bf16/fp16 cache)."""
    B, Hq, qL, D = q.shape
    Hkv, N = k.shape[1], k.shape[2]
    g = Hq // Hkv
    s = (q * scale).reshape(B, Hkv, g * qL, D) @ k.swapaxes(-1, -2)
    if qL > 1:
        s = mx.where(_causal_rows(g, qL, N), s, mx.array(-mx.inf, s.dtype))
    p = mx.softmax(s, axis=-1, precise=True)
    return (p @ v).reshape(B, Hq, qL, D)


def folded_qsdpa(q, qk, qv, scale, bits, group_size=64):
    """Same fold on an mx.quantize'd cache: qk/qv = (packed, scales, biases), [B,Hkv,N,...]."""
    B, Hq, qL, D = q.shape
    Hkv, N = qk[0].shape[1], qk[0].shape[2]
    g = Hq // Hkv
    s = mx.quantized_matmul((q * scale).reshape(B, Hkv, g * qL, D), *qk, transpose=True,
                            group_size=group_size, bits=bits)
    if qL > 1:
        s = mx.where(_causal_rows(g, qL, N), s, mx.array(-mx.inf, s.dtype))
    p = mx.softmax(s, axis=-1, precise=True)
    return mx.quantized_matmul(p, *qv, transpose=False, group_size=group_size,
                               bits=bits).reshape(B, Hq, qL, D)
