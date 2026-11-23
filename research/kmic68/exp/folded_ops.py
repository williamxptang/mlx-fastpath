"""GQA fold in plain MLX ops: queries sharing a KV head become one matmul (M = gqa*qL), K/V read once per KV head."""
import mlx.core as mx

def folded_sdpa(q, k, v, scale, n_valid=None):
    B, Hq, qL, D = q.shape
    Hkv, N = k.shape[1], k.shape[2] if n_valid is None else n_valid
    if n_valid is not None:
        k, v = k[:, :, :n_valid], v[:, :, :n_valid]
    g = Hq // Hkv
    qf = (q * scale).reshape(B, Hkv, g * qL, D)
    s = qf @ k.swapaxes(-1, -2)                                   # [B, Hkv, g*qL, N]
    if qL > 1:
        l = mx.arange(g * qL) % qL
        allowed = mx.arange(N)[None] <= (N - qL + l)[:, None]
        s = mx.where(allowed, s, mx.array(-mx.inf, s.dtype))
    p = mx.softmax(s, axis=-1, precise=True)
    return (p @ v).reshape(B, Hq, qL, D)


def folded_qsdpa(q, qk, qv, scale, bits, group_size=64):
    """GQA fold on a quantized KV cache: one quantized_matmul per KV head with M = gqa*qL."""
    B, Hq, qL, D = q.shape
    Hkv, N = qk[0].shape[1], qk[0].shape[2]
    g = Hq // Hkv
    qf = (q * scale).reshape(B, Hkv, g * qL, D)
    s = mx.quantized_matmul(qf, *qk, transpose=True, group_size=group_size, bits=bits)
    if qL > 1:
        l = mx.arange(g * qL) % qL
        allowed = mx.arange(N)[None] <= (N - qL + l)[:, None]
        s = mx.where(allowed, s, mx.array(-mx.inf, s.dtype))
    p = mx.softmax(s, axis=-1, precise=True)
    return mx.quantized_matmul(p, *qv, transpose=False, group_size=group_size, bits=bits).reshape(B, Hq, qL, D)
