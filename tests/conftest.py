import mlx.core as mx
import pytest

mx.set_memory_limit(2 * 2**30)  # fast set stays small


def reference_attention(q, k, v, scale):
    """fp32 attention, causal over the last qL keys (row l sees keys n <= N - qL + l)."""
    B, Hq, qL, D = q.shape
    Hkv, N = k.shape[1], k.shape[2]
    g = Hq // Hkv
    k = mx.repeat(k.astype(mx.float32), g, axis=1)
    v = mx.repeat(v.astype(mx.float32), g, axis=1)
    s = (q.astype(mx.float32) * scale) @ k.swapaxes(-1, -2)
    allowed = mx.arange(N)[None] <= (N - qL + mx.arange(qL))[:, None]
    s = mx.where(allowed, s, -mx.inf)
    return mx.softmax(s, axis=-1) @ v


def max_err(a, b):
    return mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item()


@pytest.fixture
def rng():
    mx.random.seed(0)
    return mx.random
