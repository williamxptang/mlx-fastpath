import mlx.core as mx


def ref(q, k, v, scale):
    _, Hq, qL, D = q.shape
    Hkv, N = k.shape[1], k.shape[2]
    qf = q.astype(mx.float32).reshape(Hkv, Hq // Hkv * qL, D)
    s = (qf * scale) @ k[0].astype(mx.float32).transpose(0, 2, 1)
    l = mx.tile(mx.arange(qL), Hq // Hkv)[:, None]
    s = mx.where(mx.arange(N)[None] <= N - qL + l, s, -mx.inf)
    return (mx.softmax(s, -1) @ v[0].astype(mx.float32)).reshape(1, Hq, qL, D)


