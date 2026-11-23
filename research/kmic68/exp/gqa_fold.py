"""GQA-folded decode/verify attention for Metal (idea from the Kmic-68 P100 fork: fold the query heads
that share one KV head into one block so K/V are read once, split the keys across blocks, then combine).

fold_sdpa(q, k, v, scale): q [1, Hq, qL, D], k/v [1, Hkv, N, D]. Causal over the last qL keys
(row l may see keys n <= N - qL + l), which is what decode (qL=1) and speculative verify need.
"""
import mlx.core as mx

_P1 = r"""
    constexpr int E = D / 32;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint kvh = threadgroup_position_in_grid.y;
    uint nblk = params[2];
    uint blk = threadgroup_position_in_grid.z % nblk;
    uint chunk = threadgroup_position_in_grid.z / nblk;
    int R = params[3];
    int N = params[0];
    float scale = as_type<float>(params[1]);

    float qr[G][E];
    for (int g = 0; g < G; g++)
        for (int e = 0; e < E; e++)
            qr[g][e] = (chunk * G + g < R) ? float(q[(kvh * R + chunk * G + g) * D + lane * E + e]) * scale : 0.f;
    float m[G], s[G], o[G][E];
    for (int g = 0; g < G; g++) { m[g] = -1e30f; s[g] = 0.f; for (int e = 0; e < E; e++) o[g][e] = 0.f; }

    int start = blk * BLOCK;
    int end = min(start + BLOCK, N);
    // v3: one 16-byte load per lane per key row (D=256 bf16 -> 32 lanes x 8), U keys in flight
    for (int n0 = start + sg * U; n0 < end; n0 += NSG * U) {
        float kr[U][E], vr[U][E];
        for (int u = 0; u < U; u++) {
            int n = min(n0 + u, end - 1);
            uint4 kk = ((const device uint4*)(k + ((size_t)kvh * N + n) * D))[lane];
            uint4 vv = ((const device uint4*)(v + ((size_t)kvh * N + n) * D))[lane];
            uint ka[4] = {kk.x, kk.y, kk.z, kk.w}, va[4] = {vv.x, vv.y, vv.z, vv.w};
            for (int j = 0; j < 4; j++) {
                kr[u][2*j] = as_type<float>(ka[j] << 16); kr[u][2*j+1] = as_type<float>(ka[j] & 0xffff0000u);
                vr[u][2*j] = as_type<float>(va[j] << 16); vr[u][2*j+1] = as_type<float>(va[j] & 0xffff0000u);
            }
        }
        for (int u = 0; u < U; u++) {
            int n = n0 + u;
            if (n >= end) break;
            for (int g = 0; g < G; g++) {
                float d = 0.f;
                for (int e = 0; e < E; e++) d += qr[g][e] * kr[u][e];
                d = simd_sum(d);
                int l = (chunk * G + g) % QL;
                if (n <= N - QL + l) {
                    float nm = max(m[g], d);
                    float f = fast::exp(m[g] - nm);
                    float p = fast::exp(d - nm);
                    s[g] = s[g] * f + p;
                    for (int e = 0; e < E; e++) o[g][e] = o[g][e] * f + p * vr[u][e];
                    m[g] = nm;
                }
            }
        }
    }

    threadgroup float tm[NSG];
    threadgroup float ts[NSG];
    threadgroup float buf[NSG * D];
    uint t = sg * 32 + lane;
    for (int g = 0; g < G; g++) {
        if (chunk * G + g >= R) break;
        if (lane == 0) { tm[sg] = m[g]; ts[sg] = s[g]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float M = -1e30f;
        for (int i = 0; i < NSG; i++) M = max(M, tm[i]);
        float f = fast::exp(m[g] - M);
        for (int e = 0; e < E; e++) buf[sg * D + lane * E + e] = o[g][e] * f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        size_t row = (size_t)(kvh * R + chunk * G + g) * nblk + blk;
        for (uint d = t; d < D; d += NSG * 32) {
            float acc = 0.f;
            for (int i = 0; i < NSG; i++) acc += buf[i * D + d];
            po[row * D + d] = acc;
        }
        if (t == 0) {
            float S = 0.f;
            for (int i = 0; i < NSG; i++) S += ts[i] * fast::exp(tm[i] - M);
            pm[row] = M; ps[row] = S;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_P2 = r"""
    uint row = threadgroup_position_in_grid.x;
    uint d = thread_position_in_threadgroup.x;
    int nblk = params[0];
    float M = -1e30f;
    for (int b = 0; b < nblk; b++) M = max(M, pm[row * nblk + b]);
    float S = 0.f, acc = 0.f;
    for (int b = 0; b < nblk; b++) {
        float f = fast::exp(pm[row * nblk + b] - M);
        S += ps[row * nblk + b] * f;
        acc += po[((size_t)row * nblk + b) * D + d] * f;
    }
    out[row * D + d] = T(acc / S);
"""

_k1 = mx.fast.metal_kernel(name="gqa_fold_p1", input_names=["q", "k", "v", "params"],
                           output_names=["po", "pm", "ps"], source=_P1)
_k2 = mx.fast.metal_kernel(name="gqa_fold_p2", input_names=["po", "pm", "ps", "params"],
                           output_names=["out"], source=_P2)


import struct
_f2i = lambda f: struct.unpack('<i', struct.pack('<f', f))[0]


def fold_sdpa(q, k, v, scale, block=64, nsg=4, rg=None, u=4):
    assert q.dtype == mx.bfloat16 and q.shape[-1] == 256, "v3 kernel: bf16, D=256"
    _, Hq, qL, D = q.shape
    _, Hkv, N, _ = k.shape
    R = (Hq // Hkv) * qL               # query rows sharing one KV head
    G = rg or min(R, Hq // Hkv)        # rows per threadgroup (registers bound this)
    nchunk = (R + G - 1) // G
    nblk = (N + block - 1) // block
    rows = Hkv * R
    p = mx.array([N, _f2i(scale), nblk, R], dtype=mx.int32)
    po, pm, ps = _k1(inputs=[q, k, v, p],
                     template=[("T", q.dtype), ("D", D), ("G", G), ("QL", qL), ("BLOCK", block), ("NSG", nsg), ("U", u)],
                     grid=(nsg * 32, Hkv, nblk * nchunk), threadgroup=(nsg * 32, 1, 1),
                     output_shapes=[(rows * nblk, D), (rows * nblk,), (rows * nblk,)],
                     output_dtypes=[mx.float32, mx.float32, mx.float32])
    out = _k2(inputs=[po, pm, ps, mx.array([nblk], dtype=mx.int32)],
              template=[("T", q.dtype), ("D", D)],
              grid=(rows * D, 1, 1), threadgroup=(D, 1, 1),
              output_shapes=[(rows, D)], output_dtypes=[q.dtype])[0]
    return out.reshape(1, Hq, qL, D)
