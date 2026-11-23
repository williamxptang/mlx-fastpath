"""Fused decode/verify attention straight on an MLX-quantized KV cache (affine, group 64, 4 or 8 bit, D=256).
Fork idea (E6): attend on the compressed cache, dequantize in registers; GQA rows folded so packed K/V are read once.
Layout = mx.quantize / mlx-lm QuantizedKVCache: packed uint32 [1,Hkv,Nalloc,D*bits/32], scales/biases [1,Hkv,Nalloc,D/64].
"""
import struct
import mlx.core as mx

_f2i = lambda f: struct.unpack('<i', struct.pack('<f', f))[0]

_P1 = r"""
    constexpr int D = 256, E = 8, WPL = E * BITS / 32, WROW = D * BITS / 32, GPR = D / 64;
    constexpr uint MASK = (1u << BITS) - 1u;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint kvh = threadgroup_position_in_grid.y;
    int N = params[0];
    float scale = as_type<float>(params[1]);
    int nblk = params[2], R = params[3], Nalloc = params[4];
    uint blk = threadgroup_position_in_grid.z % nblk;
    uint chunk = threadgroup_position_in_grid.z / nblk;

    float qr[G][E], qs[G];
    for (int g = 0; g < G; g++) {
        int r = chunk * G + g;
        qs[g] = 0.f;
        for (int e = 0; e < E; e++) {
            qr[g][e] = (r < R) ? float(q[((size_t)kvh * R + r) * D + lane * E + e]) * scale : 0.f;
            qs[g] += qr[g][e];
        }
    }
    float m[G], s[G], o[G][E];
    for (int g = 0; g < G; g++) { m[g] = -1e30f; s[g] = 0.f; for (int e = 0; e < E; e++) o[g][e] = 0.f; }

    size_t hrow = (size_t)kvh * Nalloc;
    int start = blk * BLOCK, end = min(start + BLOCK, N);
    for (int n0 = start + sg * U; n0 < end; n0 += NSG * U) {
        float kw[U][E], vw[U][E], ks[U], kb[U], vs[U], vb[U];
        for (int u = 0; u < U; u++) {
            int n = min(n0 + u, end - 1);
            size_t row = hrow + n;
            uint kk[WPL], vv[WPL];
            for (int j = 0; j < WPL; j++) { kk[j] = kp[row * WROW + lane * WPL + j]; vv[j] = vp[row * WROW + lane * WPL + j]; }
            for (int e = 0; e < E; e++) {
                int bit = e * BITS;
                kw[u][e] = float((kk[bit / 32] >> (bit % 32)) & MASK);
                vw[u][e] = float((vv[bit / 32] >> (bit % 32)) & MASK);
            }
            uint gi = row * GPR + (lane * E) / 64;
            ks[u] = float(k_s[gi]); kb[u] = float(k_b[gi]); vs[u] = float(v_s[gi]); vb[u] = float(v_b[gi]);
        }
        for (int u = 0; u < U; u++) {
            int n = n0 + u;
            if (n >= end) break;
            for (int g = 0; g < G; g++) {
                float d = 0.f;
                for (int e = 0; e < E; e++) d += qr[g][e] * kw[u][e];
                d = simd_sum(ks[u] * d + kb[u] * qs[g]);
                int l = (chunk * G + g) % QL;
                if (n <= N - QL + l) {
                    float nm = max(m[g], d);
                    float f = fast::exp(m[g] - nm);
                    float p = fast::exp(d - nm);
                    s[g] = s[g] * f + p;
                    for (int e = 0; e < E; e++) o[g][e] = o[g][e] * f + p * (vs[u] * vw[u][e] + vb[u]);
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
        size_t prow = ((size_t)kvh * R + chunk * G + g) * nblk + blk;
        for (uint d = t; d < D; d += NSG * 32) {
            float acc = 0.f;
            for (int i = 0; i < NSG; i++) acc += buf[i * D + d];
            po[prow * D + d] = acc;
        }
        if (t == 0) {
            float S = 0.f;
            for (int i = 0; i < NSG; i++) S += ts[i] * fast::exp(tm[i] - M);
            pm[prow] = M; ps[prow] = S;
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
        acc += po[((size_t)row * nblk + b) * 256 + d] * f;
    }
    out[row * 256 + d] = T(acc / S);
"""

_k1 = mx.fast.metal_kernel(name="qkv_attn_p1", input_names=["q", "kp", "k_s", "k_b", "vp", "v_s", "v_b", "params"],
                           output_names=["po", "pm", "ps"], source=_P1)
_k2 = mx.fast.metal_kernel(name="qkv_attn_p2", input_names=["po", "pm", "ps", "params"],
                           output_names=["out"], source=_P2)


def qkv_sdpa(q, qk, qv, scale, bits=4, n_valid=None, block=256, nsg=2, u=4, rg=None):
    """q [1,Hq,qL,256]; qk/qv = (packed, scales, biases) as mx.quantize(group_size=64) of [1,Hkv,Nalloc,256]."""
    _, Hq, qL, D = q.shape
    assert D == 256 and bits in (4, 8)
    Hkv, Nalloc = qk[0].shape[1], qk[0].shape[2]
    N = Nalloc if n_valid is None else n_valid
    R = (Hq // Hkv) * qL
    G = rg or min(R, Hq // Hkv)
    nchunk = (R + G - 1) // G
    nblk = (N + block - 1) // block
    p = mx.array([N, _f2i(scale), nblk, R, Nalloc], dtype=mx.int32)
    po, pm, ps = _k1(inputs=[q, qk[0], qk[1], qk[2], qv[0], qv[1], qv[2], p],
                     template=[("T", q.dtype), ("BITS", bits), ("G", G), ("QL", qL), ("BLOCK", block),
                               ("NSG", nsg), ("U", u)],
                     grid=(nsg * 32, Hkv, nblk * nchunk), threadgroup=(nsg * 32, 1, 1),
                     output_shapes=[(Hkv * R * nblk, D), (Hkv * R * nblk,), (Hkv * R * nblk,)],
                     output_dtypes=[mx.float32] * 3)
    out = _k2(inputs=[po, pm, ps, mx.array([nblk], dtype=mx.int32)], template=[("T", q.dtype)],
              grid=(Hkv * R * D, 1, 1), threadgroup=(D, 1, 1),
              output_shapes=[(Hkv * R, D)], output_dtypes=[q.dtype])[0]
    return out.reshape(1, Hq, qL, D)


# ---- v2: lane-per-key scores (no per-key simd_sum, one exp per key per row), lane-per-dims V ----
_P1B = r"""
    constexpr int D = 256, E = 8, WROW = D * BITS / 32, GPR = D / 64, WPG = 64 * BITS / 32, VPW = 32 / BITS;
    constexpr uint MASK = (1u << BITS) - 1u;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint kvh = threadgroup_position_in_grid.y;
    int N = params[0];
    float scale = as_type<float>(params[1]);
    int nblk = params[2], R = params[3], Nalloc = params[4];
    uint blk = threadgroup_position_in_grid.z % nblk;
    uint chunk = threadgroup_position_in_grid.z / nblk;
    uint t = sg * 32 + lane;

    threadgroup float qt[G * D];
    threadgroup float qsg[G * GPR];
    threadgroup float pt[NSG][G][32];
    for (uint i = t; i < G * D; i += NSG * 32) {
        int g = i / D, r = chunk * G + g;
        qt[i] = (r < R) ? float(q[((size_t)kvh * R + r) * D + i % D]) * scale : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < G * GPR; i += NSG * 32) {
        float a = 0.f;
        for (int j = 0; j < 64; j++) a += qt[(i / GPR) * D + (i % GPR) * 64 + j];
        qsg[i] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m[G], s[G], o[G][E];
    for (int g = 0; g < G; g++) { m[g] = -1e30f; s[g] = 0.f; for (int e = 0; e < E; e++) o[g][e] = 0.f; }
    size_t hrow = (size_t)kvh * Nalloc;
    int start = blk * BLOCK, end = min(start + BLOCK, N);
    for (int n0 = start + sg * 32; n0 < end; n0 += NSG * 32) {
        int n = n0 + lane;
        bool live = n < end;
        size_t row = hrow + min(n, end - 1);
        float sc[G];
        for (int g = 0; g < G; g++) sc[g] = 0.f;
        for (int gp = 0; gp < GPR; gp++) {
            float acc[G];
            for (int g = 0; g < G; g++) acc[g] = 0.f;
            for (int w = 0; w < WPG; w++) {
                uint word = kp[row * WROW + gp * WPG + w];
                for (int i = 0; i < VPW; i++) {
                    float x = float((word >> (i * BITS)) & MASK);
                    int d = gp * 64 + w * VPW + i;
                    for (int g = 0; g < G; g++) acc[g] += qt[g * D + d] * x;
                }
            }
            float ks = float(k_s[row * GPR + gp]), kb = float(k_b[row * GPR + gp]);
            for (int g = 0; g < G; g++) sc[g] += ks * acc[g] + kb * qsg[g * GPR + gp];
        }
        for (int g = 0; g < G; g++) {
            int l = (chunk * G + g) % QL;
            bool ok = live && (n <= N - QL + l);
            float v = ok ? sc[g] : -1e30f;
            float tmax = simd_max(v);
            float nm = max(m[g], tmax);
            float f = fast::exp(m[g] - nm);
            float p = ok ? fast::exp(v - nm) : 0.f;
            s[g] = s[g] * f + simd_sum(p);
            m[g] = nm;
            for (int e = 0; e < E; e++) o[g][e] *= f;
            pt[sg][g][lane] = p;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        int cnt = min(32, end - n0);
        for (int j = 0; j < cnt; j++) {
            size_t vrow = hrow + n0 + j;
            uint gi = vrow * GPR + (lane * E) / 64;
            float vs = float(v_s[gi]), vb = float(v_b[gi]);
            float vv[E];
            for (int e = 0; e < E; e++) {
                int bit = (lane * E + e) * BITS;
                vv[e] = vs * float((vp[vrow * WROW + bit / 32] >> (bit % 32)) & MASK) + vb;
            }
            for (int g = 0; g < G; g++) {
                float pj = pt[sg][g][j];
                for (int e = 0; e < E; e++) o[g][e] += pj * vv[e];
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float tm[NSG];
    threadgroup float ts[NSG];
    threadgroup float buf[NSG * D];
    for (int g = 0; g < G; g++) {
        if (chunk * G + g >= R) break;
        if (lane == 0) { tm[sg] = m[g]; ts[sg] = s[g]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float M = -1e30f;
        for (int i = 0; i < NSG; i++) M = max(M, tm[i]);
        float f = fast::exp(m[g] - M);
        for (int e = 0; e < E; e++) buf[sg * D + lane * E + e] = o[g][e] * f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        size_t prow = ((size_t)kvh * R + chunk * G + g) * nblk + blk;
        for (uint d = t; d < D; d += NSG * 32) {
            float acc = 0.f;
            for (int i = 0; i < NSG; i++) acc += buf[i * D + d];
            po[prow * D + d] = acc;
        }
        if (t == 0) {
            float S = 0.f;
            for (int i = 0; i < NSG; i++) S += ts[i] * fast::exp(tm[i] - M);
            pm[prow] = M; ps[prow] = S;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
_k1b = mx.fast.metal_kernel(name="qkv_attn_p1b", input_names=["q", "kp", "k_s", "k_b", "vp", "v_s", "v_b", "params"],
                            output_names=["po", "pm", "ps"], source=_P1B)


def qkv_sdpa2(q, qk, qv, scale, bits=4, n_valid=None, block=256, nsg=4, rg=None, **_):
    _, Hq, qL, D = q.shape
    assert D == 256 and bits in (4, 8)
    Hkv, Nalloc = qk[0].shape[1], qk[0].shape[2]
    N = Nalloc if n_valid is None else n_valid
    R = (Hq // Hkv) * qL
    G = rg or min(R, Hq // Hkv)
    nchunk = (R + G - 1) // G
    nblk = (N + block - 1) // block
    p = mx.array([N, _f2i(scale), nblk, R, Nalloc], dtype=mx.int32)
    po, pm, ps = _k1b(inputs=[q, qk[0], qk[1], qk[2], qv[0], qv[1], qv[2], p],
                      template=[("T", q.dtype), ("BITS", bits), ("G", G), ("QL", qL), ("BLOCK", block), ("NSG", nsg)],
                      grid=(nsg * 32, Hkv, nblk * nchunk), threadgroup=(nsg * 32, 1, 1),
                      output_shapes=[(Hkv * R * nblk, D), (Hkv * R * nblk,), (Hkv * R * nblk,)],
                      output_dtypes=[mx.float32] * 3)
    out = _k2(inputs=[po, pm, ps, mx.array([nblk], dtype=mx.int32)], template=[("T", q.dtype)],
              grid=(Hkv * R * D, 1, 1), threadgroup=(D, 1, 1),
              output_shapes=[(Hkv * R, D)], output_dtypes=[q.dtype])[0]
    return out.reshape(1, Hq, qL, D)
