"""Fused attention on an MLX-quantized KV cache (affine, group 64, 4 or 8 bit, head dim 256).

Reads the packed cache once and dequantizes in registers; scores use s*sum(q*w) + b*sum(q).
Scores: one lane per key (no per-key cross-lane sum, one exp per key and row).
Values: each lane owns 8 dims. Split-K over key blocks plus a combine pass.
Measured on M5 (Qwen3.6-27B attention shape, 32k, 4-bit, 1 token): 4.5x faster than mlx-lm's
quantized SDPA and 2.7x faster than bf16 mx.fast SDPA. See research/kmic68 results.md section 5.
Layout = mx.quantize / mlx-lm QuantizedKVCache: packed uint32 [1,Hkv,Nalloc,D*bits/32],
scales/biases [1,Hkv,Nalloc,D/64].
"""
import struct

import mlx.core as mx

_f2i = lambda f: struct.unpack("<i", struct.pack("<f", f))[0]

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

_k2 = mx.fast.metal_kernel(name="qkv_attn_p2", input_names=["po", "pm", "ps", "params"],
                           output_names=["out"], source=_P2)


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


def fused_qkv_sdpa(q, qk, qv, scale, bits=4, n_valid=None, block=None, nsg=4, rg=None):
    """Decode/verify attention read straight from an mx.quantize'd KV cache (group 64, 4/8-bit, head dim 256).
    q [1,Hq,qL,256] bf16; qk/qv = (packed, scales, biases) full cache buffers [1,Hkv,Nalloc,...];
    n_valid = valid length (cache.offset). Causal over the last qL keys."""
    _, Hq, qL, D = q.shape
    assert D == 256 and bits in (4, 8), "fused_qkv_sdpa: head dim 256, 4/8-bit only"
    if block is None:
        block = 512 if bits == 4 else 256  # tuned on M5 (bench_qkv.py sweep)
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
