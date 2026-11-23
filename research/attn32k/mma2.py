"""Item 2: GQA-folded verify attention on simdgroup matrices, bf16 cache, head dim 256.
Fixes of the naive version (research/kmic68/exp/gqa_mma.py, 3-4x slower than MLX):
  - K/V chunks (16 keys) are staged in threadgroup memory with coalesced 16-byte loads (all 128 threads);
  - 4 simdgroups per threadgroup split the head dims (64 each): Q slice in registers, partial scores summed in
    threadgroup memory, each simdgroup accumulates its own 64 output dims;
  - online softmax per row on the summed scores; split-K over key blocks + a combine pass.
sdpa(q, k, v, scale, n_valid=None): q [1,Hq,qL,256] bf16; k/v [1,Hkv,Nalloc,256] bf16 (a cache buffer may be longer
than n_valid). Causal over the last qL valid keys."""
import struct

import mlx.core as mx

_f2i = lambda f: struct.unpack("<i", struct.pack("<f", f))[0]

_P1 = r"""
    constexpr int D = 256, C = 16, DS = 64, DTS = 8, NT = 128;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint tid = sg * 32 + lane;
    uint kvh = threadgroup_position_in_grid.y;
    uint blk = threadgroup_position_in_grid.z;
    int N = params[0], nblk = params[1], Nalloc = params[2], R = params[3];

    threadgroup bfloat Ks[C * D];
    threadgroup bfloat Vs[C * D];
    threadgroup float Sp[4][RP * C];
    threadgroup bfloat Ps[RP * C];
    threadgroup float Fd[RT * 64];
    threadgroup float Ml[RP * 2];
    for (uint i = tid; i < RT * 64; i += NT) Fd[i] = 0.f;
    for (uint i = tid; i < RP; i += NT) { Ml[2 * i] = -1e30f; Ml[2 * i + 1] = 0.f; }

    const device bfloat* qb = q + (size_t)kvh * RP * D;
    simdgroup_bfloat8x8 Q[RT][DTS];
    simdgroup_float8x8 O[RT][DTS];
    for (int rt = 0; rt < RT; rt++)
        for (int dt = 0; dt < DTS; dt++) {
            simdgroup_load(Q[rt][dt], qb + (rt * 8) * D + sg * DS + dt * 8, D);
            O[rt][dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        }
    const device uint4* kb4 = (const device uint4*)(k + (size_t)kvh * Nalloc * D);
    const device uint4* vb4 = (const device uint4*)(v + (size_t)kvh * Nalloc * D);
    threadgroup uint4* Ks4 = (threadgroup uint4*)Ks;
    threadgroup uint4* Vs4 = (threadgroup uint4*)Vs;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int start = blk * BLOCK, end = min(start + BLOCK, N);
    for (int n0 = start; n0 < end; n0 += C) {
        for (uint i = tid; i < C * 32; i += NT) {
            int n = min(n0 + int(i / 32), N - 1);
            Ks4[i] = kb4[(size_t)n * 32 + i % 32];
            Vs4[i] = vb4[(size_t)n * 32 + i % 32];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[RT][2];
        for (int rt = 0; rt < RT; rt++) { S[rt][0] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
                                          S[rt][1] = make_filled_simdgroup_matrix<float, 8, 8>(0.f); }
        for (int dt = 0; dt < DTS; dt++)
            for (int t = 0; t < 2; t++) {
                simdgroup_bfloat8x8 Kt;
                simdgroup_load(Kt, Ks + (t * 8) * D + sg * DS + dt * 8, D, ulong2(0, 0), true);
                for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(S[rt][t], Q[rt][dt], Kt, S[rt][t]);
            }
        for (int rt = 0; rt < RT; rt++)
            for (int t = 0; t < 2; t++) simdgroup_store(S[rt][t], &Sp[sg][rt * 8 * C + t * 8], C);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < RP) {
            int r = tid;
            float m = Ml[2 * r], l = Ml[2 * r + 1];
            int lim = N - QL + (r % QL);
            float sc[C];
            float cm = m;
            for (int j = 0; j < C; j++) {
                int n = n0 + j;
                bool ok = r < R && n < end && n <= lim;
                float x = Sp[0][r * C + j] + Sp[1][r * C + j] + Sp[2][r * C + j] + Sp[3][r * C + j];
                sc[j] = ok ? x : -1e30f;
                cm = max(cm, sc[j]);
            }
            float f = fast::exp(m - cm), sum = 0.f;
            for (int j = 0; j < C; j++) {
                float p = sc[j] > -1e29f ? fast::exp(sc[j] - cm) : 0.f;
                sum += p;
                Ps[r * C + j] = bfloat(p);
            }
            Ml[2 * r] = cm; Ml[2 * r + 1] = l * f + sum;
            Fd[(r / 8) * 64 + (r % 8) * 9] = f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_bfloat8x8 P[RT][2];
        for (int rt = 0; rt < RT; rt++) {
            simdgroup_float8x8 F;
            simdgroup_load(F, &Fd[rt * 64], 8);
            for (int dt = 0; dt < DTS; dt++) simdgroup_multiply(O[rt][dt], F, O[rt][dt]);
            for (int t = 0; t < 2; t++) simdgroup_load(P[rt][t], &Ps[rt * 8 * C + t * 8], C);
        }
        for (int t = 0; t < 2; t++)
            for (int dt = 0; dt < DTS; dt++) {
                simdgroup_bfloat8x8 Vt;
                simdgroup_load(Vt, Vs + (t * 8) * D + sg * DS + dt * 8, D);
                for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(O[rt][dt], P[rt][t], Vt, O[rt][dt]);
            }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int rt = 0; rt < RT; rt++)
        for (int dt = 0; dt < DTS; dt++)
            simdgroup_store(O[rt][dt], po + ((size_t)(kvh * RP + rt * 8) * nblk + blk) * D + sg * DS + dt * 8,
                            (ulong)nblk * D);
    if (tid < RP) {
        size_t row = (size_t)(kvh * RP + tid) * nblk + blk;
        pm[row] = Ml[2 * tid]; ps[row] = Ml[2 * tid + 1];
    }
"""

_P2 = r"""
    uint orow = threadgroup_position_in_grid.x;
    uint d = thread_position_in_threadgroup.x;
    int nblk = params[0], R = params[1], RPv = params[2];
    size_t row = (size_t)(orow / R) * RPv + orow % R;
    float M = -1e30f;
    for (int b = 0; b < nblk; b++) M = max(M, pm[row * nblk + b]);
    float S = 0.f, acc = 0.f;
    for (int b = 0; b < nblk; b++) {
        float f = fast::exp(pm[row * nblk + b] - M);
        S += ps[row * nblk + b] * f;
        acc += po[(row * nblk + b) * 256 + d] * f;
    }
    out[orow * 256 + d] = bfloat(acc / S);
"""

_k1 = mx.fast.metal_kernel(name="attn32k_mma2_p1", input_names=["q", "k", "v", "params"],
                           output_names=["po", "pm", "ps"], source=_P1, ensure_row_contiguous=False)
_k2 = mx.fast.metal_kernel(name="attn32k_mma2_p2", input_names=["po", "pm", "ps", "params"],
                           output_names=["out"], source=_P2)


def sdpa(q, k, v, scale, n_valid=None, block=512):
    _, Hq, qL, D = q.shape
    _, Hkv, Nalloc, _ = k.shape
    assert D == 256 and q.dtype == k.dtype == v.dtype == mx.bfloat16 and block % 16 == 0
    N = Nalloc if n_valid is None else n_valid
    R = (Hq // Hkv) * qL
    RT = (R + 7) // 8
    RP = RT * 8
    qp = (q * scale).reshape(Hkv, R, D)
    if RP != R:
        qp = mx.pad(qp, [(0, 0), (0, RP - R), (0, 0)])
    qp = mx.contiguous(qp)
    nblk = (N + block - 1) // block
    po, pm, ps = _k1(inputs=[qp, k, v, mx.array([N, nblk, Nalloc, R], dtype=mx.int32)],
                     template=[("QL", qL), ("BLOCK", block), ("RT", RT), ("RP", RP)],
                     grid=(128, Hkv, nblk), threadgroup=(128, 1, 1),
                     output_shapes=[(Hkv * RP * nblk, D), (Hkv * RP * nblk,), (Hkv * RP * nblk,)],
                     output_dtypes=[mx.float32] * 3)
    out = _k2(inputs=[po, pm, ps, mx.array([nblk, R, RP], dtype=mx.int32)],
              grid=(Hkv * R * D, 1, 1), threadgroup=(D, 1, 1),
              output_shapes=[(Hkv * R, D)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(1, Hq, qL, D)


def hook(q, k, v, cache, scale):
    """Model-side call: read the cache's full buffers (no copy of the sliced view), valid length = cache.offset."""
    return sdpa(q, cache.keys, cache.values, scale, n_valid=cache.offset)
