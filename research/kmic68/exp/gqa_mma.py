"""GQA-folded verify/decode attention on simdgroup matrices (8x8 MMA).
Idea from the Kmic-68 fork: all query rows that share a KV head (gqa x qL) are one matrix, so K/V are read once
and the tile is filled by real rows rather than per-head padding. Split-K over key blocks + a combine pass.

mma_sdpa(q, k, v, scale, n_valid=None): q [1,Hq,qL,256] bf16; k/v [1,Hkv,Nalloc,256] bf16 (may be a cache
buffer longer than n_valid; Nalloc % 32 == 0). Causal over the last qL keys of the valid range.
"""
import struct
import mlx.core as mx

_f2i = lambda f: struct.unpack('<i', struct.pack('<f', f))[0]

_P1 = r"""
    constexpr int D = 256, DT = D / 8, C = 32, CT = C / 8;
    uint lane = thread_index_in_simdgroup;
    uint rt = simdgroup_index_in_threadgroup;          // this simdgroup's 8-row tile
    uint kvh = threadgroup_position_in_grid.y;
    uint blk = threadgroup_position_in_grid.z;
    int N = params[0];
    float scale = as_type<float>(params[1]);
    int nblk = params[2], R = params[3], RP = params[4], Nalloc = params[5];

    const device bfloat* qb = (const device bfloat*)q + ((size_t)kvh * RP + rt * 8) * D;
    const device bfloat* kb = (const device bfloat*)k + (size_t)kvh * Nalloc * D;
    const device bfloat* vb = (const device bfloat*)v + (size_t)kvh * Nalloc * D;

    simdgroup_bfloat8x8 Q[DT];
    for (int dt = 0; dt < DT; dt++) simdgroup_load(Q[dt], qb + dt * 8, D);
    simdgroup_float8x8 O[DT];
    for (int dt = 0; dt < DT; dt++) O[dt] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

    threadgroup float sc[NSG][8 * C];
    threadgroup bfloat pb[NSG][8 * C];
    threadgroup float dg[NSG][64];
    for (uint i = lane; i < 64; i += 32) dg[rt][i] = 0.f;
    float m = -1e30f, s = 0.f;                           // lanes 0..7: row rt*8+lane
    int row = rt * 8 + (lane & 7);
    int lim = N - QL + (row % QL);                       // last visible key for this row

    int start = blk * BLOCK, end = min(start + BLOCK, N);
    for (int n0 = start; n0 < end; n0 += C) {
        simdgroup_float8x8 S[CT];
        for (int t = 0; t < CT; t++) S[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        for (int dt = 0; dt < DT; dt++) {
            for (int t = 0; t < CT; t++) {
                simdgroup_bfloat8x8 Kt;
                simdgroup_load(Kt, kb + (size_t)(n0 + t * 8) * D + dt * 8, D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[t], Q[dt], Kt, S[t]);
            }
        }
        for (int t = 0; t < CT; t++) simdgroup_store(S[t], &sc[rt][t * 8], C);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 8) {
            float cm = m;
            for (int j = 0; j < C; j++) {
                int n = n0 + j;
                if (n < end && n <= lim) cm = max(cm, sc[rt][lane * C + j] * scale);
            }
            float f = fast::exp(m - cm), sum = 0.f;
            for (int j = 0; j < C; j++) {
                int n = n0 + j;
                float p = (n < end && n <= lim) ? fast::exp(sc[rt][lane * C + j] * scale - cm) : 0.f;
                sum += p;
                pb[rt][lane * C + j] = bfloat(p);
            }
            s = s * f + sum; m = cm;
            dg[rt][lane * 9] = f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 F;
        simdgroup_load(F, &dg[rt][0], 8);
        for (int dt = 0; dt < DT; dt++) simdgroup_multiply(O[dt], F, O[dt]);
        for (int t = 0; t < CT; t++) {
            simdgroup_bfloat8x8 P;
            simdgroup_load(P, &pb[rt][t * 8], C);
            for (int dt = 0; dt < DT; dt++) {
                simdgroup_bfloat8x8 Vt;
                simdgroup_load(Vt, vb + (size_t)(n0 + t * 8) * D + dt * 8, D);
                simdgroup_multiply_accumulate(O[dt], P, Vt, O[dt]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    size_t prow = (size_t)kvh * RP + rt * 8;
    for (int dt = 0; dt < DT; dt++)
        simdgroup_store(O[dt], po + (prow * nblk + blk) * D + dt * 8, (ulong)nblk * D);
    if (lane < 8) { pm[(prow + lane) * nblk + blk] = m; ps[(prow + lane) * nblk + blk] = s; }
"""

_P2 = r"""
    uint orow = threadgroup_position_in_grid.x;
    uint d = thread_position_in_threadgroup.x;
    int nblk = params[0], R = params[1], RP = params[2];
    size_t row = (size_t)(orow / R) * RP + orow % R;
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

_k1 = mx.fast.metal_kernel(name="gqa_mma_p1", input_names=["q", "k", "v", "params"],
                           output_names=["po", "pm", "ps"], source=_P1, ensure_row_contiguous=False)
_k2 = mx.fast.metal_kernel(name="gqa_mma_p2", input_names=["po", "pm", "ps", "params"],
                           output_names=["out"], source=_P2)


def mma_sdpa(q, k, v, scale, n_valid=None, block=256):
    assert q.dtype == mx.bfloat16 and q.shape[-1] == 256 and k.dtype == mx.bfloat16
    _, Hq, qL, D = q.shape
    _, Hkv, Nalloc, _ = k.shape
    assert Nalloc % 32 == 0 and block % 32 == 0
    N = Nalloc if n_valid is None else n_valid
    R = (Hq // Hkv) * qL
    RT = (R + 7) // 8
    RP = RT * 8
    qp = q.reshape(Hkv, R, D)
    if RP != R:
        qp = mx.pad(qp, [(0, 0), (0, RP - R), (0, 0)])
    qp = mx.contiguous(qp)
    nblk = (N + block - 1) // block
    p = mx.array([N, _f2i(scale), nblk, R, RP, Nalloc], dtype=mx.int32)
    po, pm, ps = _k1(inputs=[qp, k, v, p],
                     template=[("QL", qL), ("BLOCK", block), ("NSG", RT)],
                     grid=(RT * 32, Hkv, nblk), threadgroup=(RT * 32, 1, 1),
                     output_shapes=[(Hkv * RP * nblk, D), (Hkv * RP * nblk,), (Hkv * RP * nblk,)],
                     output_dtypes=[mx.float32] * 3)
    out = _k2(inputs=[po, pm, ps, mx.array([nblk, R, RP], dtype=mx.int32)],
              grid=(Hkv * R * D, 1, 1), threadgroup=(D, 1, 1),
              output_shapes=[(Hkv * R, D)], output_dtypes=[mx.bfloat16])[0]
    return out.reshape(1, Hq, qL, D)
