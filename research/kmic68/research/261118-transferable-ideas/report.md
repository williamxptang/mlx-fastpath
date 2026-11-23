# Kmic-68 P100 fork → our MLX stacks: which ideas transfer

scan / wide (9 lanes: 2 fork-reading, Apple GPU, MLX code, practitioners, academic, contrarian, zh, ja) · 1 round + coordinator
verification · as of 2026-11-18 · read-only (no benchmarks run in this research; T5 numbers are from the two
benchmarks run earlier in the session) · checks: `verified.md` (lane notes and log not published)

## Measured results (2026-11-18/19)
**Before/after for Qwen3.5-9B and Gemma-4-12B: [before_after.md](before_after.md).**
See [results.md](results.md). **Qwen3.5-9B-OptiQ-4bit on M5: 17 → 25–31 tok/s** (MTP depth 2 + fused GDN capture + 32k-vocab draft head; quality equal by teacher-forced re-score). Plain decoding is capped at 20.4 tok/s by the 5.6 GB read per token. Headline: the GQA fold in plain MLX ops makes multi-token verify attention 1.6–4.0× faster (exact); on the real Qwen3.5-9B a verify pass at 32k is 1.13–1.28× faster. Hand-written fold kernels lost to MLX at qL 1 (MLX already at ~114 GB/s of 123). Verify widths 1–4 are almost free on M5; 6–12 hit cliffs in both attention and quantized matmul. Fused attention on a 4-bit KV cache: 2.1–4.5× vs mlx-lm's quantized path, 1.3–2.7× vs bf16. Command-buffer limits: no gain on Qwen3-ASR-0.6B.

## TL;DR

The earlier "doesn't transfer" list was wrong. Of the ideas I had dismissed as Pascal/CUDA-specific, **only the
instruction-level tricks fail to transfer: hand-written SASS, the PRMT+XMAD byte trick, and cuBLAS enums.** Every idea
underneath them maps onto something concrete in mlx / mlx-lm / vmlx on the M5. Several are already done there, and
several are open gaps with outside evidence of the gain.

The strongest transfers, each verified against primary sources:
1. **Launch overhead → MLX command-buffer limits.** These are the CUDA-graphs analog: +12–14% decode on a Qwen3.6 MoE on
   an M5 Max from one environment variable, output identical. Our M5 has the tightest defaults (40 ops / 40 MB).
2. **GQA group folding with a ladder that skips 6.** MLX's grouped decode-attention kernel serves GQA 8/12/16 only.
   Qwen3.5/3.6-27B (GQA 6, D 256) misses it, and outside measurements show about 1.5× left on the table.
3. **Attend directly on the compressed KV cache.** Neither mlx-lm nor vmlx/TurboQuant does this. Two 2026 Apple-silicon
   papers, one of them MLX on an M5 Max, conclude it is required for long context. Our M5 measured 30 GB/s for quantized
   versus 93 GB/s for f16.
4. **DP4A emulation → M5 INT8 TensorOps.** This is the int8 idea in Apple form. An existing MLX add-on (Cider) gets
   1.2–1.9× prefill from it. Like on the P100, it buys nothing for decode.
5. **The fork's measurement and accuracy discipline.** It transfers whole and applies to ASR, TTS and SeedVR2 too.

## Verdict per idea

Legend: **A** = already done in our stacks · **B** = transfers as a principle, with a named target · **C** = the
mechanism does not transfer (reason cited). File:line references are in `verified.md` and `lanes/mlx-code.md`.

### The ideas previously dismissed

| Fork idea (fork ref) | Verdict | Evidence and target |
|---|---|---|
| **DP4A emulation** (A:I2,I10,I22,I33; fork att 38/64) | **B** | No scalar packed-int8 dot exists on Apple shader ALUs; this is evidence from absence: dougallj ISA, MSL spec. But the M5 Neural Accelerators do INT8→INT32 at about 2× FP16 matrix rate (Zakharko, T2 ✔). The fork's own lesson, that int8 is irrelevant for memory-bound decode (deleting all dp4a: +0.8%) and decisive for multi-column/prefill (MMQ 4× slower without it), predicts exactly Cider's result: W8A8 on M5 TensorOps gives 1.2–1.9× prefill and no decode gain (✔). Fork caveat that transfers: 8-bit *activations* were its largest decode error source (NMSE 1.3e-4 vs 4.6e-7 for fp16 activations, B:I24). Gate any W8A8 adoption with KLD. Target: prefill-heavy ASR encoder and LLM prefill on M5. |
| **Hand-tuned SASS / register-bank fixes** (B:I35) | **C** for the trick, **B** for the method | Apple exposes only compiler options (fast/relaxed/safe math, fp-contract), not ISA scheduling (T1 ✔ lane). Banks do exist (32 × 4 B on M1/M2; M5 unknown). The transferable part is the fork's ordering: ablate first (no loads / no barrier / no folds). Pipeline structure was worth 10× more than bank fixes, and bank work was the last 1–3%. Source-level loop reordering got the same gain through the compiler, which is the only route on Metal anyway. |
| **cuBLAS algorithm choice for accuracy** (B:I23) | **A** | MLX steel GEMM, steel attention, NAX attention, sdpa_vector and qmv all accumulate in fp32 for half inputs (mlx-code F4/F7/F8 T1). The P100's long fp16 accumulation chain does not exist here. The method still transfers: characterise kernels against an fp64 reference at your real shapes. MLX NAX split-K bf16 produced garbage on M5 Max under JIT until PR #3810 (contrarian F8). |
| **fp16-multiply / fp32-fold** (A:I31,I33,I47; B:I23) | **B (narrow)** | No speed reason on the NAX path: fp16 accumulators run at the same speed as fp32 there (Zakharko ✔). It matters where people drop to fp16 accumulators to save registers. In mlx#3837 a hand-written decode kernel spilled with fp32 accumulators (4.84 ms) and was fixed with fp16 accumulators (1.47 ms). That trades away exactly the accuracy the fork measured: per-tile fp32 folding was 8.7× more accurate at depth for 2.4%. Target: any custom Metal attention/matvec kernel we write (E2, E3). Keep fp16 chains short and fold into fp32 per tile. |
| **Tensor-parallel exchange overlap** (B:I36,I37,I50) | **C** on one M5, **B** elsewhere | Single GPU, unified memory: no exchange to hide. The principles apply where we do move data. (1) SSD/RAM weight streaming for MoE (vmlx `flash_moe_config.py`; the fork's expert-LRU + staged-fill + "assemble layer l+1 while l computes", B:I42/I49). (2) Multi-Mac MLX distributed: ring 210 µs vs JACCL 21 µs (contrarian F10). (3) "Reduce on the smallest tensor" (B:I50). |
| **Power-cap / energy-per-flop** (B:I38,I39; A:I66,I67) | **B (method)**, unmeasured | No T1–T2 data on M-series GPU power caps or energy per op (gap). Fanless-throttling reports are T4 only. The fork's method needs no CUDA: read clocks and power during sustained load (`powermetrics`), use random data, not zeros (zeros: 17–18 vs 13 TFLOPS on P100), and report sustained separately from burst. Same as our heat finding (30.8 cold vs 24.9 hot). |
| **CUDA graphs** (A:I41; B:I18) | **B, strong** | Apple's analog is the command-buffer boundary. MLX commits a buffer every N ops / M MB, about 20 µs GPU plus a CPU handler each. Raising `MLX_MAX_MB_PER_BUFFER` (2048) gave +11.9/+13.7% decode on Qwen3.6-35B-A3B and +2% on dense 27B on an M5 Max, tokens identical (mlx#4521 ✔). Our M5 ('g') defaults to 40/40, tighter than Max/Ultra (device.cpp ✔). Fork lesson that applies: it pays only for launch-heavy graphs (MoE, small models, Breeze's 15-step depth decoder) and must be retested per workload. Metal also has reusable indirect command buffers (T1), but MLX doesn't use them. |
| **Activation staging** (A:I11,I12,I20) | **B**, price first | MLX qmv does not stage x in threadgroup memory. Each simdgroup re-reads its x slice per 4-row block (`quantized.h:769-797` ✔). The fork's biggest single decode win was this exact fan-out. But Apple caches differ: Dynamic Caching, unknown M5 L1. So price it with the fork's deletion probe (replace x by a constant) before building anything. "Prepare once for sibling projections" (q/k/v, gate/up) is also open. |
| **Magic-number dequant** (A:I28) | **A** | Already in MLX qmv (masked-in-place bits, x pre-divided by 16/256/4096) and in llama.cpp Metal (mlx-code F1, practitioners F6). |
| **Compact causal mask** (A:I48; B:I30) | **A / small B** | The MLX fused path takes `mask="causal"`. The quantized-KV path still builds a bool mask *and* the full score matrix (base.py:89-92 ✔). That is folded into E3/E4. |
| **Launch fusions** (A:I42; B:I19,I28,I46) | **B** | `mx.compile` fuses elementwise ops only (mlx-code F10). There is no norm→matmul, gate+up+SwiGLU or quantize-epilogue fusion for affine models. vmlx's fused gate/up covers DSV4 and TurboQuant layouts only. The fork's floor of 1.5–3 µs per small kernel at batch 1 is the lever for small models (Qwen3-ASR-0.6B, Breeze's 1.7B backbone + 100M depth decoder). |

### The rest of the catalog (139 ideas in `lanes/fork-ideas-*.md`, grouped)

| Group | Already done (A) | Transfers (B) | Doesn't (C) |
|---|---|---|---|
| **Decode attention** | f32 accumulation; multi-query vector SDPA; 2-pass split-K | **GQA fold for non-power-of-two groups (A:I25/I26) — MLX lacks GQA 6 / D 256** (✔); exact-fit verify width / tile cliffs (A:I24, B:I13) — MLX vector SDPA needs qL·gqa ≤ 32, and #3826/#3838 show a 2.3–2.8× jump past it (unmerged PR); dequantize the KV tile on-chip, never the whole cache (A:I27/I54, B:I26) | Pascal-specific tile register counts |
| **Quantized matvec** | magic dequant; qmv_wide reuses weights for M 2–5 | x staging (above); multi-column regime as its own routing table (A:I63, B:I25); "kernel efficiency per format" sweep (B:I26: if a less-compressed format is faster, the loader is wasting bandwidth) | q8_1 activation path (MLX keeps float x already) |
| **Speculative / MTP** | vmlx: p/q rule, GDN rollback, sampled drafts; mlx-vlm: MTP + rollback (token-match rule) | depth-scheduled draft length (+16% at 260k); block verification (+1.8%); reduced-vocab draft head (2.05→1.4 ms); K/V-only catch-up; fixed-shape verify (relevant if `mx.compile` is used; shape changes retrace); per-request draft RNG seeding; "speculation loses when the working set exceeds the cache" (B:I45 → vmlx flash-MoE + MTP); accept rate reported beside t/s | — |
| **Host side** | mlx-lm async_eval pipeline | count syncs per step: Breeze has 15 `.item()` per frame (mlx-code F11); device-chained draft steps (B:I11); top-k without full sort (GPU argpartition is already fine) | CUDA pinned-memory specifics |
| **Long prefill** | chunked GDN kernel (`gated_delta_fused_chunk` ✔); flash-style fused attention | quantized-KV prefill → dequant + fused (mlx-lm PR #1510 unmerged; we measured 3×); measure attention-mass concentration before any sparse attention (B:I33); roofline per token first (B:I34) | exchange overlap on one GPU |
| **MoE** | vmlx fused pair-SwiGLU decode (DSV4) | expert LRU cache + staged fill + prefetch for SSD offload; slot loan (reuse idle prefill workspace); config finder minimizing R/prefill + 1/decode (B:I43) | PCIe P2P AllReduce |
| **Serving state** | — | recurrent-state checkpoint recipe (prefill with n_predict 0, save, extend); save the draft context with the slot; cap the logical batch for auxiliary heads (B:I10: acceptance 0.000 from one 5.3 GB copy) | — |
| **Correctness / measurement** | — | everything: KLD vs fp32 with a **control** (q4 vs f16 KV) to size changes; pinned eval corpus; fp64 op harness; whole-domain proofs for integer rewrites; ABBA + cold-first ordering; ≥512 tokens; minimum free memory over the run; "too clean" (identical results from different kernels) = stale binary; test at many batch widths (aliasing and kernel switches are size-dependent: mlx-lm verify T>1 uses different kernels, so greedy diverges 30–38 of 256 tokens, practitioners F13); races need repeated unsynced runs | — |

## Where sources disagree

- **Quantized-KV decode speed vs f16.** mlx-lm PR #1510 says decode is "at parity with (or ahead of)" f16. mlx-lm #1587
  (M4 Max) says it is slower and uses *more* peak memory. Our M5 measurement: q4 1.2× faster than f16 at 128k, q8 1.6×
  slower; prefill 3× slower. All three agree that the unfused path wastes most of the bandwidth advantage. They differ
  on model, shape and bits.
- **Dequantize-everything-first vs in-kernel.** llama.cpp Metal moved prefill to dequantize-to-F16 first (PR #27390
  merged): +16–78% on M2 Ultra, but a decode *loss* on M4 Max (216 → 144 t/s). The fork and both Apple papers favour
  in-kernel for decode and long context. Resolution, consistent with all of them: in-kernel for decode/verify,
  dequantize-then-fused for prefill-shaped calls (exactly mlx-lm PR #1510's split).
- **MTP on Mac.** llama.cpp Metal MTP is 11–28% slower on M1 Max (#23752; maintainer: "not a bug", others see gains).
  mlx-lm PR #990 reports 1.57× on an M4 Pro. The fork's I61/B:I45 explain both: speculation amortises weight reads only;
  verify attention grows with width × context, and verify width hits kernel cliffs. So the gain depends on chip,
  context and kernel coverage.
- **M5 Neural Accelerator decode gain.** ja lane: 1.19–1.27× decode vs one claim of about 2×. Apple's own MLX numbers
  (1.24–1.27× generation, M5/M4, matching the bandwidth ratio; apple-gpu lane F12, not re-checked by me) support the
  low figure.

## Gaps (not found or not verified)

- M-series GPU power cap, sustained clock, energy per op: no T1–T2 source. **Measure** with `powermetrics`.
- Per-dispatch and commit cost on M5 base: no source. **Measure** (E1 does it implicitly).
- M5 threadgroup bank count, L1/L2 sizes: none found.
- A published error bound for "fp16 chain folded into fp32 every k": not found (Blanchard/Higham/Mary block-FMA work is
  the lead). The fork's empirical law (NMSE ≈ 2.2e-8·k) stands alone.
- qmv_wide's re-read beyond 5 vectors (mlx-code F5) is lane-reported, not re-read by me.

## What the region lanes added

- **zh:** Cider (Mininglamp-AI), the only existence proof of the DP4A idea on Apple (M5 INT8 TensorOps in MLX). It is
  also a second, independent decode-SDPA win over MLX: up to 1.57× via GQA register tiling on M5 Pro. Zhihu returned 403
  and Juejin a bot wall.
- **ja:** M5 Neural Accelerators mainly help prefill (decode 1.19–1.27×). Mostly T3 snippets.

## Candidate experiments (ranked by value per cost; none run at the time of this report)

| # | Experiment (fork idea) | Target | Expected | Cost | How to measure |
|---|---|---|---|---|---|
| E1 | Sweep `MLX_MAX_OPS_PER_BUFFER` / `MLX_MAX_MB_PER_BUFFER` (CUDA graphs → command buffers) | Breeze depth decoder, Qwen3-ASR decode, a Qwen3.5/3.6 decode | +2–14%; biggest on launch-heavy loops | minutes, zero code | ABBA, ≥512 tokens, token/hash equality |
| E2 | Batched CFG + fewer GPU waits in Breeze (batch-1 matvec is memory-bound; device-chained steps) | another project (out of scope here) | measured 1.8× → 1.05× CFG cost; syncs untested | small | already measured; next: sync removal |
| E3 | Verify-cost curve 1..16 tokens on MLX (exact-fit verify width, B:I13/A:I24) | MLX qmv/qmv_wide + sdpa_vector at Qwen3.5/3.6 shapes | finds the cliffs (qL·gqa > 32; qmv_wide > 5) that cap MTP depth | small script | per-width ms, interleaved |
| E4 | Quantized-KV prefill → dequant + fused SDPA (mlx-lm PR #1510 logic) | mlx-lm `base.py` (monkeypatch in staging) | about 3× attention prefill (measured), 2.66× TTFT at 10k (PR) | small | equality vs current path + KLD on one model |
| E5 | GQA-6 / D-256 decode attention (fold group, split-S) via `mx.fast.metal_kernel` | Qwen3.5/3.6-27B long-context decode | about 1.5× decode attention (#3837 1.52×; Cider 1.57×) | medium | max\|Δ\| vs `mx.fast.sdpa`; fp32 accumulators or fp16 chains folded per tile |
| E6 | Fused attention over quantized / TurboQuant KV (dequant tile on-chip; never a decoded copy) | mlx-lm QuantizedKVCache, jang TurboQuant | q4 KV near f16 bandwidth: up to about 3× less attention time than f16 and about 3.5× less KV memory at 128k | large (the real project) | first a probe: TurboQuant live memory |
| E7 | Accuracy gate kit: KLD vs fp32 with a control, pinned corpus, fp64 op harness | Qwen3-ASR int8, Breeze 4-bit/mix48, SeedVR2 int8, any kernel above | turns "sounds fine" into numbers | small | — |
| E8 | x staging / sibling-projection prep in qmv: price with a deletion probe (constant x) first | MLX qmv on our M5 | unknown: 0 to double digits | probe small, kernel medium | deletion probe per fork M1 |
| E9 | W8A8 on M5 INT8 TensorOps (DP4A → NAX; Cider) | ASR encoder / LLM prefill | 1.2–1.9× prefill; 0 decode | medium (adopt or port) | KLD gate (int8 activations were the fork's top error source) |
| E10 | Sustained-load power and clock trace, random vs zero data | M5 during a 10-min decode | tells whether long runs are power- or heat-bound | small | powermetrics + t/s over time |
| E11 | MTP refinements for vmlx serving: depth-scheduled draft length, block verify, reduced-vocab draft head | vmlx native MTP | +2–16% at long context only | medium | tokens/cycle and ms/cycle at 2k/64k/128k |

## Sharp follow-ups

1. **measure:** E1 on Breeze, Qwen3-ASR and one Qwen3.5/3.6 model. Does any default-40/40 workload gain >5%?
2. **measure:** E3, the MLX verify-width cost curve at the Qwen3.6-27B shape. Where are the cliffs?
3. **solve:** E5, a GQA-6/D-256 split-S decode kernel as `mx.fast.metal_kernel`. Can it reach ≥1.4× at 64k–128k on M5
   with fp32 accumulators (no spill)?
4. **measure:** TurboQuant live memory vs plain KVCache at 32k (`mx.get_active_memory`). Is the decoded copy live?
5. **solve:** E6, fused attention on the TurboQuant/affine compressed cache, after 4 confirms the need.

## Sources (load-bearing; all re-checked by coordinator, see verified.md)

- T1 fork: OPTLOG.md / CHANGES.md / FINDINGS.md / GOAL-PREFILL300.md / VERIFICATION.md / MOE.md (local sparse clone, `src/`)
- T1 code: mlx 0.32.3 `quantized.h`, `sdpa_vector.h`, `gated_delta_update.h`; GitHub mlx v0.32.3 `device.cpp`,
  `scaled_dot_product_attention.cpp`; mlx-lm 0.31.3 `base.py`, `generate.py`, `cache.py`; jang_tools `turboquant/cache.py`,
  `attention.py`
- T1 papers: arXiv 2604.16957 (Open-TQ-Metal, 2026-04); arXiv 2610.33076 (Double KV Cache Optimization, M5 Max MLX, 2026-10)
- T2: Zakharko, Apple Neural Accelerators benchmark (2025-10); mlx#4521 (2026-09/10); mlx#3837 + comments (2026-07);
  mlx-lm#1587 (2026-07); mlx-lm PR#1510 (unmerged); mlx-lm PR#990 (open); mlx PR#3838 (unmerged); llama.cpp PR#27390
  (merged 2026-08-20)
- T1 vendor (self-reported): Mininglamp-AI/cider README
- T5 own: kv_attn_bench and breeze_cfg_bench on our M5 (this session)
