# Measured results: Kmic-68 ideas on M5 / MLX (2026-11-18)

Hardware: Apple M5 (10-core GPU), 24 GB, macOS 26.6. Stack: mlx 0.32.3, mlx-lm 0.31.3, mlx-optiq 0.8.1.
Code and raw output: `../../exp/` (`*.py`, `*.jsonl`).

Method:
- Each run used `taskpolicy -b nice -n 10` and a hard `gtimeout`.
- Kernel timings count GPU time only: the graph is built before the timer starts.
- Each timed graph uses one separate cache per layer, so on-chip caches can't hide DRAM reads.
- Methods are interleaved, and the median is reported.
- Attention correctness is the max abs error against an fp32 reference. Model correctness is logits against a bit-exact rollback control.

## 1. Measured bandwidth ceiling
- Metal read kernel: **123 GB/s** read. Copy: 129 GB/s read + write.
- MLX decode attention, one token (qL 1), already runs near that ceiling:

  | Context | 1k | 8k | 16k | 32k |
  |---|---|---|---|---|
  | GB/s | 55 | 114 | 118 | 90–108 |

- MLX quantized matvec, one row (M 1): **107–122 GB/s**, 87–99% of peak.

So on the M5, decode at width 1 is already close to bandwidth-bound in stock MLX. The fork's x-staging idea (E8) is capped at about 5–13% here. Not pursued.

## 2. GQA fold: the main result
Idea from the fork: the query rows that share one KV head become one matrix (gqa·qL rows), so K/V are read once per KV head.

### 2a. Hand-written Metal kernels: negative
| Version | Design | Result vs MLX |
|---|---|---|
| v1–v3 | Scalar split-K | Correct, never faster. v3 at qL 1 reached parity at 1k (89 vs 93 µs), lost at 8k (413 vs 295). |
| MMA | simdgroup 8×8 | Correct, 3–4× slower. |

- **Why v1–v3 lost:** MLX's per-head kernel already sits at 114 GB/s for qL 1. The 6 heads re-read K/V from on-chip cache, not DRAM.
- **Why the MMA kernel lost:** K is loaded transposed straight from device memory, with one simdgroup per threadgroup. It would need threadgroup staging, as MLX's steel kernels do.
- The 1.5× gains reported in the literature (M1 Ultra, Cider on M5 Pro) don't reproduce at qL 1 on M5 with mlx 0.32.3.

### 2b. MLX's dispatch has a verify-width cliff
Source: `sdpa_v0323.cpp`.
- Head dim 256 uses the fused vector kernel only when qL ≤ 8 and qL·gqa ≤ 32 (line 795).
- Beyond that, `use_fallback` sends qL > 8 to the unfused path (line 902).
- The unfused matmuls are slow below 16 rows per head.

Result for Qwen3.6-27B (24 query heads, 4 KV heads, head dim 256), 4k context:

| qL | 1 | 4 | 8 | 12 | 15 | 16 |
|---|---|---|---|---|---|---|
| µs | 165 | 394 | 812 | 1117 | 1216 | **506** |

Padding queries up to 16 rows (dummy rows prepended, error unchanged) gives 1.3–2.3× on qL 6–15.

### 2c. GQA fold in plain MLX ops: works
`folded_ops.py`, 20 lines, no kernel. Queries are reshaped to `[Hkv, gqa·qL, D]`, then one matmul, a precise softmax, and a second matmul. Error equals MLX's own path. Speedup vs `mx.fast.scaled_dot_product_attention`:

| Shape @ context | qL 1 | qL 2 | qL 4 | qL 8 | qL 12 | qL 15 |
|---|---|---|---|---|---|---|
| Qwen3.6-27B @4k | 0.93 | 1.00 | 1.66 | 2.80 | 3.35 | 3.41 |
| Qwen3.6-27B @32k | 1.06 | 1.22 | 2.11 | 3.27 | 3.94 | **4.02** |
| Qwen3.5-9B @4k | 0.92 | 0.72 | 1.10 | 1.75 | 2.36 | 2.48 |
| Qwen3.5-9B @32k | 0.95 | 1.16 | 1.61 | 2.91 | 2.99 | 3.04 |
| Gemma-4 global @32k | 1.21 | 1.14 | 1.45 | 1.89 | 2.13 | 2.21 |

Dispatch rule, safe in every measured case: **fold when gqa·qL ≥ 16, else stock.**

### 2d. Real model before/after: Qwen3.5-9B-OptiQ-4bit
Setup:
- Runtime swap of `mlx_lm.models.qwen3_next.scaled_dot_product_attention` (`patch_fold.py`). No installed file was touched.
- Each verify pass is undone between runs. The plain-vs-plain control was bit-identical (diff 0.0) in every case.

Verify forward pass, ms before → after:

| Context | qL 4 | qL 6 | qL 8 | qL 12 |
|---|---|---|---|---|
| 4k | 59.6 → 59.6 | 75.4 → 74.2 | 88.8 → 87.1 | 130.9 → 126.8 |
| 16k | 65.2 → 63.8 | 83.9 → 78.9 | 99.6 → 91.4 (1.09×) | 148.6 → 131.8 (1.13×) |
| 32k | 80.4 → 71.0 (1.13×) | 105.1 → 86.0 (1.22×) | 126.2 → 98.5 (**1.28×**) | 172.8 → 139.9 (1.24×) |

- Logit diff 0.09–0.21 (bf16 rounding). Argmax was the same in 20/21 cases; one 4k qL 4 near-tie flipped.
- The 9B has only 8 full-attention layers, and weight reads dominate. The 27B (16 attention layers, 4× attention gain) should gain more; it is not on this machine.

## 3. Quantized-matmul width cliff (verify width on the weights)
Example: 4-bit, 4096 → 12288.

| Rows M | 1–4 | 5 | 6–12 | 13–16 |
|---|---|---|---|---|
| µs | 240–266 | 309 | 429 → 820 | 350–370 (GEMM path) |

- 8-bit has the same shape.
- The down projection (12288 → 4096) has no cheap GEMM step: 980 µs at M 13–16.
- Splitting M into chunks helps only some shapes (≤ 1.38×).

Consequence for the whole 9B verify pass at 4k:

| qL | 1 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| ms | 57.6 | 58.5 | 59.6 | 75.4 | 88.8 | 130.9 | 122.0 |

So **verify widths 1–4 are almost free; 6–12 are expensive.** That is the fork's "exact-fit verify width" lesson, on M5.

## 4. MTP end-to-end through optiq (`optiq/runtime/engine.py`, greedy, 200 tokens)
Short prompt:

| | AR | depth 1 | depth 2 | depth 3 |
|---|---|---|---|---|
| tok/s | 17.98 | 23.2 | 23.4 | 22.4 (24.3 with fold) |
| Acceptance | — | 0.70 | 0.54 | 0.44 |

Phases measured alone, with a sync before and after:

| Phase | Short prompt | 16k |
|---|---|---|
| Draft step | 10.8 ms | 11.6 ms |
| Capture-verify, width 2 vs plain forward | 62.5 vs 57.3 ms | 67.8 vs 62.8 ms |
| Capture-verify, width 4 vs plain forward | 68.9 vs 58.3 ms | 78.4 vs 66.2 ms |

- **optiq's docstring claim is wrong on this machine.** It says "K=2 verify costs ~2x K=1" and so pins depth 1. Measured: verify at width 4 costs 1.10–1.13× width 2. The real costs are:
  1. Serial draft steps at 11 ms each (an MTP layer plus the full 248k-vocab LM head, synced per step). That is the fork's "reduced-vocab draft head" target.
  2. The per-token Python loop of the GDN state capture: +5 ms at width 2, +10–12 ms at width 4.
- **16k end-to-end numbers are invalid.** The run swapped: footprint 17 GB, 16k prefill 49 s against ~5 s per 4k earlier. They are reported as such, not used.

## 5. Fused attention on a 4/8-bit KV cache (E6): works
Files: `qkv_attn.py` (kernel `qkv_sdpa2`), `bench_qkv.py`. Memory capped at 2 GB; peak 2.1 GB.
- The kernel reads the packed cache once and dequantizes in registers.
- Scores use the per-group trick s·Σq·w + b·Σq.
- Two layouts: lane-per-key for scores (no cross-lane sum per key, one exp per key) and lane-per-dims for values.
- The first version (lane-per-dims everywhere) was compute-bound at 2× slower: every lane redundantly recomputed the softmax update for every key.

Shape: Qwen3.6-27B attention (24 query heads, 4 KV heads, head dim 256), 6 distinct layers per graph, GPU time.

| Cache | Context | qL 1 vs mlx-lm quantized | qL 1 vs bf16 fused | qL 4 vs mlx-lm | qL 8 vs mlx-lm |
|---|---|---|---|---|---|
| 4-bit | 4k | 2.14× | 1.32× | 2.38× | 1.73× |
| 4-bit | 16k | 3.66× | 1.92× | 2.25× | 1.80× |
| 4-bit | 32k | **4.53×** (564 vs 2554 µs) | **2.70×** | 2.32× | 1.88× |
| 8-bit | 32k | 3.98× | 2.03× | 1.83× | 1.41× |

- Error against an fp32 reference on the same dequantized cache is lower than mlx-lm's path: at 32k, qL 1, 4-bit, 0.0002 vs 0.0004.
- 4-bit and 8-bit run close in time, so the kernel is still compute-bound. More gain is possible with a matrix-unit QK.
- Directly relevant to Qwen3.5-9B-OptiQ, which ships `kv_config.json` with a 4-bit KV cache on its attention layers. That cache currently runs through mlx-lm's unfused path, measured here as 1.7–2.9× slower than the bf16 path at the same context.

### 5b. Real model: Qwen3.5-9B-OptiQ-4bit with a quantized KV cache
File: `model_qkv.py`.

Setup:
- Prefill in bf16, then convert the cache to 4- or 8-bit (group 64), as mlx-lm does.
- 48-token greedy decode; every run starts from the same state (KV trimmed back, GDN state restored).
- 3 interleaved runs per variant.
- Capped: MLX memory limit 8 GB, plus a watchdog that kills the run above 9 GB. Peak footprint 8.5–8.6 GB, MLX peak 7.4–8.2 GB.
- The fused kernel was hooked into `qwen3_next.scaled_dot_product_attention` at runtime and handled all 6912 attention calls.

| Context | bf16 KV | 4-bit stock | 4-bit fused | 8-bit stock | 8-bit fused | 4-bit fused vs stock | 8-bit fused vs stock |
|---|---|---|---|---|---|---|---|
| 4k | 17.07 | 16.40 | 16.73 | 16.22 | 16.74 | 1.020× | 1.032× |
| 8k | 16.62 | 15.89 | 16.69 | 15.56 | 16.53 | 1.050× | 1.062× |
| 16k | 16.05 | 14.87 | 16.30 | 14.40 | 16.20 | **1.096×** | **1.125×** |

(Decode speed in tok/s.)

- Tokens were identical to the stock quantized path, 48/48 at every context and both bit widths. Stock 4-bit also matched bf16 KV, 48/48.
- With the fused kernel, a 4-bit cache decodes as fast as a bf16 cache, or slightly faster (0.98–1.02×), while holding about 3.6× less KV memory. With the stock path, quantizing the KV cache costs 4–7% decode speed at 4–16k.
- The model-level gain is smaller than the 2–4.5× kernel gain because this 9B has only 8 attention layers among 32, and weight reads dominate each ~60 ms token. Models with more attention layers, or longer context, gain more. 32k was not run, to stay under the memory cap.

## 8. Fold × 4/8-bit KV combined (Qwen3.5-9B shape, kernel level)
File: `bench_combo.py`. 16 query heads, 4 KV heads, head dim 256, 8 layers; peak 0.6 GB. Best path vs mlx-lm's quantized SDPA:

| KV | Context | qL 1 | qL 2 | qL 4 | qL 8 | qL 12 |
|---|---|---|---|---|---|---|
| 4-bit | 4k | kernel 2.48× | kernel 2.09× | kernel 2.15× | foldq 3.14× | foldq 3.12× |
| 4-bit | 16k | kernel 3.48× | kernel 2.36× | foldq 3.31× | foldq 3.71× | foldq 4.70× |
| 8-bit | 16k | kernel 2.71× | kernel 1.70× | foldq 2.85× | foldq 4.13× | foldq 4.13× |

- Dispatch rule: fused kernel for qL ≤ 3; folded `quantized_matmul` (M = gqa·qL) for qL ≥ 4.
- Wider kernel tiles (8–16 rows per threadgroup) were always slower: register and threadgroup-memory pressure.

## 9. Target 20 tok/s on Qwen3.5-9B-OptiQ-4bit: reached, 25–31 tok/s
Files: `tps20.py`, `tps20_quality.py`, `weight_bytes.py`.

**Floor:** each token reads 5.61 GB of weights:

| Part | GB |
|---|---|
| MLP | 2.98 |
| Linear attention | 1.24 |
| LM head (8-bit) | 1.01 |
| Full attention | 0.38 |

- At the 123 GB/s peak that is ≥ 48.9 ms per token, so plain decoding can never exceed **20.4 tok/s**. It already runs at about 57 ms, 85% of that ceiling.
- 20+ needs more tokens per weight read: MTP.

**Setup:**
- optiq's MTP engine, loaded via `OptiqEngine.from_loaded`. `OptiqEngine(path)` keeps a duplicate model copy and peaked at 10 GB.
- Swaps at runtime only; no files edited.
- Greedy, 160 tokens, real text prompts. MLX memory limit 8 GB plus a 9 GB watchdog; peak footprint 8.66 GB.

| Config (tok/s) | Short | 4k | 8k |
|---|---|---|---|
| Plain decoding (AR) | 17.91 | 17.24 | 16.86 |
| MTP depth 1 (stock optiq) | 22.48 | 20.98 | 20.80 |
| MTP depth 2 (stock) | 22.94 | 19.51 | 19.06 |
| MTP1 + fused GDN capture | 23.26 | 22.51 | 21.40 |
| MTP2 + fused GDN capture | 24.87 | 21.90 | 21.10 |
| MTP1 + capture + 32k-vocab draft head | 27.11 | 23.67 | 23.78 |
| **MTP2 + capture + 32k-vocab draft head** | **30.75 (1.72×)** | **25.29 (1.47×)** | **25.43 (1.51×)** |
| MTP3 + capture + vocab + verify fold | 28.52 | 23.77 | 24.39 |
| MTP2 + capture + vocab + fold + 4-bit KV | 30.29 | 25.29 | 21.56 |

What each piece is:
- **Fused GDN capture:** optiq's own `linear_gdn_from_conv` backend, selected instead of the per-token Python loop.
- **Reduced-vocab draft head** (the fork's idea): drafts score only token IDs < 32768 (rows of the quantized LM head). The draft step drops from about 11 ms to a few ms. It's lossless by construction: the full model verifies every token.
- **Coverage of the 32k subset over the plain-decoding output:** 98.1% (short), 88.1% (4k), 89.4% (8k).
- Depth 3 and the 4-bit KV path don't help at ≤ 8k. 4-bit KV is lossy: its text diverges from bf16 KV.

**Quality.** Each output was re-scored with the plain model in one teacher-forced pass:

| Prompt | Output | Agreement with the model's argmax | Mean NLL |
|---|---|---|---|
| Short | AR | 98.75% | 0.247 |
| Short | MTP2 + capture + vocab32k | 99.38% | 0.254 |
| 4k | AR | 99.38% | 0.260 |
| 4k | MTP | 98.75% | 0.250 |
| 8k | AR | 98.75% | 0.219 |
| 8k | MTP | 100% | 0.255 |

- MTP text diverges from AR at near-ties: "document analysis" / "document content", "processor" / "processor cores".
- Stock optiq MTP diverges from AR the same way. That is batched-verify numerics, as optiq's own docstring states, not the speedups.

Rerun timings: 29.86 / 24.63 / 25.36 tok/s.

## 10. 16k and 32k, with an audit for "fast but rubbish" (2026-11-19)
Files: `tps_long.py`, `tps_long.jsonl`, `tps_long_texts.json`.

Setup:
- Prompt: the research corpus at 16k or 32k tokens, with 3 secret codes hidden at 10%, 50% and 90% depth.
- Task: list the codes, then summarize in 5 bullets. Greedy, 200 tokens.
- Capped: MLX limit 8 GB plus a 9 GB watchdog, held for the whole run. Peak footprint 8.70 GB (16k) and 8.96 GB (32k).

| Config | 16k tok/s | 32k tok/s | Codes found | Distinct 4-grams | Longest repeat | Re-score agreement / NLL |
|---|---|---|---|---|---|---|
| Plain decoding (AR) | 16.08 | 14.96 | 3/3, 3/3 | 0.985, 0.995 | 5, 4 | 0.995 / 0.286; 0.975 / 0.405 |
| MTP1 (optiq as shipped) | 19.67 | 16.17 | 3/3, 3/3 | 0.985 | 5 | — |
| MTP2 + capture + vocab32k | 23.01 | 15.74 | 3/3, 3/3 | 0.985 | 5 | 0.995 / 0.310; 1.000 / 0.320 |
| MTP2 + capture + vocab32k + fold | **23.59** | 15.61 | 3/3, 3/3 | 0.985 | 5 | — |
| MTP1 + capture + vocab32k + fold | 23.17 | 18.13 | 3/3, 3/3 | 0.985 | 5 | — |
| MTP2 + capture + vocab32k + fold + **4-bit KV** | 21.02 | **21.46** | 3/3, 3/3 | 0.985 | 5 | 0.960 / 0.326; 0.965 / 0.298 |

(Where two values share a cell, they are 16k, 32k.)

Audit:
- **Timing is independent of optiq's own counter:** wall-clock tok/s excluding prefill matches optiq's figure within 0.05 in every run. Every run produced its full 200 tokens.
- **No loops and no rubbish:**
  - Distinct 4-grams are 0.985–0.995 everywhere, and the longest repeated span is 4–5 tokens: normal for list formatting.
  - Every run found all 3 codes at both lengths.
  - I read the AR, best-MTP and 4-bit-KV texts at both lengths. All are coherent, faithful summaries of the document.
- **Re-score:** the bf16 MTP output agrees with the plain model's argmax as well as or better than AR's own output (0.995–1.000 vs 0.975–0.995).
- **The 4-bit KV path is lossy.** Agreement drops to 0.96 because the plain re-scorer uses a bf16 cache, but NLL stays within AR's range and the task still passes.

Why 32k is different: verify attention over a 32k bf16 cache dominates each MTP cycle.
- Verify widths 2–3 use MLX's vector kernel, which scales with width (§2b).
- MTP2 drops to AR speed. MTP1 holds up better at 18.1.
- With a 4-bit cache, the fused kernel (§5) makes long-context verify cheap: 21.5 tok/s.

Recommended policy for this model on M5:
- **Context ≤ ~16k:** bf16 KV, MTP depth 2, fused capture, 32k-vocab draft head and fold. 23.6–30.8 tok/s, output quality equal to AR.
- **Context ~32k:** switch to a 4-bit KV cache with the fused kernel and folded qmm. 21.5 tok/s, slightly lossy, task still correct.
- The crossover between 16k and 32k was not measured.

Load-memory finding: importing optiq before `mlx_lm.load` lifts the load peak from **7.15 GB to 10.92 GB**. Steady state is the same 7.6 GB. Load first, then import optiq.

## 11. Gemma-4-12B-OptiQ-4bit + assistant MTP drafter; Qwen 4-bit KV at 16k/32k (2026-11-19)
Caps: Gemma at MLX 12 GB + 14 GB watchdog + swap guard; swap growth 0.0 GB in every run. Files: `gemma_test.py`, `gemma_full.jsonl`, `gemma_texts_full.json`, `gemma_attn_bench.py`, `tps_long2.py`.

Gemma decode tok/s, 200 tokens, greedy:

| Context | AR | MTP default | block 2 | block 3 | block 4 | block 6 | Footprint peak |
|---|---|---|---|---|---|---|---|
| Short | 13.03 | 24.53 | 18.51 | 22.38 | 24.46 | 24.49 | 10 GB |
| 4k | 12.72 | 23.40 | 18.22 | 22.59 | **24.08** | 23.24 | 11 GB |
| 16k | 12.28 | 20.11 | 16.49 | 20.06 | **20.48** | 19.54 | 12 GB |
| 32k | 12.00 | **21.54** | 17.21 | 19.92 | 21.43 | 20.43 | 13 GB |

- All 3 codes found in every long run. Distinct 4-grams ≥ 0.985.
- MTP text is identical to AR at short and 4k for most blocks. Elsewhere it splits at near-ties (read: same content).

Gemma global-layer attention (16 query heads, 1 KV head, head dim 512), fold vs `mx.fast`: 0.88–1.13× at qL 1–7 and 4–32k. No gain: MLX's unfused fallback already batches the GQA heads.

Qwen 4-bit KV path, extra run:

| Context | AR | MTP1 + capture + vocab32k + fold + q4kv | MTP2, same |
|---|---|---|---|
| 16k | 16.53 | 22.62 | 22.91 |
| 32k | 15.26 | 21.69 | 21.85 |

- Codes 3/3. Re-score agreement 0.96–0.965, NLL 0.30–0.33.
- At 16k the bf16 path (23.6) stays best; at 32k this path wins.

Summary of both models before and after: [before_after.md](before_after.md).

## 12. The cheap ones, packaged: prefill (#6), n-gram + MTP (#3), mlx-fastpath 0.0.2 (2026-11-19)
Package: this repo (source + research). Installed as a separate copy into `<venv>`
(non-editable, v0.0.2). Every run below inside `MemoryGuard`, or checked by swap before and after: swap growth 0.

**#6 prefill** (`research/prefill/prefill_bench.py`, Qwen3.5-9B; seconds, then speedup vs stock):

| Prompt | Stock (512 chunks, LM head on every position) | Last-position head, 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| 4k | 6.67 s | 1.34× | **1.36×** | 1.34× | 0.93× |
| 16k | 26.13 s | 1.17× | **1.23×** | 1.19× | 0.87× |
| 32k | 67.85 s | 1.23× | **1.37×** | 1.34× | 1.05× |

- Final argmax identical in all 15 runs; logit difference ≤ 0.39 (bf16, different chunking).
- MLX peak at 1024: 7.88 / 8.03 / 8.32 GB, equal to or below stock. The 4096 chunk is slower and heavier: 9.6 GB MLX, 11.0 GB footprint.
- That run had no watchdog; swap still didn't grow.

**#3 n-gram + MTP** (`research/ngram/ngram_mtp_bench.py`, greedy, 400 tokens, tok/s):

| Task | Plain | A (MTP2 + patches) | N (optiq n-gram + stock MTP drafter) | N+ (n-gram + 32k-vocab MTP2 drafter) |
|---|---|---|---|---|
| Code edit (669-token prompt) | 17.3 | 26.9 | 71.0 | **71.4** |
| Copy a table (1168) | 17.3 | 31.0 | **60.2** | 59.4 |
| Free chat (32) | 16.7 | **29.9** | 23.6 | 27.3 |

- Code edit: all four outputs are byte-identical to plain decoding, and the edited file parses with both changes made.
- Copy: N and N+ split from plain decoding at char 705 and 734 of 744, at a near-tie.
- Peak footprint 8.5 GB; swap growth 0.0.
- Only short prompts so far; 4k–32k is in the next-round plan.

**Package tests:**
- 50 fast tests (6.7 s) plus 1 slow real-model test, all passing on the installed copy.
- Slow test on 0.0.2: 4k context, 24.17 tok/s, codes 3/3, peak 8.53 GB, swap 0.
- Mutation checks, each deliberate bug caught by at least one test:
  - fold causal off-by-one (25 failed),
  - kernel drops the key bias (17),
  - dispatch ignores the valid length (4),
  - the draft can pick other IDs (1),
  - changelog out of order (1),
  - version guard removed (1),
  - long-context switch moved (1),
  - prefill takes the wrong position (2),
  - drafter uses the full head (1). The first version of that test missed this bug, so it was rebuilt with a head whose overall best ID lies outside the subset.

## 13. DFlash block-diffusion drafter vs setup A, Qwen3.5-9B-OptiQ-4bit on M5 24 GB (2026-11-19)
Files: `research/dflash/` (`prompts.json`, `baseline_child.py`, `dflash_child.py`, `audit.py`;
results in `baseline.jsonl`, `dflash.jsonl`, `rescore.jsonl`).

Setup:
- `dflash-mlx 0.1.8` with the `z-lab/Qwen3.5-9B-DFlash` drafter (6 layers, 16-token blocks).
- Same token ids for every method, greedy, 300 tokens. Each config ran in its own process inside `MemoryGuard`.
- Swap growth was 0.0 in every run.
- The drafter's `config.json` (transformers 5.7) nests `rope_theta` and `block_size`. dflash-mlx 0.1.8 wants them at
  the top level, so `dflash_child.py` copies them up at load time. No files were edited.

Configs: D1 bf16 drafter, adaptive verify; D2 4-bit drafter (`w4:gs64`), adaptive; D3 = D2 + `DFLASH_VERIFY_QMM=1`;
D4 = D3 with `verify_mode dflash` (fixed 16-token blocks).

tok/s (accept rate in brackets for the speculative methods):

| Prompt (tokens) | AR | A (MTP2) | D1 bf16 | D2 w4 | D3 w4+QMM | D4 fixed blocks |
|---|---|---|---|---|---|---|
| chat (32) | 17.66 | 30.68 (0.53) | 24.76 (0.65) | 30.64 (0.65) | **31.92** (0.65) | 23.11 (0.70) |
| math (104) | 17.77 | 36.04 (0.72) | 43.03 (0.81) | 42.91 (0.79) | **44.19** (0.79) | 43.51 (0.84) |
| code_edit (664) | 17.51 | 26.99 (0.42) | 109.65 (0.93) | 115.41 (0.93) | 114.60 (0.93) | **116.31** (0.93) |
| needle_4k (4095) | 17.26 | 23.92 (0.34) | 24.03 (0.62) | **26.36** (0.60) | 26.28 (0.60) | 19.79 (0.67) |
| needle_16k (16095) | 16.32 | **24.89** (0.42) | 20.31 (0.58) | 22.59 (0.56) | 22.65 (0.56) | 16.50 (0.62) |
| Peak footprint, short / 16k (GB) | 8.6 / 9.4 | 8.4 / 9.3 | 10.0 / 11.0 | 8.2 / 10.0 | 8.7 / 11.0 | 8.2 / 10.0 |

Findings:
- **Code edit: 4.3× setup A, 6.6× AR (115 tok/s)**, ahead of n-gram + MTP (§12, 71 tok/s).
  - The drafter predicts copied spans in 16-token blocks: 0.93 accept.
- **Math: about +20% over A** (43–44 vs 36).
- **Chat and 4k: roughly a tie with A** (chat 30.6–31.9 vs 30.7; 4k 26.3 vs 23.9).
- **16k: A wins.** DFlash drops to 22.6, and fixed blocks to AR speed (16.5).
  - Verify attention over a long bf16 cache costs more per cycle than the extra accepted tokens save (same effect as §10 at 32k).
- **4-bit drafter beats bf16** everywhere except a tie on math, and uses 1–1.8 GB less: use `w4:gs64`.
- **QMM verify kernel:** +4% on chat/math, noise elsewhere, +1 GB at 16k. Optional.
- **Fixed 16-token blocks (D4):** hurts chat and long context. Keep adaptive.
- **Far below the dflash-mlx README** (M5 Max, bf16 target: 135 tok/s @1k).
  - The M5's lower bandwidth and the 4-bit target make AR cheap, so each verify pass costs relatively more.

Audit (`audit.py`):
- **Timing:** wall tok/s (excluding prefill) is 0.99–1.00 of the reported tok/s in every run. All runs produced their full 300 tokens.
- **Task checks:**
  - Hidden codes found 3/3 in all 12 needle outputs.
  - code_edit: in every method, the 941 chars written before the 300-token cut are a verbatim prefix of the expected file, with both edits made.
  - The file (~600 tokens) is longer than the cap, so the "parses" check couldn't run. Next time: max_tokens ≥ 700.
- **Repetition:** distinct 4-grams 0.89–0.997, longest repeat 4–9 tokens (math 9 in every method, AR included).
- **Splits from AR:** where outputs differ from AR (chat at token 62, needle_16k at token 5), A and DFlash split at the same tokens, which points to near-ties.
  D2 and D3 outputs are identical.
- **Re-score** by the plain model (argmax agreement / mean NLL; outputs re-encoded from text, 300/300 tokens round-trip):

| Method | chat | math | code_edit | needle_4k | needle_16k |
|---|---|---|---|---|---|
| AR | 0.997 / 0.269 | 1.000 / 0.069 | 1.000 / 0.001 | 0.983 / 0.272 | 0.983 / 0.323 |
| A | 0.990 / 0.264 | 1.000 / 0.069 | 1.000 / 0.001 | 0.980 / 0.358 | 0.997 / 0.326 |
| D1 bf16 | 0.990 / 0.256 | 1.000 / 0.069 | 1.000 / 0.001 | 0.990 / 0.303 | 0.993 / 0.334 |
| D3 w4+QMM | 0.990 / 0.246 | 0.997 / 0.077 | 1.000 / 0.001 | 0.977 / 0.305 | 0.983 / 0.314 |
| D4 fixed | 0.983 / 0.261 | 1.000 / 0.069 | 1.000 / 0.001 | 0.987 / 0.326 | 1.000 / 0.326 |

  All within AR's own range: no quality loss.

Policy on this machine:
- **DFlash, 4-bit drafter, adaptive verify:** for edits and copy-heavy output, math or reasoning, and contexts up to ~4k.
- **Setup A:** for ~16k and above.
- Not yet measured: the crossover between 4k and 16k, and DFlash combined with the mlx-fastpath patches. dflash-mlx loads and runs the target model itself.

## 6. Command-buffer limits (E1): no gain on this workload
Setup:
- Qwen3-ASR-0.6B int8, 8 FLEURS English clips × 2 passes per run.
- Each run is a fresh process with the environment variables set.
- 3 runs per setting, interleaved ABBA. Peak ≤ 2.0 GB.

| ops / MB per buffer | default (40/40) | 10/10 (control) | 100/40 | 40/200 | 100/200 | 300/1000 |
|---|---|---|---|---|---|---|
| Speed vs default | 1.00 | 0.946 | 0.993 | 1.002 | 0.980 | 0.928 |

- Transcripts were identical in every run.
- The control shows the check can detect a change. Larger buffers don't help, and the largest is 7% slower.
- The +12–14% from mlx#4521 was for MoE decode on an M5 Max. It does not transfer to a small dense ASR model on the base M5. Untested: a MoE model on this machine (none local).

## 7. Not run
- TurboQuant memory probe (no TurboQuant model local).
- 27B end-to-end (not on this machine).
- optiq `linear_gdn_from_conv` capture backend.

## Ranked next steps from these numbers
0. **4-bit KV fused attention** (`qkv_sdpa2`) for decode in OptiQ-style quantized-KV models: 2–4.5× on attention against the current quantized path, and faster than bf16. Needs a runtime hook on `mlx_lm.models.base.quantized_scaled_dot_product_attention` plus a model-level before/after (capped, ≤ 8k context).
1. **Ship the fold dispatch** (gqa·qL ≥ 16 → `folded_sdpa`) for MTP and speculative verify paths in our stacks. Measured 1.13–1.28× per verify pass on the 9B at 32k; up to 4× on the attention of a 27B-shaped model.
2. **MTP depth policy on M5:** depth 2–3 (verify width 3–4) is nearly free per pass. Avoid widths 6–12. optiq's depth-1 default leaves speed unused only if acceptance holds; it fell 0.70 → 0.44 at depth 3 on the short prompt. Measured best was depth 2: 23.4 tok/s, 1.30× AR.
3. **Draft head:** a reduced-vocab LM head for drafts (the fork's idea) cuts the 11 ms per draft step. The full-vocab head reads ~0.5 GB per draft.
4. **GDN capture:** replace the per-token loop with the chunked capture kernel. optiq already contains a `linear_gdn_from_conv` backend that is not the default; measure it.
5. Report to MLX upstream: the head-dim-256 unfused fallback for qL 6–15 is 2–4× slower than a GQA-folded matmul.

## 14. Next round (research/next_round/plan.md), unattended run (2026-11-19)
Every model run inside `MemoryGuard(min_free_gb=14, max_footprint_gb=14)`, MLX limit 12 GB, `taskpolicy -b`.

### 14.1 Branch drafting (item 1): no-go, measured before building
File: `research/branch/sim.py` → `sim.jsonl`. Qwen3.5-9B-OptiQ-4bit, 32k-vocab MTP head (setup A's drafter), greedy.
Method: plain greedy decoding for 300 tokens, keeping the main model's hidden state at every position. At each position
the MTP head drafts as a cycle would (step 1 top-2, step 2 top-2 from each step-1 branch); a draft counts as accepted
when it equals the greedy token. Then verify cost at widths 1–6 (`forward_with_gdn_capture`, rolled back, median of 6).
Peak footprint 9.36 GB, swap growth 0.0.

Acceptance (per cycle position): first draft right p1; top-2 rescues a miss; second draft right given the first.

| Prompt | p1 | top-2 rescues a miss | p2 \| p1 |
|---|---|---|---|
| chat | 0.70 | 0.43 | 0.54 |
| math | 0.84 | 0.50 | 0.75 |
| code_edit | 0.63 | 0.32 | 0.50 |
| needle_8k | 0.53 | 0.29 | 0.42 |

Verify pass, ms by width (rows incl. the last committed token); draft step 3.4 ms for 1 or 2 rows:

| Context | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| 332 | 55.2 | 55.5 | 58.9 | 60.3 | 62.0 | 74.0 |
| 8395 | 57.9 | 58.4 | 60.6 | 61.9 | 65.8 | 77.2 |

Projected tok/s = expected tokens per cycle / (draft steps + verify(width)). Every tree with ≤ 5 rows (6 hits the
width cliff); the chain2 projection (31.6 short) matches setup A's measured 30.8, so the model is calibrated:

| Prompt | chain2 (= A) | 4 rows: top-2 at d1 | 4 rows: top-2 at d2 | 5 rows: two chains | 5 rows: top-2 at d1 and d2 |
|---|---|---|---|---|---|
| chat | 31.6 | 32.9 (+3.9%) | 32.7 (+3.3%) | 33.2 (+5.0%) | **33.8 (+6.9%)** |
| math | 37.5 | 37.9 (+1.0%) | 38.1 (+1.4%) | 37.7 (+0.5%) | 38.3 (+2.1%) |
| code_edit | 29.5 | 30.7 (+3.9%) | 29.2 (−1.0%) | 30.4 (+2.9%) | 30.3 (+2.6%) |
| needle_8k | 26.0 | **27.5 (+5.8%)** | 26.9 (+3.5%) | 26.5 (+2.0%) | 27.4 (+5.2%) |

- **Fails the bar (≥ +10% at short and 8k) before any tree overhead.** The best tree gains +6.9% short and +5.8% at 8k
  in this idealised projection. A real tree verify adds cost on top: a custom attention mask (no fused `causal` path),
  a tree conv/GDN recurrence for the 24 linear layers, and a non-contiguous KV commit. So the real gain would be lower.
- Why: the second-best draft rescues only 29–50% of misses, and width 5 already costs +5–8% over width 3. The M5's
  cheap widths (≤ 4–5 rows) leave room for only one extra branch.
- Not built. What would change the verdict: a better drafter (higher p1/p2) or a cheaper width-6–8 verify (§3's
  matmul cliff), not branching itself.

### 14.2 Matrix-unit verify attention, bf16 cache (item 2): bar 1 passed, bar 2 met only at 400 tokens
Files: `research/attn32k/mma2.py` (kernel), `bench_kernel.py` → `bench_kernel.jsonl`, `profile32k.py` → `profile32k.jsonl`.

Kernel `mma2`: the GQA fold on simdgroup 8×8 matrices, with the two faults of the naive version (§2a) fixed:
- 16-key K/V chunks are staged in threadgroup memory with coalesced 16-byte loads by all 128 threads;
- 4 simdgroups split the head dims (64 each), with the Q slice held in registers. Partial scores are summed in
  threadgroup memory. Online softmax runs per row on the summed scores, then split-K over 512-key blocks and a
  combine pass.

It reads the cache's full buffers with `n_valid = cache.offset`, so the sliced view is never copied. It is hooked in
for bf16 caches at qL 2–4 only.

Kernel level, µs per layer (8 distinct layers per graph). Max error vs fp32 equals stock in every cell (≤ 0.0003):

| Shape @ context | stock qL 1 | qL 2: stock / fold / mma2 | qL 3 | qL 4 |
|---|---|---|---|---|
| 9B @8k | 481 | 342 / 384 / **328** | 705 / 696 / **478** | 536 / 435 / **381** |
| 9B @32k | 1240 | 1586 / 1433 / **1129** | 2028 / 1641 / **1365** | 2724 / 1621 / **1322** |
| 27B @32k | 1435 | 2117 / 1637 / **1381** | 3157 / **1674** / 7930 | 4072 / **1761** / 7927 |

- At the 9B shape, verify widths 2–4 at 32k cost 0.91–1.10× a stock single-token pass.
- **27B shape, ≥ 18 rows (qL ≥ 3): 4–5× slower.** With 3 row tiles, the Q and O tiles no longer fit in registers.
  That needs a row split (two passes) or more simdgroups. Not needed for the 9B; left open.

Real model, Qwen3.5-9B at 32k (`profile32k.py`): one prefill, then every measurement on the same cache, rolled back
after each. Verify pass in ms; "pkg" = the shipped dispatch (fold only when gqa·qL ≥ 16):

| Dispatch | w1 | w2 | w3 | w4 | w2–4 vs w1 |
|---|---|---|---|---|---|
| stock | 65.6 | 71.3 | 74.7 | 80.3 | 1.09–1.22 |
| pkg | 65.6 | 69.5 | 73.6 | 71.1 | 1.06–1.12 |
| fold everywhere | 63.7 | 68.4 | 69.8 | 72.7 | 1.07–1.14 |
| **mma2** | 65.8 | **66.5** | **68.9** | **69.9** | **1.01–1.06** |

- **Bar 1 (width 2–4 ≤ 1.15× single token): passed.** In an earlier run on the same prompt, pkg already met it at
  1.07–1.11.

End to end, MTP2 (setup A drafter + fused capture), pkg vs mma2, greedy. Audit per run:
- the re-score is the plain model (stock attention) teacher-forced over the output;
- codes are counted over the whole answer;
- swap growth was 0.0 in every run, peak footprint 10.0 GB.

| Prompt, tokens | pkg tok/s (accept, ms/cycle) | mma2 tok/s (accept, ms/cycle) | Same text | Re-score agree pkg / mma2 | Codes |
|---|---|---|---|---|---|
| needle_24k, 200 | 23.41 (0.42, 78.4) | **24.19** (0.42, 75.9) **+3.3%** | yes | 0.995 / 0.995 | 3/3 |
| needle_28k, 200 | 23.87 (0.47, 81.3) | **25.40** (0.48, 76.4) **+6.4%** | split at a near-tie | 0.995 / 0.995 | 3/3 |
| needle_32k, 200 | 19.57 (0.34, 85.9) | 19.22 (0.26, 78.8) | split | 0.975 / 0.995 | 3/3 |
| needle_32k, 400 | 19.17 (0.31, 84.8) | **20.01** (0.28, 77.5) **+4.4%** | split | 0.985 / 0.995 | 3/3 |

- mma2 cuts each cycle by 3% (24k), 6% (28k) and 8–9% (32k).
- At width 3, logits differ from stock by at most 0.12–0.31 (bf16 rounding; argmax the same). Where the text splits,
  both continuations are coherent and faithful; the mma2 one is denser ("(I4, I6, I12)") and drafts worse.
- In the 400-token mma2 run, the answer ended by itself at 369 tokens.
- **Bar 2 (bf16 MTP at 32k ≥ 20 tok/s): met only at 400 tokens (20.01); missed at 200 (19.22).** The miss comes
  from acceptance after the near-tie split, not from the kernel. At pkg's acceptance, 78.8 ms per cycle gives
  21.3 tok/s.
- Today's 32k baseline is 19.2–19.6, not the 15.7 in §10. §10 used another 32k prompt and an 8 GB MLX limit.
- Verdict: worth porting into `mlx_fastpath.attention` for bf16 caches at qL 2–4 and Hq/Hkv·qL ≤ 16, after the usual
  tests. A port is a package change, so it was not made in this run.

### 14.3 Fused decode step (item 3): the overhead is CPU graph build, not kernels; pipelining gives +10–11%
Files: `research/fused/profile_step.py` → `profile_step.jsonl`; `research/fused/async_ar.py` → `async_ar.jsonl`.

Profile of one plain-decoding token, chat prompt, short context:

| Part | ms |
|---|---|
| optiq's AR loop (forward, eval, `.item()`, next) | 55.7 |
| Python graph build alone | 3.3 |
| Pipelined loop: the next graph is built on the lazy argmax and queued with `async_eval`, as mlx-lm's `generate_step` does | **50.5** |
| All 249 quantized linears incl. the LM head at M = 1, one graph (5.6 GB, 115.7 GB/s) | 48.4 |
| Per layer type, as separate graphs: 24 GDN mixers / 8 attention mixers / 32 MLPs / norms | 13.5 / 4.0 / 25.5 / 0.7 |

- The 7 ms gap between optiq's loop and the weight-read floor is mostly the GPU idling while Python builds the next
  graph (3.3 ms), plus the sync and readback.
- With pipelining, only **2.0 ms (4%)** is left above the matmul floor. That is under the plan's 5% stop rule, so
  kernel fusion (norm→matmul, the GDN small ops) was not attempted.

Pipelined vs synchronous plain decoding, greedy, 300 tokens, best of 2:

| Prompt | Sync tok/s | Pipelined tok/s | Speedup | Same tokens | Codes |
|---|---|---|---|---|---|
| chat (32) | 17.64 | **19.66** | 1.114 | yes | — |
| needle_4k | 17.34 | **19.19** | 1.107 | yes | 3/3 |
| needle_16k | 16.27 | **17.94** | 1.103 | yes | 3/3 |

Peak footprint 9.6 GB; swap growth ≤ 0.

- **Passes the bar (≥ +7%, identical tokens).** The fix is the loop shape, not a kernel. 19.7 tok/s is 98% of the
  20.4 tok/s bandwidth ceiling (§9).
- mlx-lm's own `generate_step` already pipelines. The gain is over optiq's AR loop, which every "AR" baseline in this
  file used. Read old AR numbers as about 10% pessimistic for plain mlx-lm.
- Next: the same idea inside the MTP cycle. Each cycle now syncs 3 times (2 draft `.item()` calls, then verify).
  Keeping the drafts lazy and building verify on them makes it 1 sync per cycle. Estimated 3–7 ms of ~66 ms per cycle.

### 14.4 Trained draft head, EAGLE-3 style (item 4): feasibility note, no-go for EAGLE-3
Note: `research/eagle3/feasibility.md`; sources in `research/eagle3/sources.md`.
- **No-go for EAGLE-3.**
  - Published EAGLE-3 heads for Qwen3.5 lose to the native MTP head on acceptance (397B: MT-bench 3.03 vs 3.28;
    MTP wins every task).
  - EAGLE-3's speedups need deep trees, which the M5's verify-width cliff and GDN rollback rule out (§14.1).
  - Regenerating the training data at ~20 tok/s takes days to months.
- **Cheaper alternative: a FastMTP-style fine-tune of the shipped MTP head.**
  - It fits in ~9–11 GB with the 32k sub-vocab.
  - Headroom is smaller than in FastMTP: their vanilla head was at ~10% on position 2, ours is at 0.54 given
    position 1.
  - A pilot is defined with a go bar: p2 | p1 ≥ 0.65 on chat, p1 ≥ 0.60 at 8k.

### 14.5 DFlash follow-ups (item 5)
**b. mlx-fastpath patches inside DFlash.** Files: `research/dflash/dflash_child.py` (env `VOCAB`), `vocab.jsonl`.
- `fp.apply()` does not reach dflash-mlx's target:
  - dflash replaces each full-attention `__call__` with its own hook, which calls its own GQA fold
    (`engine/gqa_sdpa.py:grouped_gqa_sdpa`), so our fold has nothing to add;
  - it uses its own GDN capture/rollback (fused capture not applicable) and already pipelines with `async_eval`.
- The one transferable piece is the reduced-vocab head for the draft logits. dflash computes them with the target's
  full 248k-vocab head over 15 rows. It was applied by a runtime method swap on `EagerDraftBackend.draft_greedy`
  (draft calls only, verify keeps the full head); no installed file was edited.
- D2 (4-bit drafter, adaptive verify), greedy, 300 tokens, same session; swap growth 0.0:

| Prompt | D2 tok/s (accept) | D2 + 32k draft vocab | Change |
|---|---|---|---|
| chat | 32.24 (0.65) | 33.89 (0.63) | +5.1% |
| math | 43.96 (0.79) | **48.79** (0.81) | **+11.0%** |
| code_edit | 115.48 (0.93) | 96.93 (0.92) | **−16.1%** |
| needle_4k | 26.42 (0.60) | 26.26 (0.57) | −0.6% |
| needle_16k | 22.80 (0.56) | 22.62 (0.53) | −0.8% |

- **Fails the bar (≥ +10% on at least 2 prompts): only math passes.**
  - Code edits lose 16%: identifiers outside the lowest 32k ids break the 16-token blocks.
  - At 4–16k, acceptance drops 0.03 and cancels the cheaper head.
- Wall tok/s was 0.99–1.00 of the reported figure, and every run produced its full 300 tokens.
- Not re-scored, since the bar already failed.

**a. Crossover between DFlash (D2) and setup A, 6k–12k.**
- Files: `research/dflash/make_prompts_long.py` → `prompts_long.json` (same needle recipe), `crossover.jsonl`,
  `audit_long.py` → `crossover_audit.jsonl`.
- Greedy, 300 tokens, each method in its own guarded process. Peak footprint ≤ 10.0 GB, swap growth 0.0.

| Context | A tok/s (accept) | D2 tok/s (accept) | D2 vs A | Re-score agree A / D2 | Codes |
|---|---|---|---|---|---|
| 4k (§13) | 23.92 | 26.36 | +10% | 0.980 / 0.977 | 3/3 |
| 6k | 25.19 (0.40) | **25.64** (0.61) | +1.8% | 0.993 / 0.997 | 3/3 |
| 8k | 23.36 (0.33) | **24.05** (0.57) | +3.0% | 0.980 / 0.997 | 3/3 |
| 10k | 23.54 (0.34) | 23.70 (0.57) | +0.7% (tie) | 0.997 / 0.997 | 3/3 |
| 12k | **25.04** (0.36) | 23.80 (0.58) | −5.0% | 0.997 / 0.990 | 3/3 |
| 16k (§14.5b) | 24.89 (§13) | 22.80 | −8% | — | 3/3 |

- **Crossover at ~11k (±1k): passed.**
  - DFlash leads through 8k (by a shrinking margin) and ties at 10k; A wins from 12k on.
- Audit passed for both methods:
  - wall/reported 0.993–1.000, full 300 tokens;
  - distinct 4-grams 0.976–0.993, longest repeat ≤ 5;
  - re-score within AR's own range (§13: 0.983);
  - texts read: both list the codes, then give a faithful summary.
- A's acceptance varies by prompt (0.33–0.40), so its tok/s isn't monotonic in context. The crossover sits between the
  10k tie and the 12k loss.

**c. Whole-file code edit (max_tokens 800) and table copy (1168-token prompt).**
- Files: `research/dflash/prompts_c.json`, `long_out.jsonl`, `ngram_child.py`, `audit_c.py` → `audit_c.jsonl`.
- Every method ended by itself before the cap.

| Task | AR | A | D2 (DFlash, w4) | N+ (n-gram + 32k MTP2) |
|---|---|---|---|---|
| code_edit, 616 tokens | 17.45 | 26.18 (0.38) | **116.13** (0.93) | 54.16* |
| copy_table, ~393 tokens | 17.61 | 32.15 (0.58) | **84.55** (0.92) | 48.86* |

\* N+ tok/s includes prefill (~1.0 / 1.7 s), since `generate_ngram` returns only ids. Without it, about 59 and 62.

- **Code edit: all four return a file that parses, has both edits and equals the expected file byte for byte.**
  D2 adds one trailing newline after the closing fence.
- **Copy:** all four reproduce the `Config (tok/s)` table of §9 exactly.
  - The prompt says "first table"; §9 opens with a small `Part | GB` table, but every method, AR included, copied
    the main table, so the check uses that one.
  - Only the closing sentence differs: D2 splits from AR at char 705, N+ at 734.
- **DFlash is the clear winner for copy-heavy output:** 6.7× AR on the full code edit and 4.8× on the copy, ahead of
  N+ (§12's 71 tok/s was at 400 tokens and excluded prefill).
- Peak footprint ≤ 9.2 GB, swap growth 0.0.

**d. Router proposal (no package edit).** From §13, §14.2–14.5 on this M5, Qwen3.5-9B-OptiQ-4bit, greedy:

| Prompt type | Context ≤ ~10k | ~11k–24k | > 24k |
|---|---|---|---|
| Edit / copy / rewrite of prompt text | DFlash D2 (85–116 tok/s) | DFlash D2 (to measure above 4k for edits) | A + mma2 |
| Math / reasoning | DFlash D2 (44 vs A 36) | A | A + mma2 (or 4-bit KV path) |
| Chat / summary | DFlash D2 ≈ A (tie ±3%; A uses ~1 GB less) | **A** | **A + mma2** (bf16, 20 tok/s at 32k) |
| Any, plain decoding | pipelined AR loop (+10%) | same | same |

Detection rule:
- "edit/copy" when the n-gram match rate of the first 32 generated tokens against the prompt is high, or the request
  says return/rewrite/reproduce;
- otherwise treat it as chat.

Switch to A at a prompt length of ~11k tokens. Open questions:
- DFlash for edits at long context (only measured ≤ 1.2k);
- whether mma2 moves the bf16 vs 4-bit KV crossover (see "Also open").

### 14.6 Also open: bf16 MTP vs the 4-bit KV path at 20k–32k (where LONG_CONTEXT should sit)
- Files: `research/dflash/baseline_child.py` (cfg `Aq4`), `kvcross.jsonl`, `kvcross_audit.jsonl`.
- Setup A with a bf16 cache vs A with a 4-bit KV cache (fused kernel, folded qmm).
- Greedy, 300 tokens, MLX limit 12 GB, peak 10.0 GB, swap growth 0.0.

| Context | A bf16 tok/s (accept) | A 4-bit KV (accept) | 4-bit vs bf16 | Re-score agree bf16 / 4-bit | Codes |
|---|---|---|---|---|---|
| 20k | **24.06** (0.41) | 23.38 (0.37) | −2.8% | 0.997 / 0.977 | 3/3 |
| 24k | 22.83 (0.40) | **24.86** (0.43) | +8.9% | 0.997 / 0.960 | 3/3 |
| 28k | 22.90 (0.43) | **25.34** (0.47) | +10.7% | 0.987 / 0.990 | 3/3 |
| 32k | 18.01 (0.26) | **21.84** (0.35) | +21% | 0.993 / 0.963 | 3/3 |

- **The crossover is between 20k and 24k**, so `LONG_CONTEXT = 24_000` is about right (~22k would match the data).
  This is without mma2; mma2 adds 3–8% to bf16 at 24–32k (§14.2), which would move the crossover up but not past
  ~28k.
- **Quality at 32k: the 4-bit path dropped task 2.** It listed the 3 codes, then reproduced a section of the document
  (with its table) instead of the 5-bullet summary. Distinct 4-grams 0.943, longest repeat 12 (table formatting, not
  a loop).
  - Every other run in §14.5a and §14.6 wrote the summary (bf16 at 32k included).
  - This is the first task failure seen on the 4-bit path. §10/§11 passed at 32k on other prompts.
  - Its re-score agreement (0.96–0.98) stays below bf16's (0.987–0.997), as in §10.

### 14.7 Also open: n-gram + MTP (N+) at 4k–32k, bf16 and 4-bit KV, with re-score
- Files: `research/dflash/ngram_child.py` (`KV_BITS`), `prompts_ng.json`, `ngram_long.jsonl`, `ngram_long_audit.jsonl`.
- Summary task (needles), greedy, 300 tokens, prefill excluded.
- Wall-clock is our own timer; optiq has no counter here, so the "wall vs reported" check doesn't apply.

| Context | N+ bf16 tok/s | N+ 4-bit KV | Re-score agree / NLL, bf16 | 4-bit | Same-prompt reference |
|---|---|---|---|---|---|
| 4k | 22.35 | 22.25 | 0.987 / 0.333 | 0.927 / 0.362 | A 23.9, D2 26.4 (§13) |
| 16k | 22.67 | 22.84 | 0.993 / 0.332 | 0.950 / 0.342 | A 24.9, D2 22.8 |
| 32k | 18.59 | 19.42 | 0.983 / 0.336 | 0.947 / 0.395 | A bf16 18.0, A 4-bit 21.8 (§14.6) |

- Codes 3/3 everywhere, and every run wrote the summary. Distinct 4-grams 0.98–0.99, longest repeat ≤ 5.
  Peak footprint 10.0 GB, swap growth 0.0.
- **On summaries, n-gram lookup adds nothing over A:** 4–9% slower at 4k and 16k, a tie with bf16 A at 32k. N+ is
  for copy/edit output only (§12, §14.5c).
- **bf16 N+ re-scores within AR's range at 16k and 32k (0.993, 0.983).**
- **The 4-bit KV cache doesn't help N+ and costs quality:** agreement 0.93–0.95, the lowest in this file.
  - Its verify widths (up to 9 rows) get the folded quantized matmuls, not the fused kernel (qL ≤ 3).
  - Don't combine `generate_ngram` with a 4-bit cache.

## 15. Batch 1: cheap checks that could overturn earlier verdicts (research/batch1/plan.md, 2026-11-20)
Files: `research/batch1/` (scripts, `*.jsonl` raw rows, `*.log`). Every model run was inside `MemoryGuard(14, 14)`, one
at a time, with `taskpolicy -b nice -n 10` and a `gtimeout` cap. Swap growth was 0.0 in every run, and the peak
footprint was ≤ 11.0 GB. Every speed claim is the median of ≥ 2 interleaved runs (ABBA), with the range given.

### 15.1 B1 bandwidth ceiling, re-measured: 134.1 GB/s, just under the 135 bar
`peakbw2.py` → `peakbw2.jsonl`. It reads 1 GB (4 × 256 MB); each timed eval makes 4 passes (4.3 GB), median of 9.
Every Metal kernel XORs all it read into one word, checked against numpy: 75/75 matched.

| Kernel family (best config) | GB/s median [range] |
|---|---|
| old grid-stride uint4 XOR (control; 65536 threads, tg 256) | 123.2 [118.9, 125.5] |
| grid-stride, 4–8 loads in flight (8192 threads, tg 1024, ×4) | 122.5 [118.0, 124.2] |
| per-thread contiguous chunks (262144 threads, tg 1024, ×4) | 102.2 [96.1, 102.9] |
| **`mx.sum` over uint32** | **134.1** [129.0, 135.4] |
| `mx.max` over uint32 | 133.5 [125.7, 135.2] |
| MLX 4-bit gs64 qmv at M 1, 40 distinct 9B MLP matrices, (4096, 12288) | 121.4 [113.3, 121.9] |
| same, (12288, 4096) | 118.2 [109.2, 122.7] |

- The control reproduces the old 123 GB/s, so the old number was right for that kernel. MLX's own reduction reads
  9% faster. Threadgroup size (64–1024) and unroll (4/8) don't move the hand-written kernels past ~123.
- **New ceiling** at 5.61 GiB per token (the convention §9 used): 134.1 GB/s → 44.9 ms → **22.3 tok/s** (was 20.4).
  Pipelined AR (19.66 tok/s, §14.3) is **88%** of it, not 98%.
- **Decision rule (≥ 135 GB/s): not met**, so per the plan "98% of ceiling" and "4% left" are not formally withdrawn,
  and fusion (next_round item 3) and E8 staging stay closed. Corrections that hold anyway:
  - "can't exceed 20.4 tok/s" → 22.3; "98% of ceiling" → 88%.
  - §14.3's "2.0 ms (4%) left" was measured against the matmul graph (115.7 GB/s), and that still holds. The ~12%
    between the matmul graph and the raw-read ceiling sits inside MLX's 4-bit qmv kernel (118–121 GB/s vs 134), not
    in the small ops that fusion would remove. A faster qmv (E8-like staging) would have to target that kernel.
    Borderline: the user decides whether to reopen it.

### 15.2 B2 command-buffer limits on the 9B: no gain (confirms §6)
`b2_sweep.py` → `b2.jsonl`. `baseline_child.py` AR and A, chat and needle_4k, 300 tokens, one child per setting,
settings in ABBA order (S then reversed S), AR/A order flipped on the second pass. 2 runs each, tok/s median
[range], ratio vs default:

| ops / MB | AR chat | AR needle_4k | A chat | A needle_4k |
|---|---|---|---|---|
| default | 17.51 [17.40, 17.62] | 17.08 [16.94, 17.21] | 30.21 [29.99, 30.43] | 23.72 [23.48, 23.96] |
| 10 / 10 (control) | 0.984 | 0.984 | 0.985 | 0.990 |
| 100 / – | 1.004 | 1.001 | 1.006 | 1.008 |
| – / 200 | 1.009 | 1.012 | 1.005 | 1.007 |
| 100 / 200 | 0.994 | 1.005 | 1.002 | 0.996 |
| 300 / 1000 | 0.949 | 0.946 | 0.956 | 0.960 |

- **Pass bar (≥ +3%, ranges apart from the default's): failed.** The best setting is +1.2%, inside the default's range.
  300/1000 is 4–5% slower with ranges clear of the default's, and the 10/10 control 1–2% slower. That is the same
  pattern as E1 on the 0.6B ASR model (§6).
- Audit (`b_audit.py`): 48/48 rows have the full 300 tokens, wall/runtime tok/s 0.999–1.000, codes 3/3 on every
  needle_4k row, distinct 4-grams ≥ 0.981, longest repeat ≤ 5 words. **All 48 texts are identical across settings
  and to the §13 runs.**

### 15.3 B3 confidence-trimmed drafts: +3.6% at 32k at best, under the +5% bar
`conf_sim.py` → `conf.jsonl` (per-position p1, p2, a1, a2 and the AR text); `conf_proj.py` → `conf_proj.jsonl`.
Setup A's 32k-vocab MTP drafter, depth 2. Verify cost measured at each context, widths 1–4, median of 6 repeats.

| Prompt (context) | Verify w1 / w2 / w3 / w4 ms | p(a1) | Projected tok/s t = 0 (as shipped) | Best t | Gain |
|---|---|---|---|---|---|
| chat (332) | 55.5 / 56.9 / 57.2 / 58.8 | 0.70 | 32.47 | 0.2 | +0.03% |
| math (404) | 56.6 / 56.4 / 57.0 / 58.6 | 0.84 | 38.66 | 0.2 | +0.03% |
| code_edit (964) | 55.9 / 57.0 / 58.1 / 59.0 | 0.63 | 29.89 | 0.1 | +0.03% |
| needle_8k (8395) | 58.3 / 59.7 / 61.4 / 63.2 | 0.53 | 25.69 | 0 | 0 |
| needle_16k (16395) | 60.6 / 62.3 / 63.9 / 66.4 | 0.59 | 26.98 | 0.1 | +0.04% |
| needle_32k (32395) | 65.4 / 71.7 / 75.7 / 72.8 | 0.48 | 20.36 | 0.3 | **+3.6%** |

- The drafter's confidence predicts acceptance well. Below p1 0.3, step 1 is accepted 9–20% of the time on
  chat/needles, and above 0.8 it is accepted 71–94%. Step 2 with p2 < 0.3 is accepted 4–25%.
- But trimming saves little time: up to 16k, width 3 costs only 1.7–3.3 ms more than width 1 (~3%), plus one
  3.4 ms draft step. At 32k the gap is 10 ms, and the projection reaches +3.6%.
- **Pass bar (≥ +5% projected at some context): failed.** Not built. The fork's +13–16% was at 64k–260k, where
  wider verify costs more. This machine's guard keeps the 9B at ≤ 32k, so that range was not tested.
- (At 32k, w4 is cheaper than w3: the package's fold dispatch is used at qL 4, as in §14.2.)
- Caveat: the projection treats every position as a cycle start (as §14.1's `sim.py` did). The absolute tok/s runs
  ~7% above measured A (chat 32.5 vs 30.2); the comparison is between thresholds.

### 15.4 B4 DFlash knobs: no knob beats the defaults
`b4_sweep.py` → `b4.jsonl`. `dflash_child.py` gained `DRAFT_WINDOW` and `PROFILE` (dflash-mlx's own per-cycle trace,
switched on through the runtime context; nothing in dflash-mlx was edited). w4:gs64 drafter, `DFLASH_VERIFY_QMM`
unset (= §13 D2), 300 tokens, 2 interleaved runs each.

**Window.** dflash-mlx uses `max(requested, drafter sliding_window)`, and this drafter's `sliding_window` is 4096.
So **the "1024 default" in §13 was really 4096**, and the planned 1024-vs-4096 runs were the same config twice
(identical text, acceptance, and speed within 0.3%). Rerun as 4096 vs 16384 (16384 covers the whole 16k prompt):

| Prompt | 4096 | 16384 | Change |
|---|---|---|---|
| chat (control) | 32.24 [32.13, 32.35] | 32.09 [31.98, 32.19] | −0.5% |
| code_edit | 114.87 [114.76, 114.98] | 115.11 [115.10, 115.13] | +0.2% |
| needle_4k | 26.31 [26.27, 26.36] | 25.91 [25.89, 25.94] | −1.5% |
| needle_16k | 22.66 [22.63, 22.68] (accept 0.563) | 18.95 [18.91, 18.99] (accept 0.597) | **−16%** |

- At 16k, the wider window raises acceptance but costs more draft attention than it saves. Text is identical either
  way (lossless). Keep the default.

**Tree (`verify_mode ddtree`)** vs adaptive, all 5 prompts. Peak 10.0 GB; the guard did not stop it.

| Prompt | Adaptive | ddtree (accept, tokens/cycle) | Change |
|---|---|---|---|
| chat | 32.08 [31.97, 32.18] | 22.70 [22.25, 23.15] (0.73, 3.70) | −29% |
| math | 43.75 [43.50, 44.00] | 39.22 [38.79, 39.66] (0.84, 6.38) | −10% |
| code_edit | 114.47 [114.25, 114.68] | 110.69 [109.95, 111.43] (0.93, 15.0) | −3% |
| needle_4k | 26.17 [26.06, 26.28] | 17.47 [17.34, 17.60] (0.69, 3.23) | −33% |
| needle_16k | 22.66 [22.63, 22.68] | 10.97 [10.86, 11.08] (0.60, 2.52) | −52% |

- ddtree accepts more per cycle but loses everywhere, more so with context. In dflash-mlx's code, a target without
  tree verify gets branch candidates verified as a batch on a copy of the target cache per candidate, every cycle.
  That fits a cost that grows with context (inferred from the code, not profiled).
- ddtree's text differs from adaptive's (chat at char 560, needle_4k at char 79), stays coherent, and keeps codes 3/3.
- Texts read by eye (every ddtree output and the 16384-window needle_16k): coherent, on task, codes listed exactly,
  the 5-bullet summary written, the code_edit carrying both edits; no loops or garbage.

**Per-cycle profile of the best config** (adaptive, window 4096; tracing on; tok/s within 1.5% of the untraced runs):

| Prompt | tok/s | Cycles | Block 4 / 16 cycles | Tokens/cycle | Draft ms | Verify ms | Cycle ms | Draft share |
|---|---|---|---|---|---|---|---|---|
| chat | 31.83 | 106 | 92 / 13 | 2.83 | 17.9 | 58.9 | 79.6 | 21% |
| math | 43.58 | 63 | 35 / 27 | 4.76 | 18.8 | 61.2 | 82.3 | 18% |
| code_edit | 113.76 | 20 | 0 / 19 | 15.0 | 0.7 | 121.5 | 125.5 | 5% |
| needle_4k | 25.89 | 120 | 107 / 12 | 2.50 | 24.5 | 61.3 | 89.1 | 26% |
| needle_16k | 22.52 | 131 | 117 / 12 | 2.29 | 24.2 | 67.1 | 94.1 | 25% |

(Medians per cycle; the adaptive reduction to block 4 happened twice per run and stuck.)
- Outside code_edit, DFlash spends ~90% of cycles at block 4 and commits 2.3–2.8 tokens per cycle, about what setup
  A gets. But its 6-layer drafter costs 18–24 ms per cycle against A's 2 × 3.4 ms. That is why A ties or wins on
  chat/long context. A cheaper draft at block 4 is the lever, not window or tree.
- **Pass bar (≥ +5% over §13 on ≥ 2 prompts): failed.** No config beats §13's D2 (chat 30.6, math 42.9, code 115,
  4k 26.4, 16k 22.6). Today's adaptive runs give 32.1 / 43.8 / 114.5 / 26.2 / 22.7, all within ±5% of §13.
- Audit (`b_audit.py b4.jsonl`): every row has the full 300 tokens, wall/runtime 0.989–0.998, codes 3/3 on every
  needle row, distinct 4-grams ≥ 0.971, longest repeat ≤ 6 words. Every adaptive run (any window) is identical to
  §13's D2 text.

### 15.5 What changes in earlier verdicts
- §9 / §14.3: the bandwidth ceiling is 134 GB/s (22.3 tok/s), not 123 (20.4). Pipelined AR is at 88% of it. The
  remaining gap is in MLX's qmv kernel, under the 135 GB/s rule for reopening.
- §6: holds on the 9B (AR and MTP): no command-buffer setting helps.
- §13: its "window 1024" was 4096 in effect. Neither a wider window nor ddtree helps (verify cap not swept: adaptive
  already drops to block 4); adaptive at defaults is the best DFlash config here.
- Confidence trimming (fork attempt 198) doesn't transfer at ≤ 32k on this machine (+3.6% at best).

## 16. Batch 2, low-memory steps (research/batch2/plan.md, 2026-11-20): coverage check, N2 models, KL gate
No model was run for any of this. The heaviest step peaked at 0.81 GB, and swap didn't grow. The model runs (R1–R8)
are planned in `research/batch2/plan.md`, with pass bars set before any run.

### 16.1 Draft-vocabulary coverage (`coverage.py` → `coverage.jsonl`)
§15.4's profile puts DFlash's draft at 18–24 ms per cycle. dflash-mlx scores the draft through the target's full
8-bit LM head (`draft_backend.py`: `target_ops.logits_from_hidden`), about 1 GB read, ~8 ms per cycle. §14.5b's
32k-token head cut that, but lost 16% on code_edit.

Re-tokenizing the earlier lossless outputs shows why:

| Prompt | Output tokens the draft can't propose, ids < 32768 | … plus the prompt's ids | Extra head rows |
|---|---|---|---|
| chat | 2.7% | 2.7% | 5 |
| math | 4.3% | 4.0% | 10 |
| code_edit (300 / 616 tokens) | 12.0% / 11.0% | **0.0% / 0.0%** | 35 |
| needle 4k / 6k / 8k | 10.7 / 11.7 / 14.3% | 4.7 / 3.7 / 2.7% | 184–326 |
| needle 10k / 12k / 16k | 13.0 / 10.7 / 10.0% | 3.0 / 2.3 / 1.3% | 392–565 |
| copy_table | 2.3% | 0.5% | 39 |

- Most out-of-vocabulary tokens are copied from the prompt.
- `dflash_child.py VOCAB=32768 VOCAB_PROMPT=1` now drafts over that id set. `test_subhead_ids.py` checks that its
  argmax is the full head's argmax over the set, at the same id; it fails on the old first-k mapping.
- Run R1 measures it.

### 16.2 N2: OptiQ's 8-bit layers requantized to 4-bit (`requant.py`)
- **What OptiQ keeps at 8 bits:** 134 of 250 quantized layers (5.2 bits/weight). That is every GDN `in_proj_a/b` and
  `out_proj`, the LM head, embed_tokens, and some MLP and attention layers.
- **Method:** each 8-bit weight is dequantized as the model computes it, then quantized with `mx.quantize` (4-bit,
  group 64), streaming row chunks. Each conversion took ~20 s at ≤ 0.77 GB.
- **embed_tokens stays 8-bit:** it is a row lookup, so it reads almost nothing per token.

| Variant | Layers converted | Bytes read per token | Speed limit at 134 GB/s (§15.1) |
|---|---|---|---|
| source (`Qwen3.5-9B-OptiQ-4bit`) | — | 6.02 GB | 22.3 tok/s |
| `-N2-gdnhead`: GDN layers + LM head | 90 | 5.09 GB | 26.4 tok/s (+18%) |
| `-N2-all`: every 8-bit layer | 133 | 4.47 GB | 30.0 tok/s (+35%) |

- **Checks:** `requant.py check` found 0 problems on both. It confirms:
  - unchanged tensors are byte-identical;
  - every sampled converted row equals `mx.quantize` of the 8-bit values;
  - shapes and config agree;
  - the official safetensors parser reads the files.

  `test_requant.py` covers these on a tiny fake model, including a changed byte, wrong 4-bit codes and a wrong
  conversion, all caught.
- **Weight change:** the relative weight error 4-bit vs 8-bit is 9–11% per layer (`requant_summary.json`). That is
  information only: the KL gate (R4) and the task checks (R5) decide quality.
- **Not tested yet:** neither model has been loaded or run.

### 16.3 KL gate and prompt variants
- **`kl_gate.py`:**
  - The reference decodes greedily one token at a time and stores full log-probs.
  - The candidate is teacher-forced in chunks.
  - It reports KL(ref‖cand), the flip rate, and "sure flips" (flips where the reference's top-2 margin is > 1 nat).
  - The control (the reference scored against itself) gives the floor.
  - On a tiny random Qwen3.5: the self-score stays below 1e-3 nats with flips only at near-ties, a perturbed copy
    lands far above it, and an off-by-one teacher-forcing bug is caught (KL 4.4).
- **`make_prompts_var.py`:** 3 different documents per length (10k, 12k, 20k, 24k) for the crossover repeats. v1 is
  byte-identical to the original prompts.

### 16.4 Batch 2 model runs, on a busy Mac with a per-run memory cap (2026-11-20, 09:49–10:31)
The user was working on other things and asked for the runs to happen here, each process capped to the memory free
at the time.
- **The cap.** `research/batch2/run.py` measures free memory before every job, as MemoryGuard does. It sets the
  footprint cap to free − 1 GB (8.5–11.5 GB over the session) and the MLX limit to cap − 0.5. The MLX cache was
  0.25 GB.
- **The guard.** It kept its swap-growth stop: 64 MB tolerance, system-wide.
- **What ran.** 49 jobs started. 48 finished with zero swap growth, peaking at ≤ 8.82 GB.
- **The one stop.** DFlash at 16k: system swap grew 79 MB (810 → 890 MB) while its footprint read 8.38 GB, and the
  guard stopped it. I stopped the rest of that phase by hand, and swap stayed at 890 MB after that.
- **Not run:** DFlash at 4k/16k, the crossover repeats (R7, 9–11 GB) and the sustained-speed run (R8, needs a quiet
  Mac).
- **Speed noise.** The Mac was in use, so speeds are compared only within interleaved runs. One plain-decoding run of
  the source model was slowed to 15.6 tok/s. The clean baseline is 17.3, from the cache check below.

**R1: DFlash with the id-set draft head** (w4:gs64, adaptive, 2 runs each in ABBA order; median [range], tok/s)

| Prompt | Plain | Id-set head (ids < 32768 + prompt ids) | Change | Acceptance plain → id-set |
|---|---|---|---|---|
| chat | 30.19 [30.14, 30.24] | 31.79 [31.53, 32.04] | +5.3% | 0.647 → 0.633 |
| math | 41.31 [41.30, 41.33] | 46.32 [46.27, 46.37] | **+12.1%** | 0.790 → 0.810 |
| code_edit | 108.88 [108.83, 108.94] | 111.22 [110.95, 111.48] | +2.1% | 0.933 → 0.933 |

- **Speed bar (≥ +5% on 2 prompts, none worse than −2%): met.** The code_edit loss of the plain 32k head (§14.5b,
  −16%) is gone.
- **Texts.** Plain runs are identical to §13's D2 text, and each variant repeats itself exactly. code_edit is identical
  across variants. chat and math split, at char 1020 and char 521.
- **The splits are near-ties.** `tie_check.py` teacher-forces the source on the common prefix:
  - chat: " cache" −1.187 vs " controller" −1.312;
  - math: " suggests" −0.801 vs " appears" −0.926;
  - so 0.125 nats at both. The id-set run took the teacher-forced top-1 both times; plain DFlash took the second.
  - This is verify-width numerics, the same effect that already makes plain DFlash differ from AR. The plan's
    "identical text" bar fails strictly on 2 of 3 prompts for that reason.
- **Absolute speeds** are 3–6% below §13's (30.2 vs 32.1 on chat) because of the concurrent load.

**R2–R4: KL gate** (reference = source model, greedy one token at a time; candidates teacher-forced in chunks of 64)

| Label | Prompts | Positions | KL mean (nats) | KL p99 | Flip rate | Sure-flip rate |
|---|---|---|---|---|---|---|
| control (source vs itself) | chat, math, code_edit | 900 | 0.0003 | 0.004 | 0.6% | 0 |
| N2-gdnhead | short | 900 | 0.009 | 0.146 | 2.1% | 0 |
| N2-all | short | 900 | 0.024 | 0.378 | 3.3% | 0.3% (math 1%) |
| control | needle_4k, needle_16k | 600 | 0.0007 | 0.005 | 1.8% | 0 |
| N2-gdnhead | 4k / 16k | 300 / 300 | 0.041 / 0.067 | 0.42 / 0.63 | 5.7 / 9.3% | 0.7 / 0.7% |
| N2-all | 4k / 16k | 300 / 300 | 0.094 / 0.124 | 0.80 / 0.94 | 10.7 / 13.0% | 2.0 / 2.0% |

KL grows with context for both variants. Over all 5 prompts (the plan's bar), N2-gdnhead is at 0.027. That is under
0.03, but the long prompts alone (0.054) sit in the "user decides" band.

**R5: speed and tasks** (2 interleaved runs per model and method, 300 tokens; median tok/s and the ratio to the
source; plain-decoding ratios use the clean 17.3 baseline)

| Method / prompt | Source | N2-gdnhead | N2-all |
|---|---|---|---|
| AR (3 prompts) | 17.3 (clean; 16.5–16.6 median in R5) | 19.9–20.0 (**+15%**) | 21.6–22.2 (+25–28%) |
| A chat | 29.73 | 34.38 (+16%) | 36.80 (+24%) |
| A math | 35.37 | 41.38 (+17%) | 45.05 (+27%) |
| A code_edit | 26.57 | 30.47 (+15%) | 32.59 (+23%) |
| A acceptance chat / math / code | 0.531 / 0.715 / 0.417 | 0.528 / 0.736 / 0.423 | 0.520 / 0.721 / 0.401 |
| needle_4k codes (AR) | 3/3 (§13) | **3/3** | **1/3** (lists code 3, then drifts to another task) |
| needle_16k codes (AR) | 3/3 (§13) | **3/3** | **0/3** |
| code_edit at 800 tokens | exact file | parses, both edits, **identical to the source's file** | same |
| peak footprint | ≤ 8.14 GB | ≤ 7.9 GB | ≤ 7.2 GB |

- **The gains track bytes per token.** Predicted (§16.2): gdnhead +18%, all +35%. Measured on plain decoding: +15%
  and +25–28%.
- **Text.** code_edit text at 300 tokens is identical to the source for both variants. chat/math split early (char
  59–151), as expected for a lossy change.
- **Distinct 4-grams.** N2-all's math scored 0.885–0.891, under the 0.9 flag. That comes from the algebra repeating
  terms ("bx + c"). The text is coherent and follows the source's derivation; the longest repeated run is 9 words.

**Cache check** (does the 0.25 GB MLX cache slow plain decoding? source, chat + math, ABBA)
- 0.25 GB: 17.26–17.37. 1 GB: 16.23–17.46.
- No slowdown, and the peak is 0.4 GB lower. The slow source run in R5 was other load.

**R6: sampling at the model card's temperature** (temperature only; optiq has no top-p/top-k/presence penalty;
seeds 0–2 × 3 prompts; median [range])

| | Greedy | Temperature 0.7 | Acceptance greedy → temperature 0.7 |
|---|---|---|---|
| AR source | 16.6 | 16.8 [16.1, 17.3] | — |
| AR N2-gdnhead | 19.9 | 19.2 [18.0, 19.5] | — |
| A source | 29.7 | 27.8 [24.3, 35.4] (−7%) | 0.531 → 0.510 |
| A N2-gdnhead | 34.4 | 32.5 [27.2, 38.0] (−5%) | 0.528 → 0.503 |

- A keeps +65% over AR at temperature 0.7.
- MTP acceptance drops 4–5%, under the plan's 15% trigger, so the sampled MTP draft (fork attempts 201–202) is not
  worth building.
- No needle prompts here, so no codes check per seed.

### 16.5 Verdicts (bars from research/batch2/plan.md, set before the runs)
- **N2-gdnhead: PASS.**
  - KL 0.027 over the 5 prompts, no sure flips on short prompts, ≤ 0.7% on long.
  - Codes 3/3 at 4k and 16k; the code file is identical.
  - Speed: AR +15%, A +15–17%, A at temperature 0.7 +17%. Weights are 0.9 GB smaller.
  - Its long-context KL is 0.054, so the user decides on adoption.
- **N2-all: FAIL.** Needle codes 1/3 at 4k and 0/3 at 16k; long-context KL 0.11. OptiQ's 8-bit attention/MLP layers
  carry long-context recall, even though N2-all is +25–28% faster.
- **DFlash id-set head:**
  - The speed bar is met (+5.3%, +12.1%, +2.1%) on short prompts.
  - It is lossless to the same degree DFlash itself is (near-tie splits only).
  - Not measured at 4k+, where the runs need 10–11 GB.
- **Not done:** R7 crossover repeats, R8 sustained speed, DFlash at 4k/16k. All need more free memory or a quiet Mac.

### 16.6 Could the memory limit explain N2-all's fail or N2-gdnhead's middle-band long KL? No for the outputs; yes for what we can conclude
**The memory cap did not change any output.** Checked on the rows themselves:
- **Original model.** All 12 AR and A texts under today's caps (10.9–11.3 GB, MLX cache 0.25 GB) are identical to
  §13's. Those ran under a 9 GB MLX limit, 1 GB cache and the 14/14 guard.
- **Cache check.** The texts are identical at 0.25 GB and 1 GB.
- **Both N2 models.** Each pair of interleaved runs at different caps (10.6–11.3 GB) gave identical text: 12 of 12 pairs.
- **Determinism.** MLX's memory and cache limits only decide when memory is freed or allocation waits. They don't
  change kernels or reduction order, so the tokens stay the same.
- **The needle runs.** Each saw its whole prompt (4,095 / 16,095 tokens). Peaks were 6.8–7.9 GB, far under their
  10.8–11.2 GB caps, with zero swap growth. The original model, through the same pipeline under the same caps (the
  KL reference at 4k and 16k), returned codes 3/3 at both lengths.
- **So** N2-all's 1/3 and 0/3 are the model's own behaviour.

**What the 24 GB machine does limit is how far the results reach:**
1. **No full-precision reference.** The bf16 9B needs ~18 GB of weights. So KL is measured against OptiQ (5.2
   bits/weight), not against the full model, and the "middle band" has no scale: we don't know OptiQ's own KL from
   bf16. A bf16 reference by layer streaming (~2–3 GB of memory; the bf16 weights are an ~18 GB download) would
   calibrate it.
2. **No calibrated quantizer.** N2 is plain round-to-nearest from the 8-bit weights. Requantizing from 8-bit adds
   little: the 8-bit step is 1/17 of the 4-bit step, so most of the error is plain 4-bit RTN. A calibrated method needs
   the bf16 weights and calibration passes. Examples: GPTQ/AWQ, or OptiQ's own sensitivity sweep at a lower bit target.
   So N2-all's fail is a fail of RTN-4bit on OptiQ's sensitive layers, not proof that 4-bit cannot work there.
3. **Runs that didn't fit.** DFlash at 4k/16k, the crossover repeats (R7) and the sustained run (R8) need ≥ 10–11 GB
   free or a quiet Mac. DFlash at 16k grew system swap by 79 MB, and the guard stopped it.

**Not memory, but limits too:**
- KL covers 1,500 positions across 5 prompts.
- The needle verdicts rest on one document per length.
- `research/batch2/prompts_var.json` has 3 documents each at 10k/12k/20k/24k. AR on the N2 models peaks ~7–8.5 GB
  there, so it fits today's caps and would firm up both verdicts.
- Speeds were measured on a Mac in use, interleaved, so only the ratios hold.

## 17. Batch 3 (research/batch3/plan.md, 2026-11-21, 04:02–05:07)
How it ran: `research/batch2/run.py`, one job at a time, cap = free − 1 GB measured before each job (11.4–13.8 GB),
MLX cache 0.25 GB. New swap rule (2026-11-21): system swap may reach 3 GB in total, none preferred; the
guard now allows growth up to 3 GB − swap at job start (`GUARD_SWAP_GB`). Swap stayed at 1.14 GB: **zero growth in
all 17 jobs**. Peaks ≤ 10.0 GB. Every speed row: full 300 tokens (or a natural stop), wall/reported 0.99–1.00,
distinct 4-grams ≥ 0.956.

### 17.1 T1: needle codes on 3 documents per length (`research/batch3/t1_needle.jsonl`; AR, greedy)
| Length | Source | N2-gdnhead | N2-all |
|---|---|---|---|
| 10k v1/v2/v3 | 3/3/3 | 3/3/3 | 3/3/3 |
| 12k v1/v2/v3 | 3/3/3 | 3/3/3 | 3/3/3 |
| 20k v1/v2/v3 | 3/3/3 | 3/3/3 | 3/3/3 |
| 24k v1/v2/v3 | 3/3/**0** | 3/3/**0** | 3/3/**0** |

- **`needle_24k_v3` is not a broken prompt** (correction, 2026-11-21 audit): all three codes are in it (at 17%, 50%
  and 87% of the text) and both tasks follow the document. Only the document ends mid-table, right before "Task 1".
  All models, the source included, skip Task 1 and analyse that table. A hard case that no model passes, so it can't
  separate them; it is not an invalid prompt.
- **Bars (set before the run):** N2-gdnhead keeps its PASS (3/3 wherever the source is 3/3). N2-all's FAIL stands as
  written (11/12), but its only miss is the broken document: on codes it equals the source at 10k–24k. Batch 2's
  1/3 (4k) and 0/3 (16k) were one document each, so N2-all's recall loss is document-dependent.
- **What the codes count misses:** the source did both tasks on all 11 valid documents. N2-gdnhead dropped task 2 in
  2/11 (a "safety" refusal at 10k v1 that still lists the codes; codes-only at 20k v1); N2-all in 3/11 (refusals at
  12k v1 and 20k v1, codes-only at 24k v1). The 4-bit layers loosen instruction following before recall.

### 17.2 T3: DFlash id-set head at 4k/16k (`research/batch2/r1_dflash.jsonl`, jobs `dflash-long-*`; ABBA)
| Prompt | Plain tok/s | Id-set head | Change | Texts |
|---|---|---|---|---|
| needle_4k | 24.81 [24.39, 25.22] | 26.96 [26.90, 27.02] | **+8.7%** | identical, = §13 |
| needle_16k | 21.52 [21.42, 21.62] | 22.78 [22.30, 23.25] | **+5.8%** | split at char 782 |

- Codes 3/3 in all runs; peak 10.0 GB. Draft head rows 32,952 / 33,333.
- **The 16k split is a near-tie:** source log-probs " Bench" −1.907 vs " Context" −2.032 (0.125 nats, as for chat
  and math in §16.4); the id-set run again took the top-1.
- `tie_check.py` now prefills in chunks of 512 (one pass at 16k would build ~8 GB of logits) and takes `ONLY`. It
  reproduced §16.4's math values exactly; chat moved ~0.1 nats with the same top-1.
- **Verdict: PASS.** With §16.4, the id-set head gains on all 5 prompts (+2.1% to +12.1%) and splits only at near-ties.

### 17.3 T4 (R7): crossover repeats on 3 documents (`research/batch3/t4_cross.jsonl`; 2 runs per method, ABBA)
Change = faster method vs the other, median tok/s [range].

| Doc | 10k: D2 vs A | 12k: D2 vs A | 20k: Aq4 vs A (bf16) | 24k: Aq4 vs A |
|---|---|---|---|---|
| v1 | +0.5% (ranges overlap) | −2.3% | −2.5% | +6.5% |
| v2 | −4.7% | −5.0% | +3.2% | +7.2% |
| v3 | −0.6% (ranges overlap) | −1.3% | +6.1%, **codes 0/3** | +8.2% (broken doc) |

Absolute medians: A 22.2–23.4 (10–12k), 22.0–23.4 (20–24k); D2 21.6–22.6; Aq4 22.5–24.8. Texts repeat exactly per
method; codes 3/3 everywhere except below.
- **10k:** A faster in 2/3, two of three within noise → §14.5a's "tie at 10k" stands; D2's lead ends at or before 10k.
- **12k:** A faster in 3/3 → "A wins from 12k" **stands**.
- **20k:** Aq4 faster in 2/3 → §14.6's "bf16 A wins at 20k" **does not stand**; the bf16/4-bit KV crossover is at or
  below 20k on speed.
- **24k:** Aq4 faster in 3/3 → **stands**.
- **Quality: Aq4 failed the task on `needle_20k_v3`** (a valid document; bf16 A lists the 3 codes). It claims the codes
  "are not included in the provided text". Second task failure on the 4-bit KV path (after §14.6 at 32k). Its speed
  win at 20k comes with that risk, so `LONG_CONTEXT = 24_000` stays the safer threshold.

### 17.4 T7: GPTQ instead of rounding for N2-all's layers (`research/batch3/gptq_stream.py`, 2026-11-21 09:50–10:22)
**Why mlx-lm's own tools don't fit** (read from mlx-lm 0.31.3's source):
- `quant.gptq` keeps every layer's Hessian at once (~19 GB for a 9B) and needs a float model.
- `quant.dynamic_quant` loads the float model whole (18 GB).
- `quant.awq` streams layers but has no `qwen3_5` config.
- `quant.dwq`'s memory is unknown.
- `quant.gptq`'s in-group update (`W[..., k:k+j]` with an absolute `k`) also writes columns past the group, which the
  block update then corrects a second time. The script here follows the paper's lazy-batch form instead.

**What was built:**
- `Qwen3.5-9B-OptiQ-4bit-N2-all-gptq`: the same 133 layers as N2-all (every 8-bit layer but embed_tokens) at 4-bit,
  group 64. The bytes per token are N2-all's (4.47 GB).
- **Weights:** the original bf16 `Qwen/Qwen3.5-9B` (downloaded 2026-11-21 with aria2: 19.3 GB, sizes match the Hub,
  headers checked; the SHA-256s were not checked).
- **Calibration:** mlx-lm's `calibration_v5` text, 128 × 512 tokens, seed 123.
- **Sequential:** each decoder layer's Hessians come from the output of the layers already quantized. OptiQ's 4-bit
  layers are kept as they are. The LM head's Hessian comes from the final norm, and the head is done in row chunks.
- **Build:** 30 min, peak 5.4 GB, zero swap growth. `requant.py check` (exact=False) found 0 problems: config,
  shapes, the official parser, and untouched tensors byte-identical.
- **Weight error:** relative to OptiQ's 8-bit weights it is larger than rounding's (median 0.126 vs 0.095). GPTQ
  minimizes the layer's output error, and starts from bf16, not from the 8-bit values.

**Tests:** `research/batch3/test_gptq_stream.py`, 13 tests, plus the 6 in `test_requant.py` = 19 pass.
- Six deliberate breaks were each caught by at least one test: no in-group feedback, no cross-group feedback, codes
  packed in the wrong order, the wrong mask on GDN layers, stage files ignored, dead input columns not handled.
- One rule is not covered: each layer must see the *quantized* output of the layers before it.
- The match against a float64 column-by-column reference is "rows agree until a rounding near-tie (< 0.005)". The
  first form of the test (≥ 99.5% of codes equal) failed at 99.3%, from float32 drift: with the same scales, every
  row parts ways only at a tie, while the broken version parts ways in all 48 rows.

**Results** (batch-2 bars, set before the runs; reference = the OptiQ source, so the KL scale is uncalibrated, §16.6):

| | KL short (floor 0.0003) | KL long (floor 0.0007) | sure flips long | needle 4k | needle 16k | 12 documents | code file |
|---|---|---|---|---|---|---|---|
| N2-gdnhead | 0.009 | 0.054 | 0.67% | 3/3 | 3/3 | 33/36 | identical to source |
| N2-all (rounding) | 0.024 | 0.109 | 2.0% | 1/3 | 0/3 | 33/36 | parses, both edits |
| **N2-all-gptq** | **0.019** | **0.072** | **1.3%** | **3/3** | **0/3** | 33/36 | parses, both edits, not identical |

- **Short-prompt KL:** chat 0.039 → 0.026 and math 0.033 → 0.024, but code_edit 0.0016 → 0.0066.
- **Long-prompt KL:** needle_4k 0.095 → 0.057, needle_16k 0.124 → 0.086.
- **Speed** (`research/batch3/t7_speed.jsonl`, one ABBA pair against N2-all): AR 22.0 vs 21.7 tok/s and A
  32–42 vs 32–43 tok/s. The same as N2-all, as the bytes are the same. The batch-2 source: AR 16.6, A 29.7.
- **On the 16k needle prompt** it skips Task 1, as N2-all does: an executive summary, no codes. It drops Task 2 on
  the same three of the 12 documents as N2-all (refusals at 12k v1 and 20k v1, codes only at 24k v1).

**Verdict: FAIL** under the batch-2 bar, since a task fails (needle 16k 0/3).
- GPTQ removes about a third of rounding's long-context KL, from 0.109 to 0.072. That is below the 0.08 fail line
  and close to N2-gdnhead's middle band.
- It fixes the 4k needle but not the 16k one, and not the dropped tasks.
- So §16.6's "N2-all's fail is RTN's, not 4-bit's" holds only in part. A calibrated quantizer narrows the gap, but
  4-bit on the full-attention and MLP 8-bit layers still costs long-context instruction following on this model.
- N2-gdnhead stays the one to adopt.

### 17.5 T2: middle variant with GPTQ (`Qwen3.5-9B-OptiQ-4bit-N2-gdnhead-mlp-gptq`, 2026-11-21 10:22–11:08)
**What was built:**
- `requant.py` variant `gdn_head_mlp`: every 8-bit layer but full attention, i.e. 89 GDN layers, 19 MLP layers and
  the LM head = 109 layers. The 24 full-attention layers stay 8-bit.
- The test `test_gdn_head_mlp_converts_all_but_full_attention` failed first (unknown variant).
- `gptq_stream.py --variant` was added. Its test failed when the flag was ignored; 21 tests pass.
- Built with GPTQ from the bf16 weights, same settings as §17.4: 24 min, peak 5.2 GB, zero swap growth.
  `check` found 0 problems.
- **Bytes read per token:** 4.61 GB, a 29.1 tok/s limit. N2-gdnhead reads 5.09 GB (26.4 tok/s) and N2-all
  4.47 GB (30.0 tok/s).

**Results:**

| | KL short | KL long (4k / 16k) | p99 long | needle 4k | needle 16k | 12 docs | task 2 dropped | code file |
|---|---|---|---|---|---|---|---|---|
| N2-gdnhead | 0.009 | 0.054 (0.041 / 0.067) | 0.58 | 3/3 | 3/3 | 33/36 | 2/11 | identical |
| N2-all-gptq | 0.019 | 0.072 (0.057 / 0.086) | 0.67 | 3/3 | 0/3 | 33/36 | 3/11 | not identical |
| **Middle variant (GPTQ)** | **0.011** | **0.067 (0.033 / 0.101)** | **1.20** | **3/3** | **0/3** | 33/36 | **0/11** | **identical** |

- **Sure flips:** 0.33% short, 0.83% long (1.3% at 16k).
- **Speed** (`research/batch3/t2_speed.jsonl`, ABBA against N2-gdnhead and the source; every row has 300 tokens and
  wall ≈ runtime; the ranges don't overlap):
  - middle variant: AR **+23–24%** (21.2–21.4 tok/s), A **+21–29%** (32.7–42.8 tok/s), MTP acceptance 0.58 vs 0.53;
  - N2-gdnhead in the same runs: AR +14–16%, A +14–17%;
  - source: AR 17.2–17.3, A 26.6–35.3 tok/s.

**Verdict: FAIL** under the batch-2 bar, since one task fails (needle 16k, 0/3).
- On everything else it is the best N2 model: best KL at 4k, no dropped Task 2 on the 11 valid documents, the code
  file byte-identical to the source's, and +23% over the source.
- Its one miss: at 16k it answers "Based on the research handoff…" and never lists the codes, the same Task-1 skip
  N2-all shows there and every model shows on `needle_24k_v3`. §17.6 shows this is not a near-tie.
- Its 16k KL tail is the heaviest (p99 1.9).

### 17.6 Was the 16k needle miss a near-tie? No (`research/batch3/first_token.py`, `branch_16k.py`, 11:20–11:31)
**First-token margins** (`first_token.jsonl`): the model's top 10 for the answer's first token, after prefilling the
prompt.
- The first token decides nothing. The source's own top 1 at 16k is "Based" (margin 0.25 nats over "**"), and its
  recorded 16k answers all start "Based on the provided document, here are the three secret codes…".
- On `needle_24k_v3` every model's top 1 is "Based" by 2.4–4.5 nats.
- At 4k every model's top 1 is "**" (the codes).
- So my first reading in §17.5, "decided on the first token", was wrong.

**Branch check** (`branch_16k.jsonl`): the source's own first 40 answer tokens at 16k (AR, `dflash/baseline.jsonl`)
are teacher-forced through each model. Where the model's argmax leaves that answer, the gap is in nats between its
choice and the source's token.

| Model | agrees | first departure | gap there | gap at "Based on the provided **document**" |
|---|---|---|---|---|
| source (control) | 39/40 | pos 5: " and" vs "," | 0.25 (a tie) | 0 |
| N2-gdnhead | 35/40 | pos 0: "###" vs "Based" | 0.75 | 2.4 |
| middle variant (GPTQ) | 33/40 | pos 3: " research" vs " provided" | 0.125 | **6.7** |
| N2-all-gptq | 32/40 | pos 0: "Here" vs "Based" | 0.25 | **6.2** |
| N2-all | 36/40 | pos 4: " research" vs " document" | **6.9** | **6.9** |

**Reading:**
- The decisive token is the 5th: after "Based on the provided", the source says " document", which goes on to the
  codes. The three failing models put " document" 6.2–6.9 nats (about 1/1000) below " research…", which goes on to
  an analysis of the research logs.
- The middle variant's own departure at position 3 is a near-tie (0.125), but it doesn't matter: from either word it
  goes to " research".
- **The FAIL is fair:** the 4-bit MLP (and full-attention) layers shift this choice by a wide margin, not by
  chunk-width noise.
- **N2-gdnhead's 16k pass is fragile.** It too prefers something other than " document" (2.4 nats). It passes only
  because its first token "###" wins by 0.125 nats over "Here" (`first_token.jsonl`) and that path leads to
  "### Secret Codes". Its PASS stands under the bar, but it rests on a near-tie at 16k. Treat its long-context recall
  as weaker than the source's.
- The KL gate saw this too: long KL at 16k ranks the models in the same order as the gap (gdnhead 0.067, all-gptq
  0.086, middle 0.101, all 0.124), though the middle variant's 4k KL is the best.

### 17.7 Not run
- T5: sustained speed. It needs a quiet Mac and the user's OK.
- T6: a bf16 reference for the KL gate. The weights are on disk; the layer-streamed reference script is not written.

## 18. Batch 4 heavy run (research/batch4/plan.md, 2026-11-21, 23:32–23:58)
How it ran: `research/batch4/heavy.py`, one unattended chain, 25.5 min. 14 jobs ok, H4 "not run (condition)" (no M1
yet). A first try at 23:30 skipped every job (swap 3.36 GB, 9.2 GB free, another project's 5.6 GB process); after it
was stopped, swap was 1.79 GB and free 15.6–16.7 GB. Swap growth was measured in only 2 of 14 jobs (0.0 GB in both
optiq-AR jobs; the runner dropped the other jobs' guard reports, fixed after the run with `research/batch2/test_run.py`);
no job was stopped by the guard, so growth stayed under its limit, but it is not shown to be 0. Peaks 6.9–8.5 GB. Every speed row:
full 300 tokens; wall ≈ reported tok/s checked only where a row has `wall_tok_s` (the 8 optiq-AR rows, ratio ≈ 1.00);
the other 40 H2 rows keep only `total_s`, which includes prefill, so they bound tok/s from below; repeats of a config byte-identical, distinct
4-grams ≥ 0.964, texts read and coherent.

### 18.1 H1 (E10): sustained decode (`research/batch4/h1_sustained.jsonl`, `h1_macmon.jsonl`)
Setup A as shipped (N2-gdnhead, MTP depth 2), `chat`, 69 back-to-back 300-token generations in 607 s, one process.

| | tok/s median | GPU power | GPU clock | GPU busy | GPU temp | Chip power | J/token (chip) |
|---|---|---|---|---|---|---|---|
| First minute (6 gens) | 34.73 | 13.9 W | 1592 MHz | 0.92 | 67 °C | 20.5 W | 0.59 |
| Last minute (7 gens) | 34.62 | 14.2 W | 1592 MHz | 0.91 | 79 °C | 20.8 W | 0.60 |

- **Energy per token** (median over the run, 34.9 tok/s): GPU alone 0.41 J; chip (CPU + GPU + RAM, macmon
  `all_power`) 0.59 J; whole system (macmon `sys_power`, median 42–43 W in the two windows, 31–49 W over the run; includes the display
  and the rest of the Mac)
  1.23 J. The power windows are macmon's first and last 60 s, so the first one includes the ~9 s model load.

- **Last/first = 99.7% → PASS** (bar: ≥ 90%). Range over all 69: 33.49–35.50; acceptance 0.528 in every run.
- The GPU warmed 12 °C (median 66.6 → 78.6) with no clock drop (median 1592 MHz in both windows, minimum 1573): no
  throttling on one 10-minute `chat` decode. On this evidence E10 (power cap) has nothing to recover for the 9B;
  a longer or hotter run (other prompts, a warm room) could still show throttling.

### 18.2 H2: packaged wins on the 9B (`research/batch4/h2_packaged.jsonl`; ABBA × 2, median of 2)
| Prompt | optiq AR | `generate_ar` N2 off | N2 on | A N2 off | A N2 on | A + id-set N2 on |
|---|---|---|---|---|---|---|
| chat | 17.40 | 19.84 | 23.20 | 29.95 | 34.25 | 34.67 |
| math | 17.33 | 19.95 | 23.21 | 35.48 | 41.22 | 41.98 |
| code_edit | 17.40 | 19.87 | 23.18 | 26.73 | 30.75 | 30.28 |
| needle_4k | 17.13 | 19.42 | 22.62 | 23.80 | 29.09 | 29.22 |

**(1) `generate_ar` vs optiq AR (N2 off): speed PASS, texts FAIL.** +14.0 / +15.1 / +14.2 / +13.3% (bar ≥ +7%; more
than §14.3's +10–11%). Texts identical only on code_edit; chat, math and needle_4k split at chars 273 / 355 / 392.
**Unverified (reported by the batch-4 child session; its `tie_check.py` and output were not kept; re-check pending):**
`tie_check.py` on the source (teacher-forced, chunked prefill): chat " (" and " cores" tie at −1.093, optiq's " and"
−1.218 (0.125 nats); needle_4k " installation" and " benchmark" tie exactly at −1.187; math not resolved (the tool cut
the context back past the split). So the splits are bf16 near-ties, most likely from the different prefill
(`generate_ar` prefills `ids[:-1]` in chunks of 1024, then the last id alone). The bar as written ("texts
identical") is not met; the module's "same tokens" claim holds only up to such ties.

**(2) N2-gdnhead on vs off: PASS.** AR +17.0 / +16.4 / +16.7 / +16.5%; A +14.4 / +16.2 / +15.0 / +22.2% (bar ≥ +11%).
Texts differ (another model) except code_edit; codes 3/3.

**(3) A + id-set vs A (N2 on): FAIL.** +1.2 / +1.9 / −1.5 / +0.4%. Lossless (texts identical on all 4) and above −2%
everywhere, but no prompt reaches +3% (code_edit −1.5%, needle_4k +0.4%). Batch 2–3's +8.7% at needle_4k was on
DFlash (§17.2), not on MTP, and does not carry over. **`draft_prompt_ids` stays default off.**

**(4) needle_4k codes 3/3 in all 12 runs: PASS.** 10 rows carry `codes`; the 2 optiq-AR rows (`baseline_child.py`
writes no `codes`) were recounted from their text: 3/3.

Setup A + N2 on over optiq AR: ×1.97 chat, ×2.38 math, ×1.77 code_edit, ×1.70 needle_4k.

### 18.3 H3: slow test
`tests/test_model_slow.py` (≥ 20 tok/s at 4k, codes 3/3, no loops; N2 path by default): exit 0 in the chain (29 s).
The runner does not keep pytest's output, so exit 0 cannot tell a pass from a skip. The child session reports a hand
re-run with `-rA`: **1 passed** (23 s), but that output was not kept. **Unverified; re-run with the output saved
pending** (needs ~10 GB free).

### 18.4 Not run
- H4 (E9 int8 prefill): condition false (`M1_PASSED` / `h4_e9_prefill.py` absent). M1 is next for E9.
