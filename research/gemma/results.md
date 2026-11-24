# Gemma 4 E2B + its `-assistant` MTP drafter: optiq's loop, its profile, adaptive gamma (2026-11-24)

Setup: the M1 test Mac (Apple M1, 16 GB), mlx 0.32.4 / mlx-lm 0.32.2 / mlx-optiq 0.8.5. Target `gemma-4-e2b-it-4bit`
(mlx-community, uniform int4) or `gemma-4-e2b-it-OptiQ-4bit`; drafter `gemma-4-E2B-it-assistant-bf16` (181 MB).
Prompts: `research/batch5/ab/prompts_text.json` (chat 35 tokens, math 111, code edit 752, needle 4,184), chat template,
256 greedy tokens, stop at EOS. One model per process, loaded before optiq is imported (otherwise the load peak hits
11 GB and MemoryGuard stops the run, which it did twice). Caps: MLX 5.2 GB, footprint 7.2 GB; MemoryGuard reported
peaks of 4.1 GB (int4) and 5.6 GB (OptiQ) and swap growth 0 (run logs, not kept here).

What the assistant is: Google's MTP drafter for Gemma 4 (4 layers, hidden 256, Q-only: it reads K/V from the target's
last sliding and last full-attention layers, and drafts from the target's hidden state; a 2,048-centroid output head
scores only the top-32 centroids' tokens). optiq runs it with `optiq serve --drafter` through
`optiq.runtime.spec.assistant_generate_step`, gamma fixed at 2.

## 1. optiq's drafter vs plain and fastpath plain decode (`gemma_assistant_ab.py`, `ab_first_*.jsonl`)
Decode tok/s after the first token, ABBA, median of 2. plain = mlx-lm `generate_step`; fast-ar = `fp.generate_ar`.

| Weights | Prompt | plain | gamma 1 | gamma 2 | gamma 3 | fast-ar |
|---|---|---|---|---|---|---|
| int4 | chat | 32.3 | **45.9** | 41.6 | 36.7 | 39.3 |
| int4 | math | 32.4 | **52.0** | 50.3 | 48.2 | 39.0 |
| int4 | code edit | 31.7 | **49.6** | 48.7 | 46.1 | 38.3 |
| int4 | needle 4k | 29.9 | **42.7** | 38.8 | 35.2 | 36.2 |
| OptiQ | chat | 24.6 | 37.5 | **40.3** | 37.3 | 29.7 |
| OptiQ | math | 24.6 | 41.3 | **45.2** | 44.8 | 29.7 |
| OptiQ | code edit | 24.4 | 40.2 | **46.0** | 44.6 | 29.2 |
| OptiQ | needle 4k | 23.5 | 37.1 | **40.1** | 37.9 | 28.0 |

- At its best gamma the drafter beats fastpath's plain decode on every prompt (1.42-1.88x plain vs 1.19-1.22x);
  int4 at gamma 3 does not (chat 36.7 vs 39.3, needle 35.2 vs 36.2).
- The best gamma depends on the weights: 1 on int4, 2 on OptiQ. With optiq serve's fixed 2, int4 loses speed: gamma 1 is 1.8-10.5% faster in this table (1.3-11.4% over all 26 runs, §4).
- Acceptance at gamma 1: 69-93%. Every run's two repeats gave the same tokens. Splits from plain decoding sit at
  plain-decoding top-2 margins of 0.0-0.25 (logprob), i.e. close calls: int4 needle at token 140 (gamma 1), 123
  (gamma 2), 158 (gamma 3) and 221 (fast-ar); int4 chat gamma 2 at 195; OptiQ math at 214; other gamma-3 runs at
  66-158. Not checked against a reference beyond that margin.

## 2. Where a pass spends its time (`loop_profile.py`, `loop_profile.jsonl`)
Phase costs on a warm cache at each prompt's context, median of 15: verify(n) = target step with n tokens; draft(g)
= g drafter steps, each read back to Python (optiq) or chained on the GPU (lazy). Real loop: optiq's
`assistant_generate_step`, 128 tokens.

| Weights, prompt | verify 1 / 2 / 3 / 4 ms | draft 1 / 2 / 3 ms |
|---|---|---|
| int4, chat | 30.8 / 32.0 / 52.1 / 65.1 | 2.3 / 3.6 / 4.9 |
| int4, code edit | 28.6 / 37.0 / 49.5 / 68.3 | 2.4 / 4.5 / 7.0 |
| int4, needle | 30.3 / 40.7 / 51.4 / 68.7 | 3.2 / 6.4 / 12.4 |
| OptiQ, chat | 40.0 / 44.1 / 48.2 / 67.0 | 1.9 / 3.7 / 5.4 |
| OptiQ, code edit | 37.2 / 47.1 / 57.3 / 69.0 | 2.7 / 4.2 / 6.7 |
| OptiQ, needle | 43.3 / 43.7 / 59.0 / 74.0 | 1.9 / 3.8 / 5.7 |

- draft(g) + verify(g+1) predicts the real per-pass time within -6 to +4 ms in all 18 cases: optiq's loop has no
  hidden overhead worth removing.
- Keeping drafts on the GPU (one read-back per pass instead of g) saves 0-1.3 ms per pass (one outlier 7.4 ms on int4
  needle gamma 3 from a noisy 12.4 ms draft(3)). About 2%: not built.
- The cost is the verify width. On chat it jumps at a different width per weights: int4 +63% from 2 to 3 tokens,
  OptiQ +39% from 3 to 4. On code edit and needle the steps vary (x1.01-1.38 per extra token). Which gamma is best
  is an empirical result (§1 table), not derived from these costs.

## 3. The verify cliff (`cliff_bench.py`, `cliff_bench.jsonl`)
Every distinct quantized weight timed alone on (1, n, in) bf16 inputs, n = 1..6 (median of 15), and the whole target
step at n tokens on a 512-token cache. Per-op times include launch and sync overhead (their sum is several times the
step), so only ratios between widths are read from them. Gemma's `embed_tokens_per_layer` row is a lookup table in the
real model, timed here as a matmul by mistake: ignored.

| Model | step 1 / 2 / 3 / 4 tokens (ms) | what rises from 2 to 3 rows |
|---|---|---|
| Gemma E2B int4 | 31.2 / 39.0 / 53.1 / 68.6 | most 4-bit matmuls rise x1.1-1.5 (MLP up/gate 12288x1536 x40: x1.22; head 262144x1536: x1.49); a few are flat or lower (down 1536x12288: x0.91; 512x1536, 256x1536: ~x1.0) |
| Gemma E2B OptiQ | 37.2 / 46.4 / 57.0 / 61.4 | mixed: the 8-bit embedding/head and small 8-bit layers are flat to 3 rows (x0.97-1.04), but several large 8-bit layers rise at 3 (12288x1536 x1.27, 1536x2048 x1.31, 8960x1536 x1.17) |
| Qwen3.5-2B int4 | 26.7 / 29.1 / 42.4 / 52.5 | 4-bit MLP up/gate 6144x2048 (x66) ~+9 ms, head ~+2.4 ms |

- On int4 the step's jump from 2 to 3 tokens (+14 ms Gemma, +13 ms Qwen) comes from many 4-bit matmuls getting slower
  at 3 rows on this M1, spread over the model, not one op.
- OptiQ: this bench (512-token context) shows its step rising less from 2 to 3 (+23%) and little from 3 to 4 (+7.6%),
  while the loop profile (§2, the prompts' own contexts) measured +39% from 3 to 4 on chat. The two disagree on
  where OptiQ's cliff is; the per-op data does not single out its 8-bit layers as the cause. Not resolved here.
- Not measured here: whether splitting a 3-row matmul into 2 + 1 calls helps (an earlier M5 study in this repo,
  research/kmic68, found it slower there). A fix would need a custom 4-bit kernel for 3-4 rows: not attempted.

## 4. Choosing gamma per pass (`mlx_fastpath/assistant.py`)
The loop is optiq's, with gamma chosen per pass; output stays plain greedy decoding (every draft verified). Five
versions were measured on the same 8 cells (E2B int4 and OptiQ x chat, math, code edit, needle; one process per
weights, median of 2 repeats each, so every cell is a single run):

| Version | Rows | vs best fixed gamma (int4: max(gamma 1, gamma 2); OptiQ: gamma 2, the best in §1) (min-max, mean) | vs optiq's gamma 2, mean (int4 / OptiQ) |
|---|---|---|---|
| v1: calibrated costs, probe only at gamma 0 | `ab_adaptive_v1_*.jsonl` | 0.943-1.008, 0.993 | 1.025 (1.065 / 0.986) |
| v2: + one-deeper probe every 8th pass (cache restore timed by mistake) | `ab_adaptive.jsonl` | 0.968-1.036, 0.996 | 1.031 (1.070 / 0.991) |
| v3: restore no longer timed | `ab_adaptive_v3_calibrated.jsonl` | 0.945-1.020, 0.982 | 1.017 (1.066 / 0.968) |
| v4: costs learned from real pass times (kept, opt-in) | `ab_adaptive_v4_learned_costs.jsonl` | 0.948-0.999, 0.973 | 1.004 (1.035 / 0.974) |
| v5: choose by measured tokens/ms per gamma | `ab_adaptive_v5_throughput.jsonl` | 0.856-1.008, 0.938 | 0.971 (1.016 / 0.926) |

- On average none beats the best fixed gamma (single cells do, up to 1.036). v1-v4 are not ranked against each other: each cell is one run, and one cell
  (int4 math) went from 1.008 to 0.961 of best between v2 and its recheck after the 0.0.21 fixes (similar, not
  identical code), a swing as large as the differences between versions. v5 is worse in every summary column; a likely
  cause (not tested): a pass yields 1-4 tokens, so a few passes per gamma give a noisy throughput.
- The calibrated cost model picked badly in places (v3: gamma 3 on OptiQ chat in 94 passes, 5.5% below gamma 2).
  Why is not established; the loop profile (§2) found real passes close to draft + verify cost, so a large hidden
  per-pass overhead is not shown. v2's accidental extra cost (timing a cache copy) did better; also not explained.
- What is robust: the right FIXED gamma. On E2B int4, gamma 1 beat optiq's fixed gamma 2 in every one of the 26
  cell-runs in research/gemma that have both (`ab_first_int4`, all `ab_adaptive*` files), by 1.3-11.4% (mean 6.7%). On OptiQ, gamma 2 (what `optiq serve` hardcodes; the function's own default is 1) is the best fixed choice.
- Output: same tokens as plain decoding except close calls (plain top-2 margins 0.0-0.25). Split positions vary
  between versions and runs (e.g. adaptive int4 needle at 99 in v2-v4, 132 in v5; adaptive OptiQ math at 96 (v3), 118 (v2, v4, v5) and 214 (v1)), and a few v4/v5 cells
  had repeats that disagreed with each other: close calls tipping either way, not checked against a reference.
- The per-pass timing subtracts the time the consumer spends between yields; the GPU may keep working during it
  (async eval), so a very slow consumer could make passes look slightly cheaper. Not an issue for the server path.
- Memory (opt-in "auto" only): the cost measurement holds a copy of the whole cache while it runs, and the cost store
  keeps a reference to the drafter for the model's lifetime. Not measured at long contexts.
- Bugs found on the way, each now covered by a test that fails without the fix: cost calibration left the
  sliding-window ring buffer changed (rollback cannot undo a 1-token in-place write; then the snapshot's own arrays
  were handed back); costs cached by id() could leak to a new model; the context bucket was fixed per request.

## Not done (against the session plan)
- Planned repeat processes on the swingy cells (to fix the calibration noise) were not run: every adaptive cell is a
  single process run (median of 2 repeats); the only repeat is the recheck after the 0.0.21 fix, on 4 cells.
- A dispatch-level fix for the 4-bit cliff (custom kernel for 3-4 rows) was not attempted.

## Use
`fp.apply()` leaves optiq's assistant loop alone (default). `fp.apply(assistant_gamma=1)` makes every caller of
`optiq.runtime.spec.assistant_generate_step`, including `optiq serve --drafter`, draft 1 token per pass (recommended
for E2B int4 on M1); `assistant_gamma=2` / `3` likewise; `assistant_gamma="auto"` is the per-pass choice (v4), opt-in.
`fp.remove()` restores optiq's loop; a later `fp.apply()` without the argument does not undo an earlier setting. Not measured: Gemma 4 12B / E4B, the M5, sampling, contexts above 4k.
