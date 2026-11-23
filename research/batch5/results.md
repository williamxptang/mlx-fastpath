# Batch 5: re-validating mlx-fastpath on the upgraded stack with small models (2026-11-22)

Branch `small-models` (worktree `.worktrees/small-models`), cut from `main` @ 1d6ce8d (v0.0.8). Committed as v0.0.9 (e0648db) after the checks below; audit fixes in v0.0.10.
This file records everything done in this batch, including the mistakes. Times are local (UTC+8).

## Goal
- The MLX stack was upgraded today, so `apply()` refuses the new versions and runs stock. The re-validation is
  needed before `apply.TESTED` can move.
- Waiting for ~10 GB to test on Qwen3.5-9B blocks the work, so correctness is now checked on small models first.
  The 9B speed check stays the last step before main.
- User's choices: Qwen3.5 2B + 0.8B + Gemma 4 E2B, the base (bf16) originals, downloaded with aria2. Later: use
  prebuilt OptiQ copies where they exist. The 0.8B bf16 is also a row in the comparison grid (see "Next").

## Stack
| Package | Before | Now |
|---|---|---|
| mlx / mlx-metal | 0.32.3 | 0.32.4 |
| mlx-lm | 0.31.3 | 0.32.2 |
| mlx-optiq | 0.8.1 | 0.8.5 (`--no-deps`: it pins transformers <5.13; transformers kept at 5.15.0) |
| mlx-vlm | 0.6.13 | 0.8.0 |
| mlx-audio | 0.5.8 | 0.6.7 |
| jang | 2.5.46 | 2.5.53 |
| vmlx | 1.6.28 | 1.6.28, unchanged (1.6.84 wants mlx-vlm <0.6 on Darwin, so it was held back) |

Full freezes: `inputs/mlx-env-freeze-before-upgrade-20261122.txt`, `inputs/mlx-env-freeze-after-upgrade-20261122.txt`.
**Correction (plan-conformance audit):** the first version of this table had a wrong "Before" column for mlx-vlm
(—), mlx-audio (0.6.4), jang (2.5.26) and vmlx (1.5.23). I wrote it from an older note instead of the freeze. The
table above now matches the freeze file.

Patch points re-checked on mlx-lm 0.32.2:
- `qwen3_5` still takes its attention class from `qwen3_next`, and `qwen3_next.scaled_dot_product_attention` is
  still the name attention.py swaps.
- The optiq hooks in draft.py and prefill.py still exist under the same names.

## Models
| Model | Source | Use |
|---|---|---|
| Qwen3.5-0.8B-bf16, Qwen3.5-2B-bf16, gemma-4-E2B-it-bf16 | aria2, pinned revisions; every file's size checked, sha256 checked for the LFS (weight / tokenizer) files only (`inputs/aria2-small-bf16.manifest.json`) | originals; KL / quality reference |
| Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed | built here: `optiq convert`, 24 samples, target 5.0 bpw → 130 layers 4-bit, 57 layers 8-bit, MTP sidecar | test model |
| Qwen3.5-0.8B-OptiQ-4bit/uniform_4bit | side output of the same convert (plain 4-bit + MTP sidecar) | "normal int4" row of the grid |
| Qwen3.5-2B-OptiQ-4bit | mlx-community @ 1927cf4a4f, 14 files, vision sidecar left out, size checked for every file and sha256 for the LFS files (`inputs/aria2-small-optiq.manifest.json`) | test model |
| gemma-4-e2b-it-OptiQ-4bit | mlx-community @ 977e9a6008, 11 files, vision sidecar left out, size checked for every file and sha256 for the LFS files | test model (no MTP: AR path) |

**Mistake: I didn't check for prebuilt models.** Why not just download OptiQ copies? mlx-community
publishes OptiQ copies of all three small models, and I didn't look before converting. A download would have cost
minutes and no memory.

## Log
- **14:56** `pip freeze` saved, then the stack upgrade (dry run first, then install).
- **15:25–15:45** bf16 originals downloaded with aria2. Gemma timed out at 9.36/10.25 GB, was resumed with `-c`, and
  then verified.
- **~17:15** Branch `small-models` created. New test files (not run yet):
  - `tests/real_model.py`: memory caps from the model's size on disk. MLX limit = weights + 1.2 GB, footprint cap
    +2 GB, free memory needed +1 GB. This gives 8/10/11 GB for the 9B, the values the old slow test hard-coded.
  - `tests/test_model_slow.py`: any model via `MLX_FASTPATH_MODEL`; the 20 tok/s floor applies only to the 9B.
  - `tests/test_lossless_slow.py`: the lossless test (below).
- **17:20 First 0.8B `optiq convert`: run without a memory cap. Mistake.**
  - I told the user 2–5 GB. It reached an 11 GB footprint (10 GB of it GPU buffers), and system swap grew from
    1.1 GB to 3.7 GB before I stopped it at ~17:23, with 16 layers measured.
  - This broke the standing rule that every model run is capped and must never swap.
  - **Cause, from `optiq/core/sensitivity.py:671`:** optiq keeps the bf16 reference logits for all calibration
    samples for the whole run. That's 24 samples × 1,024 tokens × 248,320 vocab × 2 bytes ≈ 12 GB. The weights
    are only 1.7 GB. The cost scales with the vocabulary and the sample count, not the model size.
- **~17:25 Rerun**, through
  `scripts/capped_optiq_convert.py`: MLX limit 13 GB, MemoryGuard with a 14.5 GB footprint cap, kill on any swap
  growth.
  - My first wrapper imported `optiq.cli.main`, which doesn't exist (the entry is `cli`). It failed before loading
    anything.
- **17:47 Convert done**, in about 25 minutes. Peak footprint 11.0 GB, MLX peak 9.82 GB, swap growth −0.016 GB.
  Log: `runs/optiq-convert-0.8B.log`. It also wrote the `uniform_4bit` and `static_mixed` side outputs;
  `--skip-baselines` would have skipped them.
- **17:51–17:52** 2B + Gemma OptiQ downloaded with aria2 and verified (25 files OK).
- **~17:53** `apply.TESTED` moved to mlx 0.32.4 / mlx-lm 0.32.2 / mlx-optiq 0.8.5 in the working tree, so `apply()`
  runs. It was left uncommitted until the checks passed (committed in v0.0.9).
- **17:54 Run 01, all models in ONE pytest process** (`runs/01_all_models_one_process.out`). Result `.F.F`, then the
  guard stopped the process at a 13.0 GB footprint against Gemma's 7.2 GB cap. Swap didn't grow.
  - **Mistake:** the models were loaded one after another but not freed. The likely cause is failed tests'
    tracebacks keeping the earlier models alive. From here on it's one process per model (`-k q08` / `q2b` /
    `gemma`).
- **Run 02, 0.8B alone (output overwritten by later runs; result quoted here).** Same `.F`. The 4-bit test crashed in
  my own near-tie code: `engine._eos_ids()` is a set, not a list (`TypeError: 'set' object is not subscriptable`).
  It was a bug in the test, not a result. Fixed.
- **Run 03, 0.8B alone (overwritten; quoted).** With the 4-bit KV cache, the texts split at token 17 (" sunlight" vs
  " shorter"). The gap read 1.375 > 0.25. **That gap was measured wrongly:** against a full-precision (bf16) cache,
  while both runs used a 4-bit cache.
- **Diagnostic `scripts/kv4_split_gap.py`**, the same split measured under a 4-bit cache:

  | How the token is computed | Stock gap | Fastpath gap |
  |---|---|---|
  | One token at a time | +0.375 | +0.125 |
  | Block of 3 (MTP verify shape) | +0.375 | +0.5 |

  It's a near-tie: 1–4 bf16 rounding steps at this logit size.
- **Test changed.** Two checks, each with the KV cache the run used:
  - **Drift:** speedups-on vs stock logits along the stock text. Ceilings: 0.25 for bf16, 0.5 for 4-bit.
  - **Split gap:** the gap at a split must be ≤ drift + 0.25.

  Also fixed: a stock text that ended first used to give a gap of 0.
- **18:00 Run 04, 0.8B alone** (`runs/04_q08_drift_check.out`):
  - bf16 KV cache: **passed, drift 0.000 on all 3 prompts.**
  - 4-bit KV cache: **failed, drift 1.758 > 0.5.**
- **Diagnostic `scripts/kv4_truth.py`:** stock and fastpath compared to EXACT attention (fp32, on the same 4-bit
  cache), teacher-forced along the same 96 tokens:

  | 0.8B, 4-bit KV | Largest logit difference | Mean | Top-20 largest | Same top token |
  |---|---|---|---|---|
  | **Prompt 1:** stock vs exact | 3.039 | 0.1219 | 0.75 | 93.8% |
  | **Prompt 1:** fastpath vs exact | 2.938 | 0.1265 | 0.875 | 93.8% |
  | **Prompt 1:** fastpath vs stock | 1.758 | 0.1216 | 1.25 | 95.8% |
  | **Prompt 2:** stock vs exact | 3.438 | 0.1287 | 1.375 | 100% |
  | **Prompt 2:** fastpath vs exact | 3.125 | 0.1192 | 1.375 | 95.8% |
  | **Prompt 2:** fastpath vs stock | 2.078 | 0.1218 | 1.375 | 95.8% |

- **18:18 Run 05, 0.8B alone, new checks** (`runs/05_q08.out`):
  - bf16 MTP: **passed**, drift 0.000 ×3.
  - 4-bit accuracy: **passed.** Mean |err| vs exact: stock 0.1194, fastpath 0.1166. Top-1 agreement: stock 97.6%,
    fastpath 96.5%.

    | Per prompt | Stock error | Fastpath error | Stock top-1 | Fastpath top-1 |
    |---|---|---|---|---|
    | 1 | 0.122 | 0.127 | 93.8% | 93.8% |
    | 2 | 0.129 | 0.119 | 100% | 95.8% |
    | 3 | 0.108 | 0.104 | 99.0% | 100% |

  - n-gram: **failed its precondition.** The fold counter stayed at 0.
- **Diagnostics `scripts/ngram_widths.py`, `scripts/ngram_fold_count.py`:**
  - The 5- and 9-wide passes seen in the first diagnostic were optiq's one-off cost calibration
    (`ngram_spec.CALIBRATION_POSITIONS = (1, 2, 3, 5, 9)`).
  - In real decoding of these prompts on the 0.8B, optiq's gate verified at widths 1–3 (plus one 22-wide pass, over
    the fold's 16-row limit). So the fold had nothing to do.
  - **optiq keeps learned state across calls, per model:** `_COST_MODELS`, `_PASS_COSTS`, `_ACCEPTANCE_PRIOR`,
    `_MTP_DRAFT_COSTS`. In one process the second run starts from what the first learned.
  - Earlier benchmark batches ran one configuration per process, so they were not affected.
- **Test changed:** each n-gram run starts from a cleared optiq state. The fold is checked where it runs: the stock
  text is fed back at width 9 (36 rows), and the counter must be > 0 there.

- **18:21 Run 06, 0.8B alone** (`runs/06_q08.out`):
  - bf16 MTP: passed.
  - 4-bit accuracy: passed (same numbers as run 05).
  - n-gram: **failed.** The fold ran and its drift is fine (0.250 at width 9), but stock and fastpath split at token 0
    on the sky prompt with a stock logit gap of 1.75.
- **Diagnostic `scripts/ngram_first_tokens.py`.** On the sky prompt the model's own top-3 is "The", "1", "Light".
  - Stock `generate_ngram` starts "1. The primary reason…", repeatably.
  - Fastpath starts "The sky appears blue…" (the true greedy token).
  - The other two prompts agree.
  - **So stock optiq is the one that isn't greedy here.**
- **Diagnostic `scripts/ngram_ablation.py`:** one fastpath swap at a time. Only the draft-head swap (our
  `FastMtpDrafter` replacing optiq's `MtpDrafter`) changes the first token; attention, fused capture and prefill don't.
- **Diagnostics `scripts/draft_cost_cache.py`, `scripts/drafter_errors.py`: the cause is in optiq 0.8.5,
  `ngram_spec.measure_draft_cost`.**
  1. Before the first pass it prices the MTP head with a fake hidden state whose width it takes from
     `embed_tokens.weight.shape[-1]`. On a quantized embedding that's the PACKED width: 256 for 8-bit, not the model's
     1024. So the head's `draft()` fails on a shape error.
  2. Stock `MtpDrafter.draft` catches its own error, sets `ok = False` and returns `[]`. **The MTP head is then
     silently off for the whole stock n-gram run.**
  3. `measure_draft_cost` then times 5 target decode steps of token 0 **on the live cache** and undoes them with
     `trim_prompt_cache(cache, 1)`.
     - Qwen3.5's cache is `ArraysCache` (GDN layers) + `KVCache`, and `can_trim_prompt_cache` is False, so the trim
       does nothing.
     - The 5 fake tokens stay in the recurrent state, and the first real token comes out wrong ("1" instead of "The").
  4. Our `FastMtpDrafter.draft` doesn't catch errors, so the same shape error escapes to `measure_draft_cost`'s own
     `except` and returns 0.0 BEFORE the 5 steps. Our path keeps the cache intact **by accident**. Its draft cost also
     reads as 0.0, so optiq's gate treats the head as free; that can affect speed, not correctness.
  - **Reach:** any quantized Qwen3.5 / 3.6 (GDN hybrid) through optiq's n-gram + MTP path. That includes the 9B and
    `optiq serve --ngram-draft N --mtp`.
  - **For our earlier n-gram numbers:** the fastpath runs weren't corrupted. Any comparison against *stock* optiq
    n-gram + MTP measured a run with the head off and a shifted first token.

- **Fix:** `mlx_fastpath/draft.py` gets `safe_draft_cost`. While the fast drafter switch is on,
  `apply()` swaps it in for optiq's `ngram_spec.measure_draft_cost`, and `remove()` restores optiq's.
  - It uses the model's real hidden size.
  - It times the target step on a scratch cache with up to 1,024 tokens of context, never the live cache.
  - New fast test, `tests/test_draft_cost.py`: a tiny random-weight Qwen3.5 with an 8-bit embedding, which has the
    same untrimmable hybrid cache and packed embedding width.
    - With the fix: 3 passed (`runs/08_draft_cost_fast_tests.out`).
    - With optiq's pricing swapped back in: 2 failed. The KV cache offset was 11 instead of 6 (the 5 fake tokens
      stayed), and the head was asked to draft from width 16 instead of 64.
    - The first version of the test (`runs/07_…`) failed only on its precondition, because the stand-in head raised
      on the wrong width. That's the same accident our real drafter relied on. The stand-in now swallows the error
      like optiq's own drafter.
- **The slow n-gram test now compares against plain greedy decoding** (optiq AR loop, depth 0), not stock n-gram.
- **19:16 Run 09, 0.8B** (`runs/09_q08.out`):
  - bf16 MTP: passed. 4-bit accuracy: passed.
  - n-gram, first split vs greedy: **stock at tokens 0, 23, 11; fastpath at 76, none (all 96 match), 85.** The
    split at 76 has a stock logit gap of 0.000, an exact tie.
  - **Failed:** the fold drift on prompt 2 was 0.375, over my 0.25 ceiling. That ceiling was a guess.
- **Diagnostic `scripts/fold_vs_shape_drift.py`**, bf16, along greedy text. "Shape shift" = how far stock's own logits
  move when the same text is fed 9 tokens at a time instead of 1:

  | Prompt | Fold drift (fastpath vs stock, width 9) | Stock shape shift (w9 vs w1) | Stock (w3 vs w1) | Fastpath w9 vs stock w1 |
  |---|---|---|---|---|
  | sky | 0.25 | 0.406 | 0.32 | 0.406 |
  | Fibonacci | 0.375 | 0.375 | 0.344 | 0.5 |
  | train | 0.344 | 0.375 | 0.406 | 0.344 |

  The fold never moves logits more than stock's own block shape does.
- **Test bounds now measured, not guessed:**
  - drift ≤ stock's shape shift at the same width (which is 0 at width 1);
  - split gap ≤ drift + shape shift.

  The `DRIFT_CEILING` and `SHAPE_SHIFT` constants are gone.

- **19:18 Run 10, 0.8B** (`runs/10_q08.out`): **3 passed.**
  - bf16 MTP drift 0.000 (stock shape shifts 0.32–0.41).
  - n-gram vs greedy: stock splits at 0, 23, 11; fastpath at 76, none, none. Fold drift 0.25 / 0.375 / 0.344 against
    stock shifts 0.406 / 0.375 / 0.375. The split at 76 has a gap of 0.000.
  - 4-bit: as run 05.
  - **Not deterministic run to run:** fastpath's third n-gram prompt matched greedy for 85 tokens in run 09 and 96
    in run 10, with no code change. optiq's gate picks draft lengths from wall-clock timings, so verify shapes, and
    near-tie outcomes, vary between runs.
- **19:20 Run 11, 2B alone** (`runs/11_q2b.out`): **3 passed.**
  - bf16 MTP drift 0.000.
  - n-gram vs greedy: stock splits at 30, 6, 27; fastpath at 30, none, none. The shared split at 30 has a gap of
    0.125 (bound 0.250 + 0.312). Fold drift 0.25 / 0.312 / 0.258 against stock shifts 0.312 / 0.375 / 0.359.
  - **4-bit: fastpath slightly MORE accurate than stock.** Mean error vs exact 0.0747 vs 0.0756; top-1 agreement
    99.3% vs 98.6%.
- **19:21 Run 12, Gemma 4 E2B alone** (`runs/12_gemma.out`): **passed.** Identical text on all 3 prompts, drift 0.000,
  and the guard (7.2 GB footprint cap) did not stop it. This was the old test (no prefill-active precondition); the
  final test is covered by run 15; run 12 doesn't print the peak (the test prints
  `guard.report()` from v0.0.10 on). So run 01's 13 GB was leftover models, not Gemma.

- **19:22 Run 13, the fast set** (`runs/13_fast_set.out`): 89 passed, 8 deselected (the opt-in slow tests), 13.4 s.
- **Committed v0.0.9** (e0648db, tag v0.0.9).
- **Branch audit** (independent Sonnet subagent, read-only, no runs). No blockers. Items 1–7 implemented.
  It found 3 should-fix items and 7 nits:
  - stale lines in this file ("nothing committed", Finding 5 pre-fix, the Gemma cap claim not shown by run 12,
    run 13 missing);
  - the Gemma check couldn't fail in a meaningful way as written (drift 0 by construction; only the text
    comparison checks anything): now named for what it checks, with a precondition that the prefill swap is active;
  - test hygiene: `saved` read after `remove()`, the guard exited in `finally`, `guard.report()` printed;
  - `disable_fast_mtp_drafter` now pops its saved originals;
  - `safe_draft_cost`'s bf16 scratch cache is noted as an approximation.
  Fixed in v0.0.10.
- **Run 14** (`runs/14_fast_affected.out`): affected fast tests, 24 passed.
- **Run 15, Gemma alone** (`runs/15_gemma.out`): the renamed prefill-swap check **passed** (precondition held,
  identical text). Peak footprint **5.59 GB** (cap 7.2), swap growth 0.0.

- **Plan-conformance audit, round 1** (independent Sonnet, read-only, diff main...integration): P1–P7 implemented,
  no blockers. Should-fix: the stack table's Before column (wrong, corrected); no fast-set run after the v0.0.10
  code changes; run 12 predates the final Gemma test. Nits already recorded here (thin 0.8B top-1 margin,
  n-gram timing variance, bf16 MTP drift 0 shows nothing changed). Fixed in v0.0.11.
- **Run 16, full fast set on the final code** (`runs/16_fast_set_final.out`, run with -rA so every test is listed):
  89 passed, 8 deselected, 14.0 s; includes the 3 `test_draft_cost.py` tests.

- **Plan-conformance audit, round 2** (new Sonnet auditor): P1–P4, P6, P7 implemented; stack table, run numbers and
  download wording now match. Should-fix: runs 10 and 11 (the 0.8B / 2B results) predate the v0.0.10 edits to
  `test_lossless_slow.py` and `draft.py`.
- **Runs 17 and 18, 0.8B and 2B on the final code** (`runs/17_q08_final.out`, `runs/18_q2b_final.out`): **3 passed
  each, every number identical to runs 10 and 11.** Peak footprint 2.29 GB (0.8B) and 3.20 GB (2B), swap growth 0.0.
  So all final results now come from the final code: runs 15 (Gemma), 16 (fast set), 17, 18.

## Result of the re-validation
| Model | bf16 MTP (same text) | n-gram vs greedy (fold) | 4-bit KV accuracy vs exact | AR, no MTP |
|---|---|---|---|---|
| Qwen3.5-0.8B (OptiQ, built here; run 17) | pass | pass | pass (err 0.1166 vs stock 0.1194; top-1 96.5% vs 97.6%) | — |
| Qwen3.5-2B (OptiQ, mlx-community; run 18) | pass | pass | pass (err 0.0747 vs 0.0756; top-1 99.3% vs 98.6%) | — |
| Gemma 4 E2B (OptiQ, mlx-community; run 15) | — | — | — | pass (identical) |

On mlx 0.32.4 / mlx-lm 0.32.2 / mlx-optiq 0.8.5, the speedups keep greedy output where that is a fair test (bf16
cache), and are as accurate as stock where it isn't (4-bit cache). Fastpath's n-gram path is greedy; stock optiq's is
not. Speed is NOT re-measured here: that's the comparison grid and the 9B check.

## A/B on small models (2026-11-22, branch `ab-small`; full record in `ab/`)
Goal: A/B the 0.8B and 2B models across stock, uniform int4 and OptiQ, to see whether fastpath helps at all, with
output that stays as sensible as stock.
- **Before the A/B: Gemma checks.**
  - Gemma E2B runs under mlx-lm, optiq's engine and fastpath `generate_ar` (`scripts/gemma_runnable.py`). Single run:
    40.1 tok/s optiq AR vs 57.7 fastpath.
  - Its MTP is a separate 4-layer "assistant" drafter. `mlx-community/gemma-4-E2B-it-assistant-bf16` was downloaded
    (aria2, `inputs/aria2-gemma-assistant.manifest.json`). It runs only through mlx-vlm (`load(strict=False)`: the
    OptiQ file has no audio/vision towers). `scripts/gemma_assistant_check.py`, output not saved to a file (terminal
    only):
    - 1.15–2.1x over mlx-vlm plain;
    - 2 of 3 texts differ from plain decoding, not yet checked for near-ties;
    - mlx-vlm plain is only 15–16 tok/s.
  - The earlier Gemma scripts encoded the template's text and added a second BOS, which made Gemma emit a thought
    channel. The A/B tokenizes with `tokenize=True`. The lossless tests used the doubled input for stock and fastpath
    alike, so they stand.
- **Weights built** (`runs/20_build_*.out`):
  - `Qwen3.5-2B-4bit` (+ optiq MTP sidecar; peak 2.44 GB);
  - `gemma-4-e2b-it-4bit` (peak 5.44 GB; the first try was refused by its own guard);
  - the 0.8B N2 copy.
  - My first build loop failed at once: zsh doesn't split `$spec`.
  - **N2 for the 2B failed:** `mlx-fastpath n2` reads every file in the weight index, including the optional
    `optiq_vision.safetensors` we didn't download. That's a robustness gap in `n2`.
- **Run:** smoke test (6 jobs, `ab/ab_smoke.*`), then the full A/B: 62 jobs, ABBA, all ok, 36.8 min, no swap
  growth (`ab/ab.log`, `ab/ab.jsonl`). Then KL vs bf16 (`ab/kl.*`, 10 jobs).
  - **Gemma's bf16 reference grew swap by 0.78 GB** (peak 12.0 GB). That was inside the runner's 3 GB total-swap
    limit, but against the no-swap preference.
- **Results:** `ab/results.md` (tables, generated) and **`ab/verdict.md`** (the answer). In short:
  - fastpath n-gram is 2.5–3.2x stock on code edits;
  - fast-ar is +5–28% on the 0.8B and level on the 2B;
  - fastpath MTP is 1.32–1.77x optiq's MTP. It beats plain decoding on the 2B (OptiQ chat and math, int4 math);
  - no measured degradation vs stock on the same weights (thin evidence: 12 QA items, 2 splits checked);
  - N2 is no use on the 0.8B;
  - needle and math scores can't separate runtimes (300 tokens too short; 0 everywhere).
- **The early needle splits are near-ties.** Gemma int4's first token is an exact tie under a one-pass prefill
  (`scripts/prefill_first_token.py`).
- **I corrected four overstated lines in my own first draft of `verdict.md`** against the tables: MTP ratio,
  Gemma fast-ar range, 2B fast-mtp range, and the 0.8B code-edit claim. The branch audit then corrected two
  more claims (MTP gains, "never worse") and several ranges (`ab/verdict.md`, "Audit corrections").

## n2 fix: a listed-but-missing optional file
- **Cause, my bug:**
  - `n2.build` looped over every file in the weight index and opened each one, even files with nothing to convert
    (`n2.py` build loop). So the 2B, downloaded without its optional `optiq_vision.safetensors` (297 vision tensors),
    failed.
  - `n2.check` had the same flaw.
  - The tests always built fake models with every listed file present, so this case was never covered.
  - The failed run left `Qwen3.5-2B-OptiQ-4bit-N2-gdnhead.partial` (1.38 GB, one converted shard). It went to the
    Trash.
- **Fix (`n2.missing_shards`):**
  - A listed file that isn't on disk is allowed only if it holds no layer to convert. It's then left out of the copy,
    which lists it like its source does.
  - A missing file holding a layer to convert stops the build before anything is written.
  - `check` reports a file the copy lost, a file missing in only one of the two, and a file missing in both that holds
    converted layers.
- **Tests (`tests/test_n2.py`, 3 new):**
  - left-out sidecar → build ok, check clean, index unchanged;
  - missing file with a layer to convert → `SystemExit` naming it, and no copy or `.partial` folder;
  - copy lost a file → reported.
  - All 3 failed on the old code (`runs/21_n2_tests_before_fix.out`, FileNotFoundError) and pass with the fix
    (`runs/22_…`: n2 + CLI tests 24 passed). Full fast set: 92 passed (`runs/24_…`).
  - The branch audit found `check`'s "missing in both, held converted layers" branch untested. A 4th test was added;
    it fails on the old `n2.py` and passes on the new (`runs/25_n2_missing_both_test.out`). n2 tests: 22 passed.
  - "297 vision tensors" is from the 2B index (`weight_map` entries mapped to `optiq_vision.safetensors`).
  - The audit also corrected the 2B N2 speed and quality wording (verdict "N2 on the 2B").
- **Real build:** `Qwen3.5-2B-OptiQ-4bit-N2-gdnhead`, 37 layers converted in 3 s, `check` clean; the vision file was
  left out as designed (`runs/23_build_n2_2B.out`).
- **2B N2 rows:** `ab/ab_n2_2b.*` (10 jobs, ABBA, all ok, no swap growth) and KL `ab/kl_n2_2b.jsonl`, reusing the
  A/B's bf16 reference.
  - KL 0.085 vs OptiQ 0.048 vs int4 0.139. QA 10 = OptiQ, and the code edit is correct.
  - Speed near noise: about +4–6% plain decoding vs OptiQ + fast-ar; with MTP +43% on math, 0.88x on code.
  - Tables in `ab/results_n2_2b.md`; verdict section "N2 on the 2B".

## Findings so far
1. **The fused 4-bit KV attention kernel is about as accurate as stock's own 4-bit path, not better and not clearly
   worse.**
   - Mean error vs exact is 0.119–0.127 against stock's 0.122–0.129.
   - The largest error is slightly smaller.
   - Top-token agreement with exact is equal on prompt 1 and **4 tokens worse out of 96 on prompt 2** (95.8% vs
     100%). That's the one point against fastpath so far; it needs more prompts and the 2B before calling it.
2. **With a 4-bit KV cache, identical text is not a fair test.** On the 0.8B, stock's own 4-bit attention is up to
   3.4 logits from exact and picks a different top token on up to 6% of tokens. Any two 4-bit kernels split text.
   My 0.5 drift ceiling was stricter than stock's own error, so it was wrong. The right 4-bit test is "no less
   accurate than stock, measured against exact".
3. **The bf16 test does NOT exercise our attention code on these models, or on the 9B.**
   - The bf16 fold only runs when (query heads per KV head) × (verify width) ≥ 16. The 0.8B, the 2B and the 9B all
     have 4 query heads per KV head, and MTP depth 2 checks 3 tokens: 4 × 3 = 12.
   - So with the bf16 cache, `apply()` at default settings changes the draft head, the capture and the prefill,
     but never the attention. The drift of 0.000 confirms it: those changes are lossless here.
   - The fold only runs with wider verify blocks: `generate_ngram` (8 drafts → width 9) or MTP depth ≥ 4.
4. **On these small models the GQA fold is almost idle in real decoding.** MTP verifies at width 3, and optiq's
   n-gram gate also chose ≤ 3 here. The fold's gain needs verify widths ≥ 4 (4 query heads per KV head). Whether the
   9B's n-gram runs reach those widths on copy-heavy prompts is worth checking in the grid.
5. **Stock optiq 0.8.5's n-gram + MTP path is broken on quantized Qwen3.5** (see the 18:21 log entry): the MTP head
   is silently off, and the hybrid cache gets 5 untrimmable fake steps, so the output is not greedy. Fastpath's path
   avoided this only by accident and mispriced the head at 0. **Fixed in v0.0.9** by `draft.safe_draft_cost`.
6. **optiq convert's memory scales with the vocabulary, not the model size** (≈12 GB of reference logits for any
   Qwen3.5 with 24 samples). `--n-samples` is the lever.

## Mistakes, in one place
- The first conversion ran without a memory cap or guard: swap grew by 2.6 GB. It was stopped by hand.
- I didn't look for prebuilt OptiQ small models before converting.
- I estimated 2–5 GB for the conversion without reading how optiq computes sensitivity.
- The wrapper's entry-point name was wrong (`main` instead of `cli`).
- Run 01 put all models in one process, so memory piled up until the guard stopped it.
- The test's EOS lookup treated a set as a list.
- The first split gap was measured with the wrong cache type, and the first drift ceiling (0.5) was set without
  knowing stock's own error.
- The stack table's "Before" column was written from an older note, not the freeze (caught by the
  plan-conformance audit, corrected).
- The outputs of runs 02 and 03 were overwritten by later runs. From now on every run writes a new numbered file in
  `runs/`.

## Next
Items 1-3 are done (runs 10-12).
1. 4-bit KV test → "no less accurate than stock" against exact fp32 attention on the same cache, over all prompts:
   - fastpath's mean error ≤ stock's × 1.1;
   - fastpath's top-token agreement with exact ≥ stock's − 2 points.
2. bf16 test: keep "same text or near-tie". Add a `generate_ngram` case (verify width 9), so the fold runs and its
   counter is > 0.
3. One model per process: 0.8B, then 2B, then Gemma. Each run gets its own file in `runs/`.
4. Then the comparison grid (each step asked first). Rows:
   - bf16 + mlx-lm;
   - uniform int4 + mlx-lm;
   - OptiQ int4 + mlx-lm;
   - OptiQ int4 + optiq MTP;
   - OptiQ int4 + fastpath;
   - uniform int4 + fastpath.

   Speed is measured at short, 4k and 16k context; quality as KL vs bf16 plus tasks with known answers.
5. The 9B speed check before main. **Done:** passed, run 26 (section below).

## 9B speed check (run 26, 2026-11-23, integration @ 4fb20c8)

`MLX_FASTPATH_SLOW=1 pytest -m slow tests/test_model_slow.py` on Qwen3.5-9B-OptiQ-4bit, stack mlx 0.32.4 / mlx-lm
0.32.2 / mlx-optiq 0.8.5. Log: `runs/26_9B_speed_check.out`.

| | Result |
|---|---|
| Decode speed | **29.18 tok/s** at 4k context, 160 tokens (floor 20), MTP acceptance 0.42 |
| Correct output | 3 of 3 secret codes, no looping |
| Memory | MLX peak 7.10 GB (cap 8.0), footprint 8.23 GB (cap 10) |
| Swap | no growth |
| Time | 27 s |

- In line with the 9B's earlier 25-31 tok/s with MTP: the n-gram head pricing change did not slow the 9B.
- N2 is on by default and the 9B's N2-gdnhead copy is a finished copy (has `N2_manifest.json`), so `load()` used it;
  the log does not print the loaded folder, so this is inferred from the switch, not shown.
- The two warnings are SWIG deprecation notices. A first try without `-m slow` selected no test; it is not in the log.
