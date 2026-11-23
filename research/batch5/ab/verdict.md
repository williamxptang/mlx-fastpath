# A/B verdict: is fastpath any use on Qwen3.5 0.8B / 2B and Gemma 4 E2B? (2026-11-22)

Data: `results.md` (generated from `ab.jsonl`, 62 jobs: each config twice, ABBA, all ok, 36.8 min) and `kl.jsonl`.
Plan: `plan.md`. Speed is end-to-end tok/s (incl. prefill), median of 2 runs. "Stock" = mlx-lm `stream_generate`.

## Short answer
- **Yes, but in a narrow band.** The one large win is **fastpath n-gram on copy-heavy work**: 2.5–3.2x stock on the
  code edit, on every quantized Qwen. Stock optiq's own n-gram can't compete; it isn't greedy (batch5 results).
- **Plain decoding (fast-ar)** helps the 0.8B (+5–28% on int4/OptiQ) and Gemma int4 (−1% to +11%: +6–11% on the short prompts). It's level on the
  2B (0.94–1.01x)
  and on bf16 (−1% to +6%).
- **fastpath MTP is 1.32–1.77x optiq's own MTP everywhere.** It beats plain decoding (both stock and fast-ar) on the 2B:
  OptiQ chat and math (+19%, +38% vs stock) and int4 math (+17% vs stock, +20% vs fast-ar).
  - On the 0.8B it gains on math vs stock (+14% int4, +20% OptiQ) but stays below fast-ar there. On chat, code and
    needle it's level or a loss: acceptance is 15–44%.
- **No measured degradation vs stock on the same weights:** same QA score (one config +1 with MTP), same code-edit
  result, and fast-ar gives identical text on chat, math and code.
  - The two very early splits that were checked are exact ties (below). Four other early splits were not examined:
    Gemma bf16 needle at 40, Gemma OptiQ needle at 90, 2B OptiQ n-gram chat at 35, 0.8B int4 MTP chat at 58.
  - The evidence is thin: 12 QA items, and needle and math are uninformative.
  - This covers fastpath's runtime swaps. N2 changes the weights and does lose an edit (below).
- **The weights matter far more than the runtime:** int4 / OptiQ are 1.3–3x faster than bf16. OptiQ costs speed vs
  plain int4 on the 2B (54–67 vs 65–89 tok/s stock, needle row included) but is much closer to bf16 (KL 0.05 vs 0.14).

## Best setup per model (speed, with output as good as stock)
| Model | Fastest general setup | Copy-heavy (code edit) | vs bf16 stock |
|---|---|---|---|
| Qwen3.5-0.8B | OptiQ + fast-ar: 87–122 tok/s (stock OptiQ 83–103) | OptiQ + fast-ngram: 332 tok/s | 1.5–1.8x general, 4.8x code |
| Qwen3.5-2B | stock int4 (65–89); fast-mtp adds +17% on math only (slower on code/needle). For quality: OptiQ + fast-mtp 54–93 (+19% chat, +38% math vs stock OptiQ) | int4 + fast-ngram: 208 tok/s (OptiQ: 192) | 2.4–3.0x general (stock int4), 6.5–7x code |
| Gemma 4 E2B | int4 + fast-ar: 52–65 tok/s (stock int4 53–61) | — (no MTP head in fastpath's reach) | 2.5–2.7x |

## Output quality
- **QA (12 graded questions):** identical between stock and every fastpath runtime on the same weights, with one
  exception that favours fastpath: 0.8B int4 scored 8 with MTP (optiq's or fastpath's) vs 7 with plain decoding.
  Scores by weights:

  | Weights | 0.8B | 2B | Gemma |
  |---|---|---|---|
  | bf16 | 9 | 11 | 12 |
  | int4 | 7 | 9 | 12 |
  | OptiQ | 9 | 10 | 12 |
  | N2 (0.8B only) | 9 | — | — |

- **Code edit** (compiles / MODULES edit / MAX_VERIFY edit): same as stock for every fastpath runtime on the same
  weights. The 0.8B misses the MODULES edit under every runtime and weights. bf16 and N2 also miss MAX_VERIFY;
  int4 and OptiQ get it. So N2 drops an edit that its own source (OptiQ) gets right.
- **Same text as stock, first 300 tokens:**
  - fast-ar is identical on chat, math and code for every model and weights.
  - On the 4k needle, fast-ar splits from stock early (tokens 0–111; MTP and n-gram split at 3–247). The Gemma int4 split at token 0 is an **exact tie**:
    "This" vs "Under", gap 0.000 with a one-pass prefill. mlx-lm's 2048-chunk prefill tips it one way, fastpath's
    1024-chunk prefill the other, and fastpath's is closer to the one-pass logits (max diff 0.41 vs 0.77).
    `scripts/prefill_first_token.py`; Qwen 2B: all three agree, margin 2.4.
  - The 2B OptiQ chat split at token 0 under both optiq's MTP and fastpath's MTP ("Here is a step-by-step…" vs
    stock "### How a CPU Cache Works…") is also an **exact tie**: "Here" vs "###", gap 0.000 under a one-pass
    prefill (optiq's engine), while mlx-lm's ids[:-1]-then-last prefill tips it to "###" (checked inline,
    2026-11-22). It's optiq's engine prefill, not a fastpath swap.
  - MTP and n-gram texts split later. On chat, math and code, optiq's MTP and fastpath's MTP split at the same
    positions; on the needle they don't (0.8B int4 3 vs 13, 0.8B OptiQ 19 vs 7, 2B int4 6 vs 247, 2B OptiQ 142 vs 14). That's the
    verify-shape near-tie behaviour the slow tests bound (batch5 results).
- **Needle codes (0/3) and math (0) are 0 under every runtime and weights, bf16 included.** The models start with the
  summary or derivation, and 300 tokens ends first. These two columns don't separate runtimes; they're not evidence
  either way.
- **Repetition:** 4-gram uniqueness is 0.89–0.99, with no loops. N2 has the lowest (0.886 on the 0.8B chat).

## Weights quality vs bf16 (runtime-free)
| Model | int4 KL / top-1 | OptiQ KL / top-1 | N2 KL / top-1 |
|---|---|---|---|
| 0.8B | 0.179 / 77.4% | 0.059 / 86.7% | 0.115 / 81.5% |
| 2B | 0.139 / 78.0% | 0.048 / 88.1% | — |
| Gemma E2B | 1.235 / 65.0% | 0.543 / 77.2% | — |

- OptiQ is about 3x closer to bf16 than plain int4 on Qwen and 2.3x on Gemma.
- Gemma's quantization error is much larger than Qwen's on this text, though QA stays 12/12. That's a caution for
  Gemma 4-bit on harder tasks, untested here.
- **N2 doubles the 0.8B's KL and isn't faster there** (fast-mtp 74–96 vs 72–119 on OptiQ). **N2 is no use on the 0.8B.**

## N2 on the 2B (added after the n2 fix; `results_n2_2b.md`, one separate ABBA session of 10 jobs + KL)
- **Quality:**
  - KL vs bf16 0.085, between OptiQ (0.048) and plain int4 (0.139). That's 1.76x OptiQ's; on the 0.8B N2 was 1.97x.
  - Top-1 agreement 85.4% vs OptiQ's 88.1%.
  - QA 10 = OptiQ's 10, and the code edit is fully correct (1/1/1) on N2, unlike the 0.8B.
  - Text leaves OptiQ stock earlier: chat at token 0, math at 47 (OptiQ fast-ar: identical; OptiQ fast-mtp: 0, 99).
  - Repetition is fine (unique 4-grams 0.95–0.98).
- **Speed, vs OptiQ stock in the same session:**
  - N2 + fast-ar +6–9% (OptiQ + fast-ar +2–4%);
  - N2 + fast-mtp 0.88–1.43x (OptiQ + fast-mtp 0.86–1.21x): +43% on math, 1.03–1.06x on chat and needle, 0.88x on
    code.
  - **These are near the noise.** The stock baseline's two runs differ by 5–7%, and N2 fast-ar's by 3–5%. Against
    OptiQ + fast-ar, whose runs agree within 1–2%, N2 + fast-ar is +4–6%. So N2 adds a small, uncertain gain to plain
    decoding; the only clear MTP gain is on math.
- **Verdict:** on the 2B, N2 is a modest trade: about +4–6% on plain decoding (near noise) and +43% with MTP on
  math only, for about 1.8x the distance from bf16.
  - It lost no QA answers and no code edit here, but that's thin evidence: 12 questions and one edit.
  - Its text leaves OptiQ's early: chat at token 0, needle at 4, math at 47.
  - It stays behind OptiQ on KL.
- **Note on noise:** OptiQ + fast-mtp scored 1.05x on chat in this session vs 1.19x in the main A/B. MTP speed varies
  run to run, because optiq's gate sizes drafts from wall-clock timings.
  - Plain decoding isn't immune: stock's two runs here differ by 5–7%, fastpath plain on OptiQ by 1–2%.
  - Any single gain under ~7% in this session is not established.

## What doesn't help on small models
- **bf16:** nothing helps; it's bandwidth-bound (2B 27–30 tok/s, Gemma 21–25 with or without fastpath, needle row included).
- **MTP depth 2 on the 0.8B,** fastpath's or optiq's: drafts are cheap, but acceptance is too low.
- **N2 on the 0.8B:** worse quality, not faster.
- **fast-ar on the 2B and on Gemma OptiQ:** level with stock (mlx-lm already pipelines decoding).

## Caveats
- 2 runs per config; differences under ~5% are noise.
- 300-token outputs. The 4k needle is the only long context; 16k+ isn't measured here.
- One process per config, run back to back on a machine with ~12 GB free.
- **Memory:** every A/B job stayed within its cap with no swap growth.
- **The KL phase grew swap by 0.78 GB** when building Gemma's bf16 reference (peak 12.0 GB). The runner allowed it
  (inside the 3 GB total-swap limit from batch 2), but it breaks the no-swap preference. `kl.py`'s scoring also holds
  the full-vocab reference in float32 (6.8–11 GB peaks); that should be chunked before it's reused.
- **Fastpath robustness gap found:** `mlx-fastpath n2` fails when a file listed in the weight index is missing (the 2B
  without its optional `optiq_vision.safetensors`), so the 2B has no N2 row.

## Audit corrections (branch audit, 2026-11-22)
- The MTP claim was too narrow: 2B int4 math and (vs stock) 0.8B math also gain.
- "Never worse" became "no measured degradation", with the unexamined splits listed.
- Ranges now include the needle row. The MTP ratio minimum is 1.32. bf16 fast-ar is −1% to +6%.
- The needle split positions differ between optiq's and fastpath's MTP.
- Known asymmetry: the MTP runtimes' token ids and counts come from re-encoding the generated text
  (`tok.encode(st.text)`), which can differ slightly from the generated ids. That touches their e2e tok/s and split
  positions, not stock or fast-ar.
- **Branch audit of the n2 fix:** three 2B N2 claims were corrected: the speed range ("+5–20%" was wrong; it's
  +4–6% plain, near noise, and +43% MTP on math only), "reproduce within ~3%" (stock varies 5–7%), and "doesn't lose
  answers or edits" (true on QA and the code edit only; the text splits early). Also: 0.8B KL ratio 1.97x, not
  1.96x.
