# fastpath attention in mlx-lm's own generate loop (2026-11-24)

Question: does mlx-fastpath help plain mlx-lm, without optiq's engine? Only `mlx_fastpath.attention` is switched
(`enable()` / `disable()`); nothing from optiq is used. Decoding runs through `mlx_lm.generate.generate_step`, as
`mlx_lm.generate --kv-bits 4` does (quantized_kv_start 5000, group 64, prefill chunks 2048).

Setup: the M1 test Mac (Apple M1, 16 GB), mlx 0.32.4 / mlx-lm 0.32.2 (mlx-optiq 0.8.5 installed, unused), MLX limit 6 GB.
Models: Qwen3.5-0.8B-OptiQ-4bit (`optiq_mixed`), and four Qwen3.5-2B weights: uniform int4 (mlx-community 4bit),
OptiQ-4bit, OptiQ-4bit-N2-gdnhead, bf16. Prompt: research notes + "Summarize the text above in one paragraph." (chat
template). 256 greedy tokens. Two runs: 0.8B + 2B int4 first, the other three 2B weights after (same scripts).

## Speed (`mlxlm_ab.py`, `mlxlm_ab.jsonl`)
Per cell the prompt (minus its last token) is prefilled once and saved; each run reloads that cache and decodes.
Order off, on, on, off; decode tok/s after the first token, median of 2. "Split" = first token that differs.

| Model | Prompt | KV | mlx-lm alone | + fastpath attention | Speedup | Same tokens |
|---|---|---|---|---|---|---|
| 0.8B OptiQ | 4,117 | bf16 | 71.1 | 71.1 | 1.001 | yes |
| 0.8B OptiQ | 16,405 | bf16 | 59.9 | 59.9 | 0.999 | yes |
| 0.8B OptiQ | 16,405 | 4-bit | 49.9 | 64.7 | **1.296** | split at 120 |
| 0.8B OptiQ | 32,789 | bf16 | 49.6 | 49.7 | 1.004 | yes |
| 0.8B OptiQ | 32,789 | 4-bit | 38.4 | 57.0 | **1.486** | split at 33 |
| 2B int4 | 4,117 | bf16 | 47.9 | 48.0 | 1.002 | yes |
| 2B int4 | 16,405 | bf16 | 42.6 | 42.8 | 1.005 | yes |
| 2B int4 | 16,405 | 4-bit | 37.5 | 45.0 | **1.200** | split at 58 |
| 2B int4 | 32,789 | bf16 | 37.2 | 37.2 | 1.002 | yes |
| 2B int4 | 32,789 | 4-bit | 30.5 | 41.2 | **1.349** | split at 32 |
| 2B N2-gdnhead | 4,117 | bf16 | 39.0 | 39.0 | 1.001 | yes |
| 2B N2-gdnhead | 16,405 | bf16 | 35.6 | 35.5 | 0.998 | yes |
| 2B N2-gdnhead | 16,405 | 4-bit | 31.8 | 37.0 | **1.165** | split at 34 |
| 2B N2-gdnhead | 32,789 | bf16 | 31.6 | 31.5 | 0.997 | yes |
| 2B N2-gdnhead | 32,789 | 4-bit | 26.7 | 34.4 | **1.288** | split at 61 |
| 2B OptiQ | 4,117 | bf16 | 35.6 | 35.7 | 1.001 | yes |
| 2B OptiQ | 16,405 | bf16 | 32.7 | 32.9 | 1.005 | yes |
| 2B OptiQ | 16,405 | 4-bit | 29.6 | 34.1 | **1.152** | split at 36 |
| 2B OptiQ | 32,789 | bf16 | 29.5 | 29.4 | 0.998 | yes |
| 2B OptiQ | 32,789 | 4-bit | 25.1 | 31.8 | **1.267** | split at 22 |
| 2B bf16 | 4,117 | bf16 | 15.1 | 15.1 | 1.001 | yes |
| 2B bf16 | 16,405 | bf16 | 14.6 | 14.5 | 0.997 | yes |
| 2B bf16 | 16,405 | 4-bit | 13.9 | 14.8 | **1.067** | split at 41 |
| 2B bf16 | 32,789 | bf16 | 13.8 | 13.8 | 1.000 | yes |
| 2B bf16 | 32,789 | 4-bit | 12.8 | 14.4 | **1.121** | split at 129 |

- The two "off" runs gave identical tokens in every cell (the cache reload is sound); run-to-run spread at most 1.2%.
- bf16 KV: decode is qL = 1, which the dispatch sends to stock attention: level (0.997-1.005), identical tokens.
- 4-bit KV: the fused 4-bit kernel replaces mlx-lm's quantized attention: 1.07-1.49x. The gain is largest where
  attention is the biggest share of a step (small or 4-bit weights, long context) and smallest with bf16 weights,
  whose weight reads dominate.
- With the patch, 4-bit KV decodes faster than bf16 KV at 32k on every model (0.8B 57.0 vs 49.6; 2B int4 41.2 vs
  37.2, N2 34.4 vs 31.6, OptiQ 31.8 vs 29.5, bf16 14.4 vs 13.8) with a lower peak (0.8B 0.82 vs 1.27 GB; 2B int4
  1.20 vs 1.59, N2 1.50 vs 1.89, OptiQ 1.64 vs 2.03, bf16 3.69 vs 4.04). Without it, `--kv-bits 4` costs 7-23% at 32k.
- 2B weights on this M1 in mlx-lm's loop (4k, bf16 KV): int4 47.9 > N2 39.0 > OptiQ 35.6 > bf16 15.1 tok/s.
- At 4k the 4-bit rows are not run: below quantized_kv_start (5000) mlx-lm does not quantize.
- Not testable here: mlx-lm speculative decoding with a draft model. It refuses Qwen3.5 (`Speculative decoding
  requires a trimmable prompt cache`: the GDN layers' ArraysCache), so the patch's verify fold has no mlx-lm path
  on these models.

## Are the 4-bit splits near-ties? (`split_check.py`, `split_check.jsonl`)
Per 4-bit cell: the same prompt is also prefilled with a plain bf16 KV cache (the reference: no KV quantization).
mlx-lm's own tokens (A) are teacher-forced one token per step three ways: 4-bit KV + stock attention, 4-bit KV +
fastpath attention, and bf16 KV + stock (reference). Each side's own token at the split is reproduced (checks true
in all ten).

| Model, prompt | Split | Tokens (mlx-lm / fastpath) | Reference picks | Reference gap | Gap mlx-lm / fastpath | Mean KL vs ref (mlx-lm / fastpath) | Top-1 = ref (mlx-lm / fastpath) |
|---|---|---|---|---|---|---|---|
| 0.8B, 16k | 120 | " DP" / ":" | mlx-lm | 0.50 | 0.125 / 0.000 | 0.02257 / 0.02262 | 115 / 114 of 121 |
| 0.8B, 32k | 33 | " and" / " that" | fastpath | 0.25 | 0.000 / -0.125 | 0.02094 / 0.02092 | 32 / 33 of 34 |
| 2B int4, 16k | 58 | " fork" / " contr" | fastpath | 0.625 | 0.250 / -0.125 | 0.01316 / 0.01310 | 58 / 59 of 59 |
| 2B int4, 32k | 32 | " constraints" / " request" | mlx-lm | 0.25 | 0.250 / 0.000 | 0.00939 / 0.00935 | 32 / 31 of 33 |
| 2B N2, 16k | 34 | " project" / " specific" | fastpath | 0.375 | 0.000 / -0.125 | 0.01327 / 0.01343 | 33 / 34 of 35 |
| 2B N2, 32k | 61 | " syn" / " cond" | mlx-lm | 0.125 | 0.125 / -0.125 | 0.00806 / 0.00850 | 60 / 59 of 62 |
| 2B OptiQ, 16k | 36 | " specific" / " user" | mlx-lm | 0.375 | 0.125 / 0.000 | 0.01014 / 0.01040 | 36 / 35 of 37 |
| 2B OptiQ, 32k | 22 | " text" / "," | fastpath | 0.125 | 0.125 / 0.000 | 0.00403 / 0.00269 | 22 / 23 of 23 |
| 2B bf16, 16k | 41 | " optimizing" / " AI" | fastpath | 0.375 | 0.125 / 0.000 | 0.01124 / 0.01222 | 39 / 39 of 42 |
| 2B bf16, 32k | 129 | "M" / "Apple" | fastpath | 0.125 | 0.000 / -0.250 | 0.00928 / 0.00884 | 126 / 127 of 130 |

(Gap = logit of mlx-lm's token minus fastpath's token; positive favours mlx-lm.)

- Every split is close: the two tokens are within 0.125-0.625 logits of each other under the reference.
- The reference sides with fastpath in 6 of 10 and with mlx-lm in 4 of 10. Mean KL to the reference: fastpath lower
  in 5 of 10, higher in 5 (largest gaps: 2B OptiQ 32k 0.0027 vs 0.0040 in fastpath's favour, 2B bf16 16k 0.0122 vs
  0.0112 against). No consistent direction, so no measurable accuracy difference in these 10 cells (no statistical test;
  per-cell KL differs by up to a third either way), in line with the slow test "4-bit KV no less accurate than stock vs exact fp32" (batch5).

## Use in mlx-lm
```python
from mlx_fastpath import attention
attention.enable()          # Qwen3.5 / Qwen3-Next attention; then use mlx_lm generate/stream_generate with kv_bits=4
```
Only the attention patch acts in mlx-lm's loop; prefill already skips the LM head there (lazy logits), and the MTP,
n-gram and draft pieces need optiq's engine. Measured on M1 with the 0.8B and 2B models only; the 9B was not run here.
