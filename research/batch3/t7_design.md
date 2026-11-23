# T7: calibrated quantizer instead of RTN (design only; nothing has run)

Goal: requantize OptiQ's 8-bit layers to 4-bit with less error than N2's round-to-nearest (§16.2, §16.6), inside the
per-job cap (free − 1 GB, ~11–13 GB on 2026-11-21) with no swap.

## What mlx-lm 0.31.3 ships (read from source and `--help`; nothing run)
| Tool | How it works | Fits a 9B under ~11 GB? | Needs bf16 (T6)? |
|---|---|---|---|
| `mlx_lm.quant.gptq` | Wraps **every** Linear in a Hessian catcher, runs the whole model on all calibration batches, then quantizes (`gptq.py:60-76`) | **No.** All Hessians live at once: the MLP down-proj alone is 12288² × 4 B = 0.6 GB × 32 layers ≈ 19 GB, plus a full-model forward | its input must be a float model |
| `mlx_lm.quant.awq` | Layer by layer (`awq.py:452-505`, `mx.clear_cache` per block), memory-friendly | Would fit, but **`qwen3_5` is not in `AWQ_MODEL_CONFIGS`** (llama/qwen2/qwen3 …) → `NotImplementedError`. The GDN layers would need a new scale map | float model |
| `mlx_lm.quant.dwq` | Distillation: trains the quantized model's scales/biases to match teacher logits (KL) | Unknown: backprop through 9B at seq 1025, batch 4. Likely > 10 GB; must be measured on a tiny model first | teacher can be OptiQ itself |
| `mlx_lm.quant.dynamic_quant` | OptiQ-style sensitivity → mixed bits at a target bpw | **No.** Loads the full float model non-lazy (`load(args.model)`): 18 GB in bf16 | yes |

All four fetch calibration text with `load_data` (`quant/utils.py`), which **downloads** a gist on first use unless
`~/.cache/mlx-lm/calibration_v5.txt` exists.

## Proposal: layer-streamed GPTQ on the 8-bit layers only (research script, no package edit)
- **Input:** OptiQ's own 8-bit weights, dequantized per layer (the same source `requant.py` uses). The 8-bit step is
  1/17 of the 4-bit step, so 8-bit is a near-lossless stand-in for float here. **Does not need T6.**
- **Method:** standard sequential GPTQ (the paper's form, not mlx-lm's all-at-once form):
  1. run the calibration batch through the model block by block, keeping only the current block's input activations;
  2. inside the block, collect the Hessian X^T X for the 8-bit Linear layers only (4-bit layers stay as they are);
  3. quantize those with GPTQ (reuse `mlx_lm.quant.gptq`'s per-layer math by import), group 64, 4-bit;
  4. recompute the block's output with the quantized weights, move to the next block, free the block.
- **Peak estimate:** one block's weights dequantized (~0.5 GB) + its Hessians (≤ 0.6 GB for down-proj, ≤ 1.2 GB per
  block) + activations (128 × 512 × 4096 × 2 B ≈ 0.5 GB, twice) + embeddings/head ≈ **3–4 GB**. To be measured on the
  tiny random Qwen3.5 of `test_requant.py` first, then on the 9B with `MemoryGuard`.
- **Time:** ~133 layers; mlx-lm's GPTQ step per 4096-wide layer is seconds → est. 20–40 min incl. calibration passes.
- **Output:** `…/Qwen3.5-9B-OptiQ-4bit-N2-all-gptq` (same layer set as N2-all, which RTN failed), then the same
  `requant.py check` (relaxed: converted rows are GPTQ, not RTN), KL short/long, needle 4k/16k + T1's 12, code file.
- **Bar (proposed, set before any run):** PASS if it meets the batch-2 KL bar and all needle tasks as N2-gdnhead did;
  this would show N2-all's fail was RTN's.

## Decisions for the user
1. **Calibration text:** download mlx-lm's calibration gist (an internet fetch into `~/.cache`), or use local text
   (e.g. this repo's research notes + code; narrower domain, no download)?
2. Build the layer-streamed GPTQ script (research files only) and measure it on the tiny model? (~1 h of work.)
3. AWQ for `qwen3_5` (needs a GDN scale map) and DWQ (memory unknown) are not proposed now.
