# Before / after: decode speed on M5 (24 GB), 2026-11-19

Setup:
- All runs greedy (temperature 0), with an audit (see "Audit" below).
- Memory capped: Qwen at 8 GB MLX limit + 9 GB watchdog; Gemma at 12 GB + 14 GB watchdog + a swap-growth guard. **Swap grew 0 GB in every run.**
- Detail per run: `results.md` §9–11. Raw data: `../../exp/tps20*.jsonl`, `tps_long*.jsonl`, `gemma_full.jsonl`. Texts: `*_texts*.json`.

## Qwen3.5-9B-OptiQ-4bit (tok/s)
| Context | Before: plain decoding | Before: optiq MTP as shipped | **After: best setup** | Speedup vs plain | Best setup |
|---|---|---|---|---|---|
| Short | 17.9 | 22.5 | **30.8** | 1.72× | A |
| 4k | 17.2 | 21.0 | **25.3** | 1.47× | A |
| 8k | 16.9 | 20.8 | **25.4** | 1.51× | A |
| 16k | 16.1 | 19.7 | **23.6** | 1.47× | A + fold |
| 32k | 15.0 | 16.2 | **21.5–21.9** | 1.43–1.46× | B (4-bit KV) |

The two setups:
- **A:** optiq MTP at depth 2, plus:
  - optiq's fused GDN-capture kernel, which it ships but doesn't use by default;
  - a 32k-vocab draft head (the fork's reduced-vocab idea).

  The output is checked by the full model, so quality matches plain greedy.
- **B:** A plus the 4-bit KV cache from the model's `kv_config.json`, the fused 4-bit KV attention kernel (verify widths ≤ 3), and folded quantized matmuls (widths ≥ 4). Slightly lossy: 96% agreement with the bf16 re-score. The task still passes.

A on a bf16 cache falls back to plain speed at 32k (15.7 tok/s), so above ~16k switch to B. The crossover between 16k and 32k was not measured.

## Gemma-4-12B-it-qat-OptiQ-4bit + its assistant drafter (tok/s)
| Context | Before: plain decoding | **After: MTP drafter** (mlx-vlm 0.6.13) | Speedup | Best block |
|---|---|---|---|---|
| Short | 13.0 | **24.5** | 1.88× | default (4 and 6 equal) |
| 4k | 12.7 | **24.1** | 1.89× | 4 (default 23.4) |
| 16k | 12.3 | **20.5** | 1.67× | 4 (default 20.1) |
| 32k | 12.0 | **21.5** | 1.79× | default |

- The drafter is used exactly as Google and mlx-community ship it: `draft_kind="mtp"`, which already has a 2048-centroid reduced-vocab head.
- Block 2 is always worst (16.5–18.5 tok/s). The default or block 4 is best.
- My extra ideas didn't help Gemma: the GQA fold measured 0.88–1.13× on its 1-KV-head, head-dim-512 global layers. MLX's fallback already batches those heads, and global attention is only about 6 ms of each ~80 ms token at 32k.

## Audit ("fast but rubbish?"): passed for both models
- **Task with ground truth:** 3 secret codes hidden at 10%, 50% and 90% depth of a 4k–32k document. Every run of every config at every length listed all 3 exactly.
- **No loops:** distinct 4-grams 0.985–1.0, longest repeated span 2–6 tokens (normal list formatting).
- **Independent timing:** wall-clock tok/s (excluding prefill) matches the runtimes' own numbers within about 2%. Every run produced its full token count.
- **Re-scored by the plain Qwen model:** MTP-A output agrees with its argmax 99.5–100% (plain greedy itself: 97.5–99.5%), NLL in the same range. B: 96%.
- **Gemma MTP text:** byte-identical to plain greedy on short and 4k at the default block. At 16k and 32k it splits at near-ties ("**Secret Codes:**" vs "**Task 1: Secret Codes**"), with the same content.
- **I read the outputs** (short, 16k, 32k, both models): coherent, faithful summaries.

## Memory facts learned
- Qwen 9B: load peak 7.2 GB if the model is loaded **before** importing optiq, 10.9 GB if optiq is imported first. `OptiqEngine(path)` holds a duplicate model copy; use `OptiqEngine.from_loaded`.
- Gemma 12B + drafter: 8.6 GB active after load. Footprint peaks: 10 GB short, 11 GB 4k–16k, 13 GB 32k.

## How to reproduce or use
- **Qwen:** `exp/tps20.py` (configure: `cap=True, vocab=32768`, plus `fold=True` and `kv_bits=4` above ~16k). All swaps are runtime patches. No installed package or project file was edited.
- **Gemma:** `mlx_vlm.generate(..., draft_model=load_drafter(assistant, kind="mtp")[0], draft_kind="mtp", draft_block_size=4)`. See `exp/gemma_test.py`.
