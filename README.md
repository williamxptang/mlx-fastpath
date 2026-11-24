# mlx-fastpath

Measured, audited runtime speedups for MLX / mlx-lm / mlx-optiq on Apple Silicon (M5, 24 GB).
Nothing inside mlx, mlx-lm or optiq is edited: everything is swapped in at runtime by `apply()`, and
`apply()` refuses (runs stock) on package versions it was not tested on.

| Qwen3.5-9B-OptiQ-4bit, greedy | Plain | With mlx-fastpath |
|---|---|---|
| Decode, short / 4k / 8k / 16k / 32k | 17.9 / 17.2 / 16.9 / 16.1 / 15.0 tok/s | 30.8 / 25.3 / 25.4 / 23.6 / 21.5 |
| Code edit / copy from prompt (n-gram + MTP) | 17.3 / 17.3 | 71 / 59 |
| Prefill 4k / 16k / 32k | 614 / 627 / 483 tok/s | 836 / 772 / 664 |

All audited: hidden-code task 3/3, plain-model re-score, loop check, independent timing, no swap.
Table measured on mlx 0.32.3 / mlx-lm 0.31.3 / mlx-optiq 0.8.1. On the current tested stack (mlx 0.32.4 /
mlx-lm 0.32.2 / mlx-optiq 0.8.5) the 4k decode check measured 29.2 tok/s (`research/batch5/results.md`).

**Status:** research code, published as is. It patches private internals of exactly the versions in
`mlx_fastpath/apply.py` (`TESTED`) and runs stock on anything else. Not maintained for later releases.

## Use

```python
import mlx_fastpath as fp
with fp.MemoryGuard(min_free_gb=11, max_footprint_gb=10):     # refuse / stop instead of swapping
    engine = fp.load("…/Qwen3.5-9B-OptiQ-4bit")                # load BEFORE importing optiq (7.2 vs 10.9 GB peak)
    fp.apply()
    stats = fp.generate(engine, prompt_ids, max_tokens=256)     # MTP depth 2; 4-bit KV above 24k tokens
    ids = fp.generate_ngram(engine, prompt_ids)                # when the answer copies from the prompt
```

Server: `mlx-fastpath serve --model … --mtp --mtp-depth 2 [--ngram-draft 8]` (optiq's own server, patched).
Versions: `mlx-fastpath versions`.

## What it does
- `attention`: GQA fold for verify (gqa·qL ≥ 16) and a fused 4-bit/8-bit KV attention kernel (qL ≤ 3), folded quantized matmuls (qL 4–16).
- `draft`: 32k-vocab MTP draft head (engine MTP and optiq's n-gram speculator), optiq's fused GDN capture.
- `prefill`: LM head on the last prompt position only, 1024-token chunks.
- `guard`: `MemoryGuard`. `load`: safe load order.
- `assistant`: optiq's Gemma 4 `-assistant` drafting loop with a settable draft length. `apply(assistant_gamma=1)`
  replaces `optiq serve --drafter`'s fixed 2: on E2B int4 (M1) gamma 1 was faster in all 26 runs measured, +6.7% mean.
  `"auto"` (per-pass choice) is opt-in: it averaged level with gamma 2, below the best fixed gamma
  (`research/gemma/results.md`). Default: optiq's loop untouched.
- In mlx-lm's own loop (no optiq engine), only `attention` acts: `attention.enable()`, then mlx-lm with
  `kv_bits=4` decodes 1.07-1.49x faster at 16k-32k on Qwen3.5 0.8B/2B (M1); tokens split at close calls, no measurable accuracy difference
  (`research/mlxlm/results.md`).

## Develop / release
- Prototypes and evidence: `research/` (Kmic-68 study in `research/kmic68/`). Not installed.
- `research/loop/`: looping a block of a normal Qwen3.5's layers (no training: 0.4-1.9% of the gap to the next size; a small learned gate
  trained on the research notes improved unseen WikiText where a same-size plain adapter did not). Research only.
- Install into the env that already has mlx, mlx-lm and mlx-optiq: `pip install --no-deps --no-build-isolation .`
- Tests: `pytest` (fast set, < 10 s, each test < 1 s); `MLX_FASTPATH_SLOW=1 pytest -m slow` loads the real model (~30 s, ≤ 9 GB).
- Version: root `VERSION` only; +0.0.1 per commit with a changelog line and tag `v<version>`.

## Research notes
`research/` is a lab notebook: scripts, raw results (`*.jsonl`, `*.out`, `*.log`) and write-ups
(`results.md`, `verdict.md`, `report.md`) per batch, kept as they were run. In this public copy:
- Model paths point to `$MLX_MODELS` (default `~/models`); in logs and results, `<repo>`, `<venv>`,
  `<staging>`, `<data>`, `<tmp>` and `<pytest-tmp>` stand for local folders.
- Left out: the prompt corpora (`prompts*.json`), generated texts (`*_texts.json`), run plans and launch
  scripts, research lane notes, environment freezes and a raw power log. Regenerate prompts with the
  `make_prompts*.py` scripts; their source text differs from the original, so token counts will too.
- `research/loop/` trains and scores on the repo's research notes (`doc_split.txt`); some listed notes (run plans,
  lane notes) are not in this copy, so its notes-based numbers (TEST-A) cannot be re-run exactly from here. The
  WikiText-2 numbers (TEST-B) can.

## Related
- Two optiq 0.8.5 issues found here (n-gram + MTP draft-cost pricing, prefill LM head) are described in
  `research/batch5/results.md`; `mlx_fastpath/draft.py` and `prefill.py` hold the runtime fixes.
- [brjen/optiq-mtp-fix](https://github.com/brjen/optiq-mtp-fix) first proposed running the LM head on the last
  prefill chunk only (optiq 0.4.25).

## License
MIT (see `LICENSE`). Models keep their own licences.
