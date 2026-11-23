"""Offline check before any model run: how often is the target's next token outside the draft vocabulary?

A draft that only scores ids in a set S can never propose a token outside S, so every output token outside S cuts the
accepted block there. Compares S = ids < 32768 (results.md §14.5b: code_edit -16% with it) with S = ids < 32768 plus
every id in the prompt. Uses the texts already produced (lossless runs, so they are the target's greedy tokens),
re-tokenized; no model is loaded. Writes coverage.jsonl.

Usage: python coverage.py
"""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).parent
DF = HERE.parent / "dflash"
TOK = Tokenizer.from_file(MODELS + "/Qwen3.5-9B-OptiQ-4bit/tokenizer.json")
K = 32768

# (results file, method) -> the prompts file its prompt ids come from
SOURCES = [("dflash.jsonl", "D2-w4-adaptive", "prompts.json"),
           ("crossover.jsonl", "D2-w4", "prompts_long.json"),
           ("long_out.jsonl", "AR", "prompts_c.json")]


def miss_rates(prompt_ids, out_ids):
    """Fraction of output tokens a draft over each id set can't propose."""
    assert out_ids, "empty output"
    union = set(range(K)) | set(prompt_ids)
    return {"miss_32k": sum(t >= K for t in out_ids) / len(out_ids),
            "miss_32k_prompt": sum(t not in union for t in out_ids) / len(out_ids),
            "extra_rows": len(union) - K}


def main():
    rows = []
    for fname, method, pfile in SOURCES:
        prompts = json.load(open(DF / pfile))["prompts"]
        seen = set()
        for line in open(DF / fname):
            r = json.loads(line)
            if r["method"] != method or r["prompt"] in seen:
                continue
            seen.add(r["prompt"])
            out = TOK.encode(r["text"], add_special_tokens=False).ids
            row = {"source": fname, "prompt": r["prompt"], "out_tokens": len(out),
                   **{k: round(v, 4) if isinstance(v, float) else v for k, v in miss_rates(prompts[r["prompt"]], out).items()}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    assert rows, "no rows read"
    with open(HERE / "coverage.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
