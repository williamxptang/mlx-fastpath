"""Held-out next-token loss (NLL, nats/token) of a model, plain and looped, on fixed text chunks.

Text sets (fixed, deterministic):
  TEST-A  our research notes (mlx-fastpath/research/**/*.md), documents with an odd index after sorting by path
          (even-index documents are TRAIN, kept for T1/T2)
  TEST-B  WikiText-2 raw test split
Each set: the documents joined and tokenized, cut into CHUNK-token pieces, the first N_CHUNKS of them. Every chunk is one
full-sequence forward (no cache); loss over positions 1..CHUNK-1. The output head runs in 256-row slices.

    python eval_nll.py MODEL_DIR OUT_JSONL CONFIGS_JSON      (CONFIGS_JSON: list of {"name", "block", "passes", "alphas"})
"""
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loopfwd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CHUNK = int(os.environ.get("CHUNK", 1024))
N_CHUNKS = int(os.environ.get("N_CHUNKS", 48))
LIMIT_GB = float(os.environ.get("MLX_LIMIT_GB", 8))
mx.set_memory_limit(int(LIMIT_GB * 2**30))
mx.set_cache_limit(1 * 2**30)

model_dir, out_path, configs_path = sys.argv[1:4]
configs = json.load(open(configs_path))
assert configs and configs[0]["name"] == "plain", "first config must be the plain baseline"


def split_docs(kind):
    """The research notes behind TEST-A / TRAIN, pinned in doc_split.txt (the set at main 21d4e5b, v0.0.18): later
    commits add .md files under research/, which would otherwise shift the split."""
    rows = [l.split(None, 1) for l in open(Path(__file__).resolve().parent / "doc_split.txt") if not l.startswith("#")]
    return [str(REPO / p.strip()) for k, p in rows if k == kind]


def text_sets():
    test_docs = split_docs("TEST-A")
    assert len(test_docs) >= 5, test_docs
    test_a = "\n\n".join(open(p).read() for p in test_docs)
    import pandas as pd
    wt = pd.read_parquet(os.path.expanduser("~/Documents/AI/datasets/wikitext-2/test-00000-of-00001.parquet"))
    test_b = "".join(wt.text)
    return {"TEST-A": test_a, "TEST-B": test_b}


def chunks(tok, text):
    ids = tok.encode(text)
    out = [ids[i:i + CHUNK] for i in range(0, len(ids) - CHUNK + 1, CHUNK)][:N_CHUNKS]
    assert len(out) == N_CHUNKS, f"only {len(out)} chunks"
    return out


def chunk_nll(model, ids, cfg):
    x = mx.array([ids])
    h = loopfwd.forward(model, x, block=cfg.get("block"), passes=cfg.get("passes", 1),
                        alphas={int(k): v for k, v in cfg.get("alphas", {}).items()}, return_hidden=True)
    total, tgt = 0.0, x[0, 1:]
    for i in range(0, CHUNK - 1, 256):
        lg = loopfwd.head(model, h[:, i:min(i + 256, CHUNK - 1)]).astype(mx.float32)[0]
        t = tgt[i:i + lg.shape[0]]
        nll = mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, t[:, None], axis=-1)[:, 0]
        total += nll.sum().item()
    return total / (CHUNK - 1)


sys.path.insert(0, str(REPO))
from mlx_fastpath.guard import MemoryGuard  # noqa: E402
from mlx_lm import load  # noqa: E402

guard = MemoryGuard(min_free_gb=LIMIT_GB + 3, max_footprint_gb=LIMIT_GB + 2)
guard.__enter__()
model, tok = load(model_dir)
sets = {k: chunks(tok, v) for k, v in text_sets().items()}
with open(out_path, "a") as f:
    for cfg in configs:
        t0 = time.perf_counter()
        rec = {"model": os.path.basename(model_dir.rstrip("/")), "config": cfg["name"], "block": cfg.get("block"),
               "passes": cfg.get("passes", 1), "alphas": cfg.get("alphas", {}), "chunk": CHUNK}
        for name, cs in sets.items():
            vals = []
            for ids in cs:
                vals.append(round(chunk_nll(model, ids, cfg), 5))
                mx.clear_cache()
            rec[name] = vals
            rec[name + "_mean"] = round(sum(vals) / len(vals), 5)
        rec["seconds"] = round(time.perf_counter() - t0, 1)
        rec["peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(json.dumps({k: v for k, v in rec.items() if k not in sets}), flush=True)
guard.__exit__(None, None, None)
print(json.dumps({"guard": guard.report()}), flush=True)
