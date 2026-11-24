"""T1: learn only how a second pass is mixed in (base weights frozen), then score it like eval_nll.py.

Gate kinds (a = how much of the extra pass to take, per pass p):
  scalar   a = sigmoid(w)                         1 number
  channel  a = sigmoid(w)  per hidden channel     H numbers
  lowrank  a = sigmoid(b + U relu(V [h; x]))      per token and channel, rank R (U starts at 0, so a starts at a0)
Loss: next-token cross-entropy on TRAIN text (even-index research notes + WikiText-2 train), SEQ-token pieces.
Then: plain and gated NLL on the same TEST-A / TEST-B chunks as eval_nll.py, appended as rows for analyze.py.

    python train_gate.py MODEL_DIR OUT_JSONL BLOCK_S BLOCK_E KIND A0 [STEPS]
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loopfwd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mlx_fastpath.guard import MemoryGuard  # noqa: E402

model_dir, out_path = sys.argv[1:3]
S, E, KIND, A0 = int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], float(sys.argv[6])
STEPS = int(sys.argv[7]) if len(sys.argv) > 7 else 300
SEQ, RANK, LR = int(os.environ.get("SEQ", 512)), int(os.environ.get("RANK", 32)), float(os.environ.get("LR", 3e-3))
CHUNK, N_CHUNKS = int(os.environ.get("CHUNK", 1024)), int(os.environ.get("N_CHUNKS", 48))
LIMIT_GB = float(os.environ.get("MLX_LIMIT_GB", 12))
mx.set_memory_limit(int(LIMIT_GB * 2**30))
mx.set_cache_limit(int(float(os.environ.get("CACHE_GB", 0.25)) * 2**30))


def docs():
    """All pinned notes in the original order (even index = TRAIN, odd = TEST-A), so docs()[0::2] / [1::2] still hold."""
    rows = [l.split(None, 1) for l in open(Path(__file__).resolve().parent / "doc_split.txt") if not l.startswith("#")]
    return [str(REPO / p.strip()) for _, p in rows]


def wikitext(split):
    import pandas as pd
    return "".join(pd.read_parquet(os.path.expanduser(
        f"~/Documents/AI/datasets/wikitext-2/{split}-00000-of-00001.parquet")).text)


def pieces(tok, text, n):
    ids = tok.encode(text)
    out = [ids[i:i + n] for i in range(0, len(ids) - n + 1, n)]
    assert out
    return out


def nll(model, ids, mixer):
    x = mx.array([ids])
    h = loopfwd.forward(model, x, block=(S, E), passes=2 if mixer else 1, mixer=mixer, return_hidden=True)
    total, tgt = 0.0, x[0, 1:]
    for i in range(0, len(ids) - 1, 256):
        lg = loopfwd.head(model, h[:, i:min(i + 256, len(ids) - 1)]).astype(mx.float32)[0]
        t = tgt[i:i + lg.shape[0]]
        total += (mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, t[:, None], axis=-1)[:, 0]).sum().item()
    return total / (len(ids) - 1)


guard = MemoryGuard(min_free_gb=LIMIT_GB + 1, max_footprint_gb=LIMIT_GB + 2)
guard.__enter__()
from mlx_lm import load  # noqa: E402

model, tok = load(model_dir)
model.freeze()
model.train()        # GDN layers use the differentiable ops path only in training mode (no dropout in Qwen3.5)
hidden = (model.language_model if hasattr(model, "language_model") else model).args.hidden_size
SEED = int(os.environ.get("SEED", 0))
if SEED:
    mx.random.seed(SEED)                             # gate init (V) and data order both follow SEED
gate = loopfwd.Gate(KIND, hidden, A0, rank=RANK)
TRAIN_SRC = os.environ.get("TRAIN_SRC", "both")      # both | notes | wiki: what the gate is trained on
train = ((pieces(tok, "\n\n".join(open(p).read() for p in docs()[0::2]), SEQ) if TRAIN_SRC in ("both", "notes") else [])
         + (pieces(tok, wikitext("train"), SEQ) if TRAIN_SRC in ("both", "wiki") else []))
assert train, TRAIN_SRC
random.Random(SEED).shuffle(train)


def loss_fn(g, ids):
    x = mx.array([ids])
    logits = loopfwd.forward(model, x[:, :-1], block=(S, E), passes=2, mixer=g).astype(mx.float32)
    return nn.losses.cross_entropy(logits[0], x[0, 1:], reduction="mean")


loss_and_grad = nn.value_and_grad(gate, loss_fn)
opt = optim.Adam(learning_rate=LR)
t0, losses = time.perf_counter(), []
for step in range(STEPS):
    loss, grads = loss_and_grad(gate, train[step % len(train)])
    opt.update(gate, grads)
    mx.eval(gate.parameters(), opt.state, loss)
    losses.append(loss.item())
    mx.clear_cache()
    if step % 50 == 0 or step == STEPS - 1:
        a = gate.alpha(mx.zeros((1, 1, hidden)), mx.zeros((1, 1, hidden)))
        print(json.dumps({"step": step, "loss_avg50": round(sum(losses[-50:]) / len(losses[-50:]), 4),
                          "alpha_mean": round(a.mean().item(), 4), "s": round(time.perf_counter() - t0, 1)}),
              flush=True)
train_s = round(time.perf_counter() - t0, 1)
name = f"T1_{KIND}_b{S}-{E}_a{A0}" + ("" if TRAIN_SRC == "both" else f"_{TRAIN_SRC}") + (f"_seed{SEED}" if SEED else "")
mx.save_safetensors(str(Path(out_path).with_name(name + ".safetensors")),
                    dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(gate.parameters())))

model.eval()         # back to the fast kernels for scoring
tests = {"TEST-A": pieces(tok, "\n\n".join(open(p).read() for p in docs()[1::2]), CHUNK)[:N_CHUNKS],
         "TEST-B": pieces(tok, wikitext("test"), CHUNK)[:N_CHUNKS]}
assert all(len(v) == N_CHUNKS for v in tests.values())
with open(out_path, "a") as f:
    for cfg, mixer in (("plain", None), (name, gate)):
        rec = {"model": os.path.basename(model_dir.rstrip("/")), "config": cfg, "block": [S, E],
               "passes": 2 if mixer else 1, "kind": KIND, "a0": A0, "steps": STEPS, "train_s": train_s,
               "train_loss_first50": round(sum(losses[:50]) / min(50, len(losses)), 4),
               "train_loss_last50": round(sum(losses[-50:]) / min(50, len(losses)), 4), "chunk": CHUNK}
        for k, cs in tests.items():
            vals = [round(nll(model, ids, mixer), 5) for ids in cs]
            rec[k], rec[k + "_mean"] = vals, round(sum(vals) / len(vals), 5)
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(json.dumps({k: v for k, v in rec.items() if k not in tests}), flush=True)
guard.__exit__(None, None, None)
print(json.dumps({"guard": guard.report(), "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2)}), flush=True)
