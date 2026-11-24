"""Task check: GSM8K test problems, greedy, plain vs looped (and optionally a trained gate), exact final answer.

Prompt: the chat template with thinking off, asking for the final answer after '####'. Generation is token by
token with per-pass caches (loopfwd.LoopCache), MAX_NEW tokens, stop at EOS. Score: the last number after '####' (or, without
'####', the last number in the output) equals the reference answer. Paired: same problems for every config; reports solved counts and the discordant pairs.

    python gsm_eval.py MODEL_DIR OUT_JSONL CONFIGS_JSON [N]
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loopfwd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mlx_fastpath.guard import MemoryGuard  # noqa: E402

model_dir, out_path, configs_path = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 100
MAX_NEW = int(os.environ.get("MAX_NEW", 320))
LIMIT_GB = float(os.environ.get("MLX_LIMIT_GB", 8))
mx.set_memory_limit(int(LIMIT_GB * 2**30))
mx.set_cache_limit(1 * 2**30)
configs = json.load(open(configs_path))
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text):
    nums = NUM.findall(text.split("####")[-1]) or NUM.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def generate(model, tok, prompt_ids, cfg, mixer, eos):
    block = tuple(cfg["block"]) if cfg.get("block") else None
    passes = cfg.get("passes", 1)
    alphas = {int(k): v for k, v in cfg.get("alphas", {}).items()}
    cache = loopfwd.LoopCache(model, block or (0, 0), passes)
    kw = dict(cache=cache, block=block, passes=passes, alphas=alphas, mixer=mixer)
    logits = loopfwd.forward(model, mx.array([prompt_ids]), **kw)
    out = []
    for _ in range(MAX_NEW):
        t = int(mx.argmax(logits[0, -1]).item())
        if t in eos:
            break
        out.append(t)
        logits = loopfwd.forward(model, mx.array([[t]]), **kw)
    return out


with MemoryGuard(min_free_gb=LIMIT_GB + 3, max_footprint_gb=LIMIT_GB + 2) as guard:
    import pandas as pd
    from mlx_lm import load
    model, tok = load(model_dir)
    eos = set(tok.eos_token_ids) if getattr(tok, "eos_token_ids", None) else {tok.eos_token_id}
    df = pd.read_parquet(os.path.expanduser("~/Documents/AI/datasets/gsm8k/test-00000-of-00001.parquet")).head(N)
    probs = [(q, a.split("####")[-1].strip().replace(",", "")) for q, a in zip(df.question, df.answer)]
    assert len(probs) == N > 0
    hidden = (model.language_model if hasattr(model, "language_model") else model).args.hidden_size
    results = {}
    for cfg in configs:
        mixer = None
        if cfg.get("gate"):
            mixer = loopfwd.Gate(cfg["gate_kind"], hidden, 0.5, rank=cfg.get("rank", 32))
            mixer.load_weights(str(Path(configs_path).resolve().parent / cfg["gate"]))
        t0, solved, outs = time.perf_counter(), [], []
        for q, ans in probs:
            msgs = [{"role": "user", "content": q + "\nGive the final answer as a number after '####'."}]
            try:
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
            except TypeError:
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            text = tok.decode(generate(model, tok, ids, cfg, mixer, eos))
            solved.append(same(last_number(text), ans))
            outs.append(text[-200:])
            mx.clear_cache()
        results[cfg["name"]] = solved
        rec = {"model": os.path.basename(model_dir.rstrip("/")), "config": cfg["name"], "n": N,
               "solved": sum(solved), "per_problem": solved, "seconds": round(time.perf_counter() - t0, 1),
               "tails": outs[:5]}
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k not in ("per_problem", "tails")}), flush=True)
    base = results[configs[0]["name"]]
    for name, s in results.items():
        if name != configs[0]["name"]:
            gained = sum(1 for a, b in zip(base, s) if b and not a)
            lost = sum(1 for a, b in zip(base, s) if a and not b)
            print(json.dumps({"config": name, "vs": configs[0]["name"], "gained": gained, "lost": lost}), flush=True)
    print(json.dumps({"guard": guard.report()}), flush=True)
