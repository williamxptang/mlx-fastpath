"""Weights quality, runtime-free: KL(bf16 || weights) and top-1 agreement, teacher-forced on held-out text
(research/batch4/plan.md, not in any A/B prompt), first N tokens, plain mlx-lm forward, one model per process.
  python kl.py ref   MODEL_PATH OUT.npy     bf16 reference: log-softmax rows saved (fp16)
  python kl.py score MODEL_PATH REF.npy LABEL   prints {"prompt": "kl", "label", "kl_mean", "kl_p99", "top1"}
Env from the runner: GUARD_FREE_GB GUARD_CAP_GB GUARD_SWAP_GB MLX_LIMIT_GB MLX_CACHE_GB."""
import json
import os
import sys
from pathlib import Path

import numpy as np

from mlx_fastpath.guard import MemoryGuard

N = 1024
TEXT = (Path(__file__).resolve().parents[2] / "batch4" / "plan.md").read_text()

with MemoryGuard(min_free_gb=float(os.environ["GUARD_FREE_GB"]), max_footprint_gb=float(os.environ["GUARD_CAP_GB"]),
                 swap_tolerance_gb=float(os.environ["GUARD_SWAP_GB"])) as g:
    import mlx.core as mx
    mx.set_memory_limit(int(float(os.environ["MLX_LIMIT_GB"]) * 2**30))
    mx.set_cache_limit(int(float(os.environ["MLX_CACHE_GB"]) * 2**30))
    from mlx_lm import load
    mode, path = sys.argv[1], sys.argv[2]
    model, tok = load(path)
    ids = tok.encode(TEXT)[:N]
    assert len(ids) == N, len(ids)
    logp = []
    for i in range(0, N, 256):                       # chunks keep the full-vocab logits small
        if i == 0:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(model)
        lg = model(mx.array([ids[i:i + 256]]), cache=cache)[0].astype(mx.float32)
        logp.append(np.array(lg - mx.logsumexp(lg, axis=-1, keepdims=True)))
    logp = np.concatenate(logp)
    if mode == "ref":
        np.save(sys.argv[3], logp.astype(np.float16))
        print(json.dumps({"ref": sys.argv[3], "rows": int(logp.shape[0]), "vocab": int(logp.shape[1])}))
    else:
        ref = np.load(sys.argv[3]).astype(np.float32)
        assert ref.shape == logp.shape, (ref.shape, logp.shape)
        kl = (np.exp(ref) * (ref - logp)).sum(-1)
        print(json.dumps({"prompt": "kl", "label": sys.argv[4], "kl_mean": round(float(kl.mean()), 5),
                          "kl_p99": round(float(np.percentile(kl, 99)), 4),
                          "top1": round(float((ref.argmax(-1) == logp.argmax(-1)).mean()), 4)}))
    print(json.dumps({"guard": g.report()}), flush=True)
