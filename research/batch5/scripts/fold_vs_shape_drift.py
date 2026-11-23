"""0.8B, bf16 cache, along greedy text: fold drift (fastpath vs stock, both width 9) against stock's own shape drift
(stock width 9 vs stock width 1) and the fold's distance from one-at-a-time (fastpath width 9 vs stock width 1)."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import sys
sys.path.insert(0, "tests")
import mlx.core as mx
import mlx_fastpath as fp
from mlx_fastpath.guard import MemoryGuard
import test_lossless_slow as T
M = MODELS + "/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed"
with MemoryGuard(min_free_gb=6, max_footprint_gb=4.5):
    eng = fp.load(M, memory_limit_gb=2.5, n2=False)
    eos = min(eng._eos_ids())
    for p in T.PROMPTS:
        ids = T._prompt_ids(eng, p)
        fp.remove()
        g = T._ids(eng, eng.generate("", prompt_ids=ids, max_tokens=T.MAX_TOKENS, depth=0).text) + [eos]
        s9 = T._logits_along(eng, ids, g, block=9, mode="stock")
        f9 = T._logits_along(eng, ids, g, block=9, mode="fast")
        s1 = T._logits_along(eng, ids, g, block=1, mode="stock")
        s3 = T._logits_along(eng, ids, g, block=3, mode="stock")
        d = lambda a, b: round(mx.abs(a - b).max().item(), 3)
        print(f"{p[:24]:24s} fold (f9-s9) {d(f9, s9)} | stock shape (s9-s1) {d(s9, s1)} (s3-s1) {d(s3, s1)} | f9-s1 {d(f9, s1)}")
