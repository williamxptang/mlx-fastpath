"""0.8B: first tokens of generate_ngram, stock vs apply(), each from a cleared optiq n-gram state; and the model's own argmax."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import mlx.core as mx
import mlx_fastpath as fp
from mlx_fastpath.guard import MemoryGuard
from optiq.runtime import ngram_spec
M = MODELS + "/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed"
def clean():
    for s in (ngram_spec._COST_MODELS, ngram_spec._PASS_COSTS, ngram_spec._ACCEPTANCE_PRIOR, ngram_spec._MTP_DRAFT_COSTS):
        s.clear()
with MemoryGuard(min_free_gb=6, max_footprint_gb=4.5):
    eng = fp.load(M, memory_limit_gb=2.5, n2=False)
    tok = eng.tokenizer
    for p in ["Explain in five sentences why the sky is blue.",
              "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
              "A train leaves at 9:40 and arrives at 13:15. How long is the trip? Show the steps."]:
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
        top = mx.argsort(-eng._model(mx.array([ids]))[0, -1].astype(mx.float32))[:3].tolist()
        fp.remove(); clean(); s = fp.generate_ngram(eng, ids, max_tokens=12)
        fp.remove(); clean(); s2 = fp.generate_ngram(eng, ids, max_tokens=12)
        fp.apply(); clean(); f = fp.generate_ngram(eng, ids, max_tokens=12); fp.remove()
        fp.apply(); clean(); f2 = fp.generate_ngram(eng, ids, max_tokens=12); fp.remove()
        show = lambda t: [tok.decode([x]) for x in t[:8]]
        print(p[:25], "| model top3:", [tok.decode([x]) for x in top])
        print("  stock  ", show(s), "\n  stock2 ", show(s2), "\n  fast   ", show(f), "\n  fast2  ", show(f2))
