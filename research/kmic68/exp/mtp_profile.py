import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import cProfile, pstats, glob, io, sys
import mlx.core as mx
from optiq.runtime.engine import OptiqEngine
eng = OptiqEngine(MODELS + "/Qwen3.5-9B-OptiQ-4bit")
tok = eng.tokenizer
docs = sorted(glob.glob("<staging>/research/261118-transferable-ideas/**/*.md", recursive=True))
ids = tok.encode("\n\n".join(open(p).read() for p in docs))[:16000]
rnd = mx.random.randint(1000, 60000, (16000,)).tolist()
for name, pids in (("real", ids), ("random", rnd)):
    for depth in (0, 1):
        pr = cProfile.Profile(); pr.enable()
        st = eng.generate("", prompt_ids=pids, max_tokens=100, depth=depth, temperature=0.0)
        pr.disable()
        print(name, "depth", depth, "tok/s", round(st.decode_tok_s, 2), "prefill_s", round(st.prefill_time_s, 1), "accept", round(st.acceptance_rate, 2), flush=True)
        if depth == 1:
            s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12); print(s.getvalue()[:3500])
