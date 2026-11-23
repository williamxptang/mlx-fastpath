"""#6 prefill on Qwen3.5-9B-OptiQ-4bit: chunk size x 'LM head on every position' (stock optiq/mlx-lm) vs
'LM head on the last token only' (lossless). Time, MLX peak, final-logit agreement. Capped 8 GB MLX."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json, sys, time, glob
from pathlib import Path
import mlx.core as mx
mx.set_memory_limit(8 * 2**30); mx.set_cache_limit(1 * 2**30)
import mlx_fastpath as fp
eng = fp.load(MODELS + "/Qwen3.5-9B-OptiQ-4bit", memory_limit_gb=8)
from mlx_lm.models.cache import make_prompt_cache
inner, head = eng._inner, eng._lm_head
docs = sorted(glob.glob(str(Path(__file__).resolve().parents[2]) + "/research/kmic68/research/**/*.md", recursive=True))
corpus = eng.tokenizer.encode("\n\n".join(open(p).read() for p in docs) * 3)


def prefill(ids, step, all_logits):
    cache = make_prompt_cache(eng._model)
    mx.synchronize(); mx.reset_peak_memory(); t = time.perf_counter()
    for i in range(0, len(ids), step):
        chunk = mx.array([ids[i:i + step]])
        h = inner(chunk, cache=cache)
        last = i + step >= len(ids)
        if all_logits:
            lg = head(h); mx.eval(lg)               # stock: logits for every position
        else:
            lg = head(h[:, -1:, :]) if last else None
            mx.eval([c.state for c in cache] + ([lg] if lg is not None else []))
    final = lg[0, -1].astype(mx.float32)
    el = time.perf_counter() - t
    return el, mx.get_peak_memory() / 2**30, final


for N in (4096, 16384, 32768):
    ids = corpus[:N]
    ref_t, ref_pk, ref = prefill(ids, 512, True)
    print(json.dumps({"N": N, "mode": "stock_512_all_logits", "s": round(ref_t, 2), "tok_s": round(N / ref_t),
                      "mlx_peak_gb": round(ref_pk, 2)}), flush=True)
    for step in (256, 512, 1024, 2048, 4096):
        t, pk, fin = prefill(ids, step, False)
        print(json.dumps({"N": N, "mode": f"last_logit_{step}", "s": round(t, 2), "tok_s": round(N / t),
                          "speedup": round(ref_t / t, 2), "mlx_peak_gb": round(pk, 2),
                          "final_argmax_same": fin.argmax().item() == ref.argmax().item(),
                          "final_logit_maxdiff": round(mx.abs(fin - ref).max().item(), 3)}), flush=True)
