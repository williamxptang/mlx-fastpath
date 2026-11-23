"""0.8B: why stock MtpDrafter disables itself, and what hidden width optiq's costing feeds a drafter."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import traceback
import mlx.core as mx
import mlx_fastpath as fp
from mlx_fastpath.guard import MemoryGuard
from optiq.runtime import ngram_spec
M = MODELS + "/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed"
with MemoryGuard(min_free_gb=6, max_footprint_gb=4.5):
    eng = fp.load(M, memory_limit_gb=2.5, n2=False)
    fp.remove()
    d = ngram_spec.MtpDrafter(eng._model, depth=1)
    lm = d.model
    print("drafter.model:", type(lm).__name__, "| has mtp_forward:", hasattr(lm, "mtp_forward"), "| has _mtp_core:", hasattr(lm, "_mtp_core"))
    emb = lm.model.embed_tokens
    print("embed:", type(emb).__name__, "weight", emb.weight.shape, emb.weight.dtype, "| hidden_size", lm.args.hidden_size)
    h = mx.zeros((1, 1, lm.args.hidden_size), mx.bfloat16)
    try:
        lm.mtp_forward(h, mx.array([[0]], mx.uint32), mtp_cache=lm.make_mtp_cache(), return_hidden=True, position_offset=None)
        print("stock mtp_forward: OK")
    except Exception as e:
        print("stock mtp_forward RAISES:", type(e).__name__, str(e)[:300])
    fp.apply()
    ours = ngram_spec.MtpDrafter(eng._model, depth=2)
    try:
        ours.draft(mx.zeros((1, 1, emb.weight.shape[-1]), mx.bfloat16), 0)
        print("ours with optiq's costing width", emb.weight.shape[-1], ": OK")
    except Exception as e:
        print("ours with optiq's costing width", emb.weight.shape[-1], "RAISES:", type(e).__name__, str(e)[:200])
    fp.remove()
