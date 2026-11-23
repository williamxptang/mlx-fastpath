"""Item 3, step 1: where does a plain-decoding token go on Qwen3.5-9B-OptiQ-4bit (short context)?
  sync   : optiq's AR loop shape: forward, eval, .item(), next (graph build and GPU never overlap)
  build  : Python graph construction only (no eval)
  async  : mlx-lm's pipelined shape: next step's graph is built on the lazy argmax and queued with async_eval
           while the GPU runs the current one (same tokens, by construction)
  matmul : every quantized linear of the main model (incl. the LM head) at M=1, one graph, nothing else:
           the weight-read floor this machine actually reaches
  layers : per-layer-type GPU time (all 24 GDN layers / 8 attention layers / 32 MLPs as one graph each, decode step)
Guarded: MemoryGuard(14, 14). Appends to profile_step.jsonl."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
import statistics
import time
from pathlib import Path

import mlx_fastpath as fp

HERE = Path(__file__).parent
TARGET = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
ids = json.load(open(HERE.parent / "dflash/prompts.json"))["prompts"]["chat"]
N = 64
out = open(HERE / "profile_step.jsonl", "a")


def emit(row):
    print(json.dumps(row), flush=True)
    out.write(json.dumps(row) + "\n"); out.flush()


with fp.MemoryGuard(min_free_gb=14, max_footprint_gb=14) as g:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import make_prompt_cache
    eng = fp.load(TARGET, memory_limit_gb=12)
    model = eng._model

    def fresh():
        c = make_prompt_cache(model)
        lo = model(mx.array([ids]), cache=c)
        mx.eval(lo)
        return c, int(mx.argmax(lo[0, -1]).item())

    def run_sync():
        c, t = fresh()
        toks, ts = [t], []
        for _ in range(N):
            t0 = time.perf_counter()
            lo = model(mx.array([[toks[-1]]]), cache=c)
            mx.eval(lo)
            toks.append(int(mx.argmax(lo[0, -1]).item()))
            ts.append(time.perf_counter() - t0)
        return toks, ts

    def run_build():
        c, t = fresh()
        ts = []
        for _ in range(16):
            t0 = time.perf_counter()
            lo = model(mx.array([[t]]), cache=c)
            ts.append(time.perf_counter() - t0)
            mx.eval(lo)
        return ts

    def run_async():
        c, t = fresh()
        y = mx.array([t])
        toks = [t]
        nxt = mx.argmax(model(y[None], cache=c)[0, -1], keepdims=True)
        mx.async_eval(nxt)
        t0 = time.perf_counter()
        ts = []
        for _ in range(N):
            cur = nxt
            nxt = mx.argmax(model(cur[None], cache=c)[0, -1], keepdims=True)
            mx.async_eval(nxt)
            toks.append(int(cur.item()))
            ts.append(time.perf_counter() - t0); t0 = time.perf_counter()
        return toks, ts

    res = {}
    for rep in range(3):
        for name, fn in (("sync", run_sync), ("async", run_async)):
            toks, ts = fn()
            res.setdefault(name, []).append(statistics.median(ts[4:]) * 1e3)
            res.setdefault(name + "_toks", toks)
    b = run_build()
    emit({"what": "loop", "sync_ms": round(statistics.median(res["sync"]), 2), "async_ms": round(statistics.median(res["async"]), 2),
          "build_ms": round(statistics.median(b[2:]) * 1e3, 2),
          "same_tokens": res["sync_toks"][:N] == res["async_toks"][:N], **g.report()})

    # matmul floor
    qls = [m for name, m in model.named_modules() if isinstance(m, nn.QuantizedLinear) and ".mtp." not in f".{name}."
           and not name.startswith("mtp")]
    xs = [mx.random.normal((1, 1, m.weight.shape[1] * 32 // m.bits)).astype(mx.bfloat16) for m in qls]
    mx.eval(xs)
    nbytes = sum(m.weight.nbytes + m.scales.nbytes + (m.biases.nbytes if m.biases is not None else 0) for m in qls)
    emb = model.language_model.model.embed_tokens if hasattr(model, "language_model") else model.model.embed_tokens
    tied = not any(n.endswith("lm_head") for n, _ in model.named_modules())
    ts = []
    for _ in range(7):
        outs = [m(x) for m, x in zip(qls, xs)]
        if tied:
            outs.append(emb.as_linear(xs[0][..., :emb.weight.shape[1] * 32 // emb.bits]))
        t0 = time.perf_counter(); mx.eval(outs); ts.append(time.perf_counter() - t0)
    emit({"what": "matmul_floor", "n_linear": len(qls), "tied_head": tied, "gb": round(nbytes / 2**30, 2),
          "ms": round(statistics.median(ts[1:]) * 1e3, 2), "gb_s": round(nbytes / 2**30 / statistics.median(ts[1:]), 1)})

    # per layer type, decode step on a real cache (no cache update kept: fresh cache copy per repeat is too costly,
    # so the timed graphs run on a cache that grows by one per repeat; negligible at this length)
    inner = model.language_model.model if hasattr(model, "language_model") else model.model
    c, t = fresh()
    h = mx.random.normal((1, 1, inner.layers[0].input_layernorm.weight.shape[0])).astype(mx.bfloat16)
    mx.eval(h)
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    kinds = {}
    for rep in range(5):
        for kind in ("gdn_mixer", "attn_mixer", "mlp", "norms"):
            outs = []
            for li, layer in enumerate(inner.layers):
                x = h
                if kind == "gdn_mixer" and layer.is_linear:
                    outs.append(layer.linear_attn(layer.input_layernorm(x), mask=None, cache=c[li]))
                elif kind == "attn_mixer" and not layer.is_linear:
                    outs.append(layer.self_attn(layer.input_layernorm(x), mask=None, cache=c[li]))
                elif kind == "mlp":
                    outs.append(layer.mlp(x))
                elif kind == "norms":
                    outs.append(layer.post_attention_layernorm(layer.input_layernorm(x)))
            t0 = time.perf_counter(); mx.eval(outs); kinds.setdefault(kind, []).append(time.perf_counter() - t0)
    emit({"what": "layer_types_ms", **{k: round(statistics.median(v[1:]) * 1e3, 2) for k, v in kinds.items()}, **g.report()})
