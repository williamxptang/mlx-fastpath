"""T7: GPTQ instead of round-to-nearest for OptiQ's 8-bit layers -> 4-bit, one decoder layer at a time.

Batch 2's N2-all requantized every 8-bit layer (all but embed_tokens) with plain rounding and failed the needle tasks.
This builds the same layer set with GPTQ (arXiv 2210.17323), so the comparison isolates the quantizer:
  - weights: the original bf16 Qwen/Qwen3.5-9B (default), or OptiQ's own 8-bit values (--weights optiq8);
  - calibration: mlx-lm's standard calibration text (mlx_lm.quant.utils.load_data), 128 x 512 tokens, seed 123;
  - sequential: layer b's Hessians come from the output of the already-quantized layers 0..b-1; OptiQ's 4-bit
    layers are used as they are; the LM head's Hessian comes from the final norm's output.
mlx-lm's own gptq.py can't do this on a 9B in 24 GB: it holds every Hessian at once (~19 GB) and needs a float model.
Its in-group update also writes past the group (W[..., k:k+j] with absolute k), which the block update then applies a
second time; gptq() below is the paper's lazy-batch form.

Memory: one layer's float weights, its Hessians (<= 0.6 GB for down_proj), the calibration activations (2 x 0.54 GB)
and the LM head in row chunks: ~3-4 GB.

Usage (writes stage files, then the model folder through requant.convert; resumes from stage files):
  python gptq_stream.py SRC DST [--variant all|gdn_head|gdn_head_mlp] [--weights bf16|optiq8] [--bf16 DIR]
                       [--samples 128] [--seq 512]
"""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "batch2"))
import requant  # noqa: E402

BITS, GROUP, DAMP = 4, 64, 0.01
HEAD = "language_model.lm_head"
ROW_BYTES = 512 << 20          # float32 bytes of W per GPTQ row chunk (rows are independent given H)


# ---------- GPTQ on one weight ----------
def inverse_hessian(H):
    """Upper Cholesky factor of H^-1 after dead-column handling and dampening (the paper's form)."""
    with mx.stream(mx.cpu):
        H = H.astype(mx.float32)
        n = H.shape[0]
        d = mx.diag(H)
        dead = d == 0
        idx = mx.arange(n)
        H[idx, idx] = mx.where(dead, 1.0, d) + DAMP * mx.mean(mx.where(dead, 1.0, d))
        L = mx.linalg.cholesky(H)
        Hinv = mx.linalg.cholesky(mx.linalg.cholesky_inv(L), upper=True)
        mx.eval(Hinv, dead)
    return Hinv, dead


def _scales(w):
    """MLX's affine scales/biases for these columns, rounded to bf16 as stored, back in float32."""
    _, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
    return s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def gptq(W, Hinv, dead, propagate=True):
    """W (rows, cols) float -> (wq uint32, scales bf16, biases bf16, codes float32 (rows, cols)).
    propagate=False skips the error feedback (a broken GPTQ, for the tests)."""
    W = W.astype(mx.float32)
    W = mx.where(dead[None, :], 0.0, W)
    rows, cols = W.shape
    n_bins = 2 ** BITS - 1
    S, B, C = [], [], []
    for i in range(0, cols, GROUP):
        j = i + GROUP
        W1 = W[:, i:j]
        s16, b16 = _scales(W1)
        s, b = s16.astype(mx.float32), b16.astype(mx.float32)
        Hb = Hinv[i:j, i:j]
        err = []
        codes = []
        for k in range(GROUP):
            w = W1[:, k:k + 1]
            c = mx.clip(mx.round((w - b) / s), 0, n_bins)
            e = (w - (c * s + b)) / Hb[k, k]
            if propagate and k + 1 < GROUP:
                W1 = mx.concatenate([W1[:, :k + 1], W1[:, k + 1:] - e @ Hb[k:k + 1, k + 1:]], axis=1)
            err.append(e)
            codes.append(c)
            mx.eval(W1, e, c)
        E = mx.concatenate(err, axis=1)
        if propagate and j < cols:
            W = mx.concatenate([W[:, :j], W[:, j:] - E @ Hinv[i:j, j:]], axis=1)
        S.append(s16); B.append(b16); C.append(mx.concatenate(codes, axis=1))
        mx.eval(W)
    codes = mx.concatenate(C, axis=1)
    scales, biases = mx.concatenate(S, axis=1), mx.concatenate(B, axis=1)
    return pack(codes), scales, biases, codes


def pack(codes):
    """4-bit codes (rows, cols) -> MLX's uint32 layout (8 codes per word, low bits first)."""
    per = 32 // BITS
    c = codes.astype(mx.uint32).reshape(codes.shape[0], -1, per)
    shifts = mx.power(2, mx.arange(0, 32, BITS, dtype=mx.uint32))
    return mx.sum(c * shifts, axis=-1).astype(mx.uint32)


def gptq_rows(W, H):
    """GPTQ over row chunks of W (rows are independent given H). Returns (wq, scales, biases)."""
    Hinv, dead = inverse_hessian(H)
    rows, cols = W.shape
    step = max(1, ROW_BYTES // (4 * cols))
    parts = [gptq(W[r:r + step], Hinv, dead)[:3] for r in range(0, rows, step)]
    out = tuple(mx.concatenate([p[x] for p in parts]) for x in range(3))
    mx.eval(*out)
    return out


def proxy_loss(W, Q, H):
    """tr((W-Q) H (W-Q)^T): the layer-output error GPTQ minimizes."""
    D = (W.astype(mx.float32) - Q.astype(mx.float32))
    return float(mx.sum((D @ H.astype(mx.float32)) * D))


# ---------- weights ----------
def hf_key(m):
    """mlx module path -> key in the original Hugging Face checkpoint."""
    if m == HEAD:
        return "lm_head.weight"
    assert m.startswith("language_model.model."), m
    return "model.language_model." + m[len("language_model.model."):] + ".weight"


class FloatSource:
    """Float weights of the layers to quantize, read by row ranges (never a whole checkpoint in memory)."""

    def __init__(self, src, kind, bf16_dir=None):
        self.kind = kind
        if kind == "bf16":
            d = Path(bf16_dir)
            self.idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
            self.dir = d
        else:
            self.idx = json.load(open(Path(src) / "model.safetensors.index.json"))["weight_map"]
            self.dir = Path(src)
        self._h = {}

    def _header(self, shard):
        if shard not in self._h:
            self._h[shard] = requant.read_header(self.dir / shard)[:2]
        return self._h[shard]

    def shape(self, m):
        k = hf_key(m) if self.kind == "bf16" else f"{m}.weight"
        h, _ = self._header(self.idx[k])
        s = h[k]["shape"]
        return (s[0], s[1] * 4) if self.kind == "optiq8" else tuple(s)

    def rows(self, m, r0=None, r1=None):
        if self.kind == "bf16":
            k = hf_key(m)
            shard = self.idx[k]
            h, start = self._header(shard)
            return requant.read_rows(self.dir / shard, start, h[k], r0, r1)
        shard = self.idx[f"{m}.weight"]
        h, start = self._header(shard)
        w, s, b = (requant.read_rows(self.dir / shard, start, h[f"{m}.{x}"], r0, r1) for x in ("weight", "scales", "biases"))
        return mx.dequantize(w, s, b, group_size=GROUP, bits=8)


# ---------- the sequential pass ----------
class Catcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.H = None

    def __call__(self, x):
        xf = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        h = xf.T @ xf
        self.H = h if self.H is None else self.H + h
        return self.module(x)


def stage_path(stage, m):
    return Path(stage) / f"{m}.safetensors"


def save_stage(stage, m, wq, s, b):
    tmp = stage_path(stage, m).with_suffix(".partial.safetensors")
    mx.save_safetensors(str(tmp), {"weight": wq, "scales": s, "biases": b})
    tmp.rename(stage_path(stage, m))


def quantized_linear(wq, s, b):
    out_dims, in_dims = wq.shape[0], wq.shape[1] * 32 // BITS
    q = nn.QuantizedLinear(in_dims, out_dims, bias=False, group_size=GROUP, bits=BITS)
    q.weight, q.scales, q.biases = wq, s, b
    return q


def run_layers(model, targets, fsrc, tokens, stage, batch=8, log=print):
    """Quantize every target linear in the decoder layers, then the LM head, writing stage files. Returns the
    final normed hidden states of the calibration tokens through the quantized layers.
    model: an mlx_lm Qwen3.5 Model (lazy); targets: set of module paths; tokens: (N, S) int array."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    tm = model.language_model.model
    X = mx.concatenate([tm.embed_tokens(tokens[i:i + batch]) for i in range(0, tokens.shape[0], batch)])
    mx.eval(X)
    tm.embed_tokens = nn.Identity()                     # drop the 8-bit embedding (~1 GB) once used
    mx.clear_cache()
    fa = create_attention_mask(X[:batch], None)
    ssm = create_ssm_mask(X[:batch], None)
    for li in range(len(tm.layers)):
        layer = tm.layers[li]
        mask = ssm if layer.is_linear else fa
        prefix = f"language_model.model.layers.{li}."
        mine = [(p, prefix + p) for p, _ in tree_flatten(layer.leaf_modules(), is_leaf=nn.Module.is_module)
                if prefix + p in targets]
        t0 = time.time()
        todo = [(p, m) for p, m in mine if not stage_path(stage, m).exists()]
        if todo:
            cats = {}
            for p, m in todo:
                w = fsrc.rows(m)
                lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
                lin.weight = w.astype(mx.bfloat16)
                cats[p] = Catcher(lin)
            layer.update_modules(tree_unflatten(list(cats.items())))
            for i in range(0, X.shape[0], batch):
                layer(X[i:i + batch], mask=mask, cache=None)
                mx.eval([c.H for c in cats.values()])
            for p, m in todo:
                c = cats[p]
                wq, s, b = gptq_rows(c.module.weight, c.H)
                save_stage(stage, m, wq, s, b)
                cats[p] = None
                mx.clear_cache()
        # the layer's output from its quantized form (stage files: new or from an earlier, stopped run)
        if mine:
            q = []
            for p, m in mine:
                d = mx.load(str(stage_path(stage, m)))
                q.append((p, quantized_linear(d["weight"], d["scales"], d["biases"])))
            layer.update_modules(tree_unflatten(q))
        X = mx.concatenate([layer(X[i:i + batch], mask=mask, cache=None) for i in range(0, X.shape[0], batch)])
        mx.eval(X)
        tm.layers[li] = nn.Identity()                    # free this layer's weights
        mx.clear_cache()
        log(json.dumps({"layer": li, "quantized": len(todo), "staged": len(mine) - len(todo),
                        "seconds": round(time.time() - t0, 1), "active_gb": round(mx.get_active_memory() / 2**30, 2),
                        "peak_gb": round(mx.get_peak_memory() / 2**30, 2)}))
    Xn = tm.norm(X)
    mx.eval(Xn)
    del X
    if HEAD in targets and not stage_path(stage, HEAD).exists():
        t0 = time.time()
        H = None
        for i in range(0, Xn.shape[0], batch):
            xf = Xn[i:i + batch].reshape(-1, Xn.shape[-1]).astype(mx.float32)
            H = xf.T @ xf if H is None else H + xf.T @ xf
            mx.eval(H)
        mx.clear_cache()
        Hinv, dead = inverse_hessian(H)
        rows, cols = fsrc.shape(HEAD)
        step = max(1, ROW_BYTES // (4 * cols))
        parts = []
        for r in range(0, rows, step):
            wq, s, b, _ = gptq(fsrc.rows(HEAD, r, min(r + step, rows)), Hinv, dead)
            mx.eval(wq, s, b)
            parts.append((wq, s, b))
            mx.clear_cache()
        save_stage(stage, HEAD, *(mx.concatenate([p[x] for p in parts]) for x in range(3)))
        log(json.dumps({"layer": "lm_head", "seconds": round(time.time() - t0, 1),
                        "peak_gb": round(mx.get_peak_memory() / 2**30, 2)}))
    return Xn                                           # final normed hidden states (for the tests)


def stage_rows(stage):
    """requant.convert's rows_fn: the staged 4-bit rows of a layer."""
    def rows_fn(path, start, h, m, r0, r1):
        p = stage_path(stage, m)
        hs, st, _ = requant.read_header(p)
        return tuple(requant.read_rows(p, st, hs[x], r0, r1) for x in ("weight", "scales", "biases"))
    return rows_fn


def build(src, dst, weights="bf16", bf16_dir=None, tokens=None, samples=128, seq=512, batch=8, log=print,
          variant="all"):
    from mlx_lm.utils import load_model
    src, dst = Path(src), Path(dst)
    stage = dst.with_name(dst.name + ".stage")
    stage.mkdir(exist_ok=True)
    config = json.load(open(src / "config.json"))
    targets = requant.selected(config, variant)
    model, _ = load_model(src, lazy=True)
    if tokens is None:
        from mlx_lm.quant.utils import load_data
        from mlx_lm.utils import load_tokenizer
        mx.random.seed(123)
        tokens = load_data(load_tokenizer(src), samples, seq)
    fsrc = FloatSource(src, weights, bf16_dir)
    for m in targets:                                    # shapes agree before any work
        want = model_shape(model, m)
        assert fsrc.shape(m) == want, (m, fsrc.shape(m), want)
    run_layers(model, set(targets), fsrc, tokens, stage, batch=batch, log=log)
    del model
    mx.clear_cache()
    missing = [m for m in targets if not stage_path(stage, m).exists()]
    assert not missing, missing[:5]
    requant.convert(src, dst, f"gptq-{variant}-{weights}", modules=targets, rows_fn=stage_rows(stage),
                    what=f"OptiQ 8-bit layers -> 4-bit with GPTQ (weights: {weights}, calibration: mlx-lm "
                         f"calibration_v5, {tokens.shape[0]}x{tokens.shape[1]} tokens), local research copy")
    return targets


def model_shape(model, m):
    mod = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))[m]
    return (mod.weight.shape[0], mod.weight.shape[1] * 32 // mod.bits)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--weights", choices=["bf16", "optiq8"], default="bf16")
    ap.add_argument("--bf16", default=MODELS + "/Qwen3.5-9B-bf16")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--variant", default="all", help="requant.py layer set: all | gdn_head | gdn_head_mlp")
    a = ap.parse_args()
    import os
    import mlx_fastpath as fp
    with fp.MemoryGuard(min_free_gb=float(os.environ.get("GUARD_FREE_GB", 6)),
                        max_footprint_gb=float(os.environ.get("GUARD_CAP_GB", 6))) as g:
        mx.set_memory_limit(int(float(os.environ.get("MLX_LIMIT_GB", 5)) * 2**30))
        mx.set_cache_limit(256 << 20)
        t = build(a.src, a.dst, a.weights, a.bf16, samples=a.samples, seq=a.seq,
                  log=lambda s: print(s, flush=True), variant=a.variant)
        print(json.dumps({"converted": len(t), "guard": g.report()}))
