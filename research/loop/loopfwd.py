"""Looped forward for mlx-lm Qwen3.5 (qwen3_5): re-run a block of layers [s, e) on each token's hidden state.

    h = layers[0:s](embed(ids))
    pass 1:   h = layers[s:e](h)                               (the normal forward)
    pass p>1: h = h + alpha_p * (layers[s:e](h) - h)           (own caches per pass)
    logits = head(norm(layers[e:](h)))

Weights are untouched. Each extra pass gets its OWN cache objects for the block's layers: GDN layers keep a running
state and attention layers keep keys/values, and pass p must see pass p's history, not pass 1's.
Masks are built exactly as the stock forward does (qwen3_5.py: fa_mask from a full-attention cache, ssm_mask from a
GDN cache) and shared by all passes: every pass's caches sit at the same offset.
"""
import copy
import math

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask


def _parts(model):
    tm = model.language_model if hasattr(model, "language_model") else model
    return tm, tm.model


class LoopCache:
    """Caches for the looped model: `base` (one per layer, as model.make_cache()) and `extra[p]` for passes 2..P
    (one per block layer)."""

    def __init__(self, model, block, passes):
        tm, _ = _parts(model)
        self.base = tm.make_cache()
        s, e = block
        self.extra = {p: [copy.deepcopy(self.base[i]) for i in range(s, e)] for p in range(2, passes + 1)}



def forward(model, ids, cache=None, block=None, passes=1, alphas=None, return_hidden=False,
            mixer=None, pass_layers=None):
    """Logits for ids (shape (1, L)). block=(s, e), passes P >= 1, alphas: {p: alpha} for p = 2..P (default 1.0).
    cache: a LoopCache built with the same block and passes (or None for a cache-free full-sequence forward).
    mixer: optional callable (p, h, x) -> new h, replacing the fixed alpha blend (a learned gate).
    pass_layers: optional {p: [modules for the block]} run in pass p instead of the shared layers (T2: per-pass LoRA)."""
    tm, inner = _parts(model)
    layers = inner.layers
    n = len(layers)
    s, e = block if block is not None else (n, n)
    alphas = alphas or {}
    base = cache.base if cache is not None else [None] * n

    h = inner.embed_tokens(ids)
    fa_mask = create_attention_mask(h, base[inner.fa_idx]) if inner.fa_idx is not None else None
    ssm_mask = create_ssm_mask(h, base[inner.ssm_idx]) if inner.ssm_idx is not None else None

    def run(i, x, c):
        return layers[i](x, mask=ssm_mask if layers[i].is_linear else fa_mask, cache=c)

    for i in range(0, s):
        h = run(i, h, base[i])
    for k, i in enumerate(range(s, e)):                          # pass 1 (pass_layers[1]: a no-loop LoRA control)
        if pass_layers is not None and 1 in pass_layers:
            li = pass_layers[1][k]
            h = li(h, mask=ssm_mask if li.is_linear else fa_mask, cache=base[i])
        else:
            h = run(i, h, base[i])
    for p in range(2, passes + 1):
        x = h
        for k, i in enumerate(range(s, e)):
            c = cache.extra[p][k] if cache is not None else None
            if pass_layers is not None and p in pass_layers:
                li = pass_layers[p][k]
                x = li(x, mask=ssm_mask if li.is_linear else fa_mask, cache=c)
            else:
                x = run(i, x, c)
        if mixer is not None:
            h = mixer(p, h, x)
            continue
        a = alphas.get(p, 1.0)
        h = x if a == 1.0 else h + a * (x - h)          # exact at full strength
    for i in range(e, n):
        h = run(i, h, base[i])

    h = inner.norm(h)
    if return_hidden:
        return h
    return head(model, h)


def head(model, h):
    tm, inner = _parts(model)
    if tm.args.tie_word_embeddings:
        return inner.embed_tokens.as_linear(h)
    return tm.lm_head(h)


class Gate(nn.Module):
    """A learned mix for the extra passes: scalar, per-channel, or low-rank per-token (see train_gate.py)."""
    def __init__(self, kind, hidden, a0, rank=32, stop_grad_x=True):
        super().__init__()
        self.kind = kind
        self.stop_grad_x = stop_grad_x
        w0 = math.log(a0 / (1 - a0))
        if kind == "scalar":
            self.w = mx.array([w0])
        elif kind == "channel":
            self.w = mx.full((hidden,), w0)
        elif kind == "adapter":                  # CONTROL: same size, no use of the extra pass (x is ignored)
            self.V = nn.Linear(hidden, 3 * rank // 2, bias=False)
            self.U = nn.Linear(3 * rank // 2, hidden, bias=False)
            self.U.weight = mx.zeros_like(self.U.weight)
        elif kind == "lowrank":
            self.b = mx.full((hidden,), w0)
            self.V = nn.Linear(2 * hidden, rank, bias=False)
            self.U = nn.Linear(rank, hidden, bias=False)
            self.U.weight = mx.zeros_like(self.U.weight)
        else:
            raise ValueError(kind)

    def alpha(self, h, x):
        if self.kind == "adapter":               # the control mixes in nothing from the extra pass
            return mx.zeros(h.shape[-1:])
        if self.kind in ("scalar", "channel"):
            return mx.sigmoid(self.w)
        z = mx.concatenate([h, x], axis=-1).astype(mx.float32)
        return mx.sigmoid(self.b + self.U(nn.relu(self.V(z))))

    def __call__(self, p, h, x):
        # The base weights are frozen, so nothing before the mix needs a gradient: cutting the graph here keeps the
        # entry layers and the extra pass out of backprop (memory), the gate still learns through the layers after it.
        h = mx.stop_gradient(h)
        if self.stop_grad_x:                     # T2 trains the extra pass itself: keep its gradient
            x = mx.stop_gradient(x)
        if self.kind == "adapter":
            return (h.astype(mx.float32) + self.U(nn.relu(self.V(h.astype(mx.float32))))).astype(h.dtype)
        a = self.alpha(h, x)
        return (h.astype(mx.float32) + a * (x - h).astype(mx.float32)).astype(h.dtype)




def lora_pass_layers(model, block, rank=8, scale=20.0):
    """Copies of the block's layers for an extra pass, every Linear wrapped in a LoRA adapter (B starts at zero, so
    the copies start equal to the originals). Only lora_a / lora_b are trainable; the model itself is not touched."""
    from mlx.utils import tree_unflatten
    from mlx_lm.tuner.lora import LoRALinear
    _, inner = _parts(model)
    s, e = block
    out = []
    for i in range(s, e):
        layer = copy.deepcopy(inner.layers[i])
        swaps = [(name, LoRALinear.from_base(m, r=rank, scale=scale, dropout=0.0))
                 for name, m in layer.named_modules() if isinstance(m, (nn.Linear, nn.QuantizedLinear))]
        layer.update_modules(tree_unflatten(swaps))
        layer.freeze()
        layer.unfreeze(keys=["lora_a", "lora_b"], strict=False)
        out.append(layer)
    return out
