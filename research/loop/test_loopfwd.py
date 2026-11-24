"""Looped forward on a tiny random Qwen3.5 (same hybrid GDN + attention layout as the real models, a few MB):
off means stock, and token-by-token decoding with per-pass caches equals one full-sequence forward."""
import sys
from pathlib import Path

import mlx.core as mx
import pytest
from mlx_lm.models import qwen3_5

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loopfwd  # noqa: E402

IDS = [5, 17, 42, 9, 101, 33, 7, 64, 2, 88, 19, 140, 3, 77, 250, 12]


def fresh_model():
    mx.random.seed(0)
    args = qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": dict(
        model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=1, head_dim=16, vocab_size=256, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, full_attention_interval=4)})
    m = qwen3_5.Model(args)
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module")
def model():
    return fresh_model()


def stock_logits(model, ids):
    return model(mx.array([ids]))


def test_a_single_pass_is_the_stock_forward(model):
    a = stock_logits(model, IDS)
    b = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=1)
    assert mx.array_equal(a, b).item()


def test_a_second_pass_with_alpha_zero_changes_nothing(model):
    a = stock_logits(model, IDS)
    b = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=2, alphas={2: 0.0})
    assert mx.array_equal(a, b).item()


def test_a_real_second_pass_changes_the_output(model):
    a = stock_logits(model, IDS)
    b = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=2, alphas={2: 0.5})
    assert mx.abs(a - b).max().item() > 1e-3


class unrolled:
    """The same model with the block's layers listed again after themselves (passes times), run by the STOCK forward
    with stock caches: the reference for a loop at alpha = 1."""

    def __init__(self, model, block, passes):
        self.inner = model.language_model.model
        s, e = block
        self.orig = self.inner.layers
        self.new = self.orig[:e] + self.orig[s:e] * (passes - 1) + self.orig[e:]

    def __enter__(self):
        self.inner.layers = self.new
        return self

    def __exit__(self, *exc):
        self.inner.layers = self.orig


def decode(step, prefix=9):
    """Prefill IDS[:prefix], then one token at a time: logits from position prefix-1 on."""
    out = [step(IDS[:prefix])[:, -1:]]
    out += [step([t]) for t in IDS[prefix:]]
    return mx.concatenate(out, axis=1)


@pytest.mark.parametrize("block,passes", [((2, 6), 2), ((0, 8), 3), ((4, 8), 2), ((3, 4), 2)])
@pytest.mark.parametrize("mode", ["full-sequence", "token-by-token"])
def test_a_full_strength_loop_equals_the_unrolled_model(block, passes, mode):
    """Per-pass caches (GDN state and attention keys) make the loop identical to listing the block's layers again.
    (Comparing token-by-token with full-sequence directly does not work: Qwen3.5's GDN layers already give different
    numbers in the two modes in the stock model.)"""
    from mlx_lm.models.cache import make_prompt_cache
    model = fresh_model()                # unrolled() swaps the layer list: keep it off the shared fixture
    with unrolled(model, block, passes):
        ref_cache = make_prompt_cache(model)
        ref = (model(mx.array([IDS])) if mode == "full-sequence"
               else decode(lambda ids: model(mx.array([ids]), cache=ref_cache)))
    cache = loopfwd.LoopCache(model, block, passes)
    got = (loopfwd.forward(model, mx.array([IDS]), block=block, passes=passes) if mode == "full-sequence"
           else decode(lambda ids: loopfwd.forward(model, mx.array([ids]), cache=cache, block=block, passes=passes)))
    assert ref.shape == got.shape and ref.shape[1] > 1
    assert mx.allclose(ref, got, atol=1e-5, rtol=1e-5).item()


def test_each_pass_keeps_its_own_cache(model):
    block, passes = (2, 6), 3
    cache = loopfwd.LoopCache(model, block, passes)
    loopfwd.forward(model, mx.array([IDS]), cache=cache, block=block, passes=passes)
    objs = [id(c) for c in cache.base[2:6]] + [id(c) for p in (2, 3) for c in cache.extra[p]]
    assert len(set(objs)) == len(objs) == 12
    assert all(getattr(c, "offset", len(IDS)) == len(IDS) for p in (2, 3) for c in cache.extra[p])


def test_a_mixer_that_returns_the_old_state_changes_nothing(model):
    a = stock_logits(model, IDS)
    b = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=2, mixer=lambda p, h, x: h)
    assert mx.array_equal(a, b).item()


def test_a_mixer_receives_each_extra_pass_and_can_blend(model):
    seen = []

    def mix(p, h, x):
        seen.append(p)
        return h + 0.5 * (x - h)

    a = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=3, alphas={2: 0.5, 3: 0.5})
    b = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=3, mixer=mix)
    assert seen == [2, 3]
    assert mx.allclose(a, b, atol=1e-6).item()


@pytest.mark.parametrize("kind", ["scalar", "channel", "lowrank"])
def test_an_untrained_gate_equals_the_fixed_alpha_loop(model, kind):
    a = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=2, alphas={2: 0.25})
    g = loopfwd.Gate(kind, 64, 0.25, rank=8)
    b = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=2, mixer=g)
    assert mx.allclose(a, b, atol=1e-5).item()


@pytest.mark.parametrize("kind", ["scalar", "channel", "lowrank"])
def test_training_moves_only_the_gate(kind):
    model = fresh_model()                # its own copy: freeze() must not leak into other tests
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    model.freeze()
    before = {k: v for k, v in tree_flatten(model.parameters())}
    assert not tree_flatten(model.trainable_parameters())     # the base is frozen: only the gate can learn
    g = loopfwd.Gate(kind, 64, 0.25, rank=8)
    g0 = {k: mx.array(v) for k, v in tree_flatten(g.parameters())}
    x = mx.array([IDS])

    def loss_fn(gate):
        lg = loopfwd.forward(model, x[:, :-1], block=(2, 6), passes=2, mixer=gate)
        return nn.losses.cross_entropy(lg[0], x[0, 1:], reduction="mean")

    vg = nn.value_and_grad(g, loss_fn)
    opt = optim.Adam(learning_rate=1e-1)
    losses = []
    for _ in range(5):
        loss, grads = vg(g)
        opt.update(g, grads)
        mx.eval(g.parameters(), opt.state)
        losses.append(loss.item())
    after = {k: v for k, v in tree_flatten(model.parameters())}
    assert all(mx.array_equal(before[k], after[k]).item() for k in before)
    moved = [k for k, v in tree_flatten(g.parameters()) if not mx.array_equal(v, g0[k]).item()]
    assert moved, "the gate must learn something"
    assert losses[-1] < losses[0]


def test_fresh_lora_pass_layers_start_equal_to_the_shared_ones(model):
    layers = loopfwd.lora_pass_layers(model, (2, 6), rank=4)
    a = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=2, alphas={2: 0.5})
    b = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=2, alphas={2: 0.5}, pass_layers={2: layers})
    assert mx.allclose(a, b, atol=1e-6).item()


def test_lora_pass_training_moves_only_the_adapters_and_the_gate():
    model = fresh_model()
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    model.freeze()
    before = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}

    class T2(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = loopfwd.Gate("channel", 64, 0.25, stop_grad_x=False)
            self.layers = loopfwd.lora_pass_layers(model, (2, 6), rank=4)

    t2 = T2()
    trainable = [k for k, _ in tree_flatten(t2.trainable_parameters())]
    assert trainable and all("lora_" in k or k.startswith("gate.") for k in trainable)
    x = mx.array([IDS])

    def loss_fn(m):
        lg = loopfwd.forward(model, x[:, :-1], block=(2, 6), passes=2, mixer=m.gate, pass_layers={2: m.layers})
        return nn.losses.cross_entropy(lg[0], x[0, 1:], reduction="mean")

    vg = nn.value_and_grad(t2, loss_fn)
    opt = optim.Adam(learning_rate=5e-2)
    losses = []
    for _ in range(6):
        loss, grads = vg(t2)
        opt.update(t2, grads)
        mx.eval(t2.parameters(), opt.state)
        losses.append(loss.item())
    after = {k: v for k, v in tree_flatten(model.parameters())}
    assert all(mx.array_equal(before[k], after[k]).item() for k in before)
    lora_b = [v for k, v in tree_flatten(t2.parameters()) if k.endswith("lora_b")]
    assert lora_b and any(mx.abs(v).max().item() > 0 for v in lora_b)
    assert losses[-1] < losses[0]


def test_the_control_adapter_starts_as_the_plain_model_and_ignores_the_extra_pass(model):
    from mlx.utils import tree_flatten
    g = loopfwd.Gate("adapter", 64, 0.1, rank=8)
    a = stock_logits(model, IDS)
    b = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=2, mixer=g)
    assert mx.allclose(a, b, atol=1e-5).item()
    g.U.weight = mx.ones_like(g.U.weight) * 0.01                    # once trained it changes the output ...
    c1 = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=2, mixer=g)
    c2 = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=2, alphas={2: 0.0}, mixer=lambda p, h, x: g(p, h, h))
    assert mx.abs(c1 - a).max().item() > 1e-4
    assert mx.allclose(c1, c2, atol=1e-5).item()                    # ... without using the extra pass
    n_lowrank = sum(v.size for _, v in tree_flatten(loopfwd.Gate("lowrank", 64, 0.1, rank=8).parameters()))
    n_adapter = sum(v.size for _, v in tree_flatten(g.parameters()))
    assert abs(n_adapter - n_lowrank) / n_lowrank < 0.1               # same size as the gate it controls for


@pytest.mark.parametrize("kind,expected", [("scalar", 0.1), ("channel", 0.1), ("lowrank", 0.1), ("adapter", 0.0)])
def test_every_gate_kind_starts_at_its_mix(kind, expected):
    g = loopfwd.Gate(kind, 64, 0.1, rank=8)
    a = g.alpha(mx.ones((1, 1, 64)), mx.ones((1, 1, 64)))
    assert a.size >= 1
    assert mx.allclose(a, mx.full(a.shape, expected), atol=1e-5).item()     # untrained: the starting alpha (control: none)


def test_a_fresh_lora_in_pass_one_is_the_stock_model(model):
    layers = loopfwd.lora_pass_layers(model, (2, 6), rank=4)
    a = stock_logits(model, IDS)
    b = loopfwd.forward(model, mx.array([IDS]), block=(2, 6), passes=1, pass_layers={1: layers})
    assert mx.allclose(a, b, atol=1e-6).item()
