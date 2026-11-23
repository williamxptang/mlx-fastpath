"""Pipelined plain decoding (generate_ar) gives exactly the greedy tokens, stops at EOS and at max_tokens.
Tiny random Qwen3.5 (hybrid GDN + attention), so the cache path of both layer types is exercised."""
import types

import mlx.core as mx
import pytest
from mlx.utils import tree_map
from mlx_lm.models import qwen3_5

import mlx_fastpath as fp

TEXT = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=300,
            linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
            full_attention_interval=4, tie_word_embeddings=False,
            rope_parameters={"type": "default", "mrope_section": [2, 2, 0], "rope_theta": 10000,
                             "partial_rotary_factor": 0.25})
PROMPT = [5, 17, 99, 3, 250, 41, 7, 7, 120, 64, 33, 8, 200, 150]


@pytest.fixture(scope="module")
def model():
    mx.random.seed(0)
    m = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=TEXT))
    m.update(tree_map(lambda p: p * 4 if p.ndim == 2 else p, m.parameters()))   # sharper logits than the init
    mx.eval(m.parameters())
    return m


def engine(model, eos=()):
    return types.SimpleNamespace(_model=model, _eos_ids=lambda: set(eos))


def greedy_without_cache(model, ids, n):
    """Reference: recompute the whole sequence for every token (no cache, no pipelining)."""
    ids, out = list(ids), []
    for _ in range(n):
        t = int(mx.argmax(model(mx.array(ids)[None])[0, -1]).item())
        out.append(t)
        ids.append(t)
    return out


@pytest.mark.parametrize("step", [1024, 4])    # one prefill chunk, and several
def test_pipelined_decoding_gives_the_greedy_tokens(model, step):
    want = greedy_without_cache(model, PROMPT, 24)
    assert len(set(want)) > 3                                           # a real continuation, not one token
    got = fp.generate_ar(engine(model), PROMPT, max_tokens=24, prefill_step=step)
    assert got["tokens"] == want
    assert got["tok_s"] > 0


def test_stops_before_the_eos_token(model):
    want = greedy_without_cache(model, PROMPT, 24)
    eos = want[9]
    got = fp.generate_ar(engine(model, eos=[eos]), PROMPT, max_tokens=24)
    assert got["tokens"] == want[:want.index(eos)]


@pytest.mark.parametrize("n", [1, 2, 7])
def test_returns_exactly_max_tokens(model, n):
    assert fp.generate_ar(engine(model), PROMPT, max_tokens=n)["tokens"] == greedy_without_cache(model, PROMPT, n)
