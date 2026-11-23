"""kl_gate.py on a tiny random Qwen3.5 (hybrid GDN + attention, a few MB). Run:
<venv>/bin/python -m pytest -q -p no:cacheprovider test_kl_gate.py"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_map
from mlx_lm.models import qwen3_5

import kl_gate

TEXT = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=300,
            linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
            full_attention_interval=4, tie_word_embeddings=False,
            rope_parameters={"type": "default", "mrope_section": [2, 2, 0], "rope_theta": 10000,
                             "partial_rotary_factor": 0.25})
PROMPT = [5, 17, 99, 3, 250, 41, 7, 7, 120, 64, 33]
N = 40


def tiny(seed=0):
    mx.random.seed(seed)
    m = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=TEXT))
    m.update(tree_map(lambda p: p * 4 if p.ndim == 2 else p, m.parameters()))   # sharper logits than the init
    mx.eval(m.parameters())
    return m.language_model


def reference(lm):
    rows = {}
    ids = kl_gate.greedy_reference(lm, PROMPT, N, eos=set(), on_row=lambda i, lp: rows.__setitem__(i, np.array(lp)))
    return ids, np.stack([rows[i] for i in range(len(ids))])


@pytest.mark.parametrize("ref_lp, cand_lp, kl, flip", [
    ([0.5, 0.5], [0.5, 0.5], 0.0, False),
    ([0.9, 0.1], [0.1, 0.9], 0.9 * math.log(9) + 0.1 * math.log(1 / 9), True),
    ([0.7, 0.3], [0.6, 0.4], 0.7 * math.log(7 / 6) + 0.3 * math.log(3 / 4), False),
])
def test_compare_gives_kl_and_flips(ref_lp, cand_lp, kl, flip):
    m = kl_gate.compare(mx.log(mx.array([ref_lp])), mx.log(mx.array([cand_lp])))
    assert float(m["kl"][0]) == pytest.approx(kl, abs=1e-6)
    assert bool(m["flip"][0]) is flip


def test_the_reference_scored_against_itself_is_a_near_zero_floor():
    lm = tiny()
    ids, ref = reference(lm)
    assert len(ids) == N and len(set(ids)) > 3            # a real continuation, not one token repeated
    for chunk in (1, 7, 64):                                # chunked teacher forcing = step-by-step decoding
        m = kl_gate.score_prompt(lm, PROMPT, ids, lambda a, b: ref[a:b], chunk)
        s = kl_gate.summarize(m)
        assert s["positions"] == N
        # wider passes use other kernels, so not bit-equal: tiny KL, and flips only at near-ties
        assert s["kl_max"] < 1e-3 and s["sure_flip_rate"] == 0, (chunk, s)


def test_a_disturbed_model_scores_well_above_the_floor():
    lm = tiny()
    ids, ref = reference(lm)
    other = tiny()
    mx.random.seed(7)
    other.update(tree_map(lambda p: p * (1 + 0.2 * mx.random.normal(p.shape)) if p.ndim == 2 else p, other.parameters()))
    floor = kl_gate.summarize(kl_gate.score_prompt(lm, PROMPT, ids, lambda a, b: ref[a:b], 16))
    s = kl_gate.summarize(kl_gate.score_prompt(other, PROMPT, ids, lambda a, b: ref[a:b], 16))
    assert s["kl_mean"] > 50 * max(floor["kl_mean"], 1e-6) and s["sure_flip_rate"] > 0, (s, floor)
