"""The id-set draft head keeps token ids: its argmax is the full head's argmax over the set, at the same id.
Tiny random heads only (no model). Run: <venv>/bin/python -m pytest -q test_subhead_ids.py"""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "dflash"))
from subhead_ids import draft_ids, rows_head, subset_logits  # noqa: E402

VOCAB, DIM, K = 1000, 128, 300


def heads():
    mx.random.seed(0)
    lin = nn.Linear(DIM, VOCAB, bias=False)
    return {"float": lin, "quantized": nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)}


@pytest.mark.parametrize("kind", ["float", "quantized"])
def test_argmax_is_full_heads_argmax_over_the_set(kind):
    head = heads()[kind]
    prompt = [5, 412, 977, 412, 640]               # ids >= K the draft must still be able to propose
    idx = draft_ids(prompt, K)
    allowed = mx.zeros((VOCAB,), dtype=mx.bool_)
    allowed[idx] = True
    h = mx.random.normal((2, 7, DIM))
    full = head(h)
    sub = subset_logits(rows_head(head, idx), idx, h, VOCAB)
    assert sub.shape == full.shape
    expected = mx.argmax(mx.where(allowed, full, -mx.inf), axis=-1)
    assert mx.array_equal(mx.argmax(sub, axis=-1), expected)
    # the set's logits are the full head's at the same ids; everything outside is never drafted
    assert mx.allclose(sub[..., idx], full[..., idx], atol=1e-4)
    assert mx.all(mx.where(allowed, -1e30, sub) <= -1e29)


def test_prompt_ids_above_k_are_in_the_set():
    idx = draft_ids([3, 999, 500, 999], K).tolist()
    assert idx == sorted(set(range(K)) | {999, 500})
