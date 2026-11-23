"""The reduced-vocab draft head scores the first K token ids exactly as the full head does, and never drafts others."""
import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_fastpath.draft import expand, sub_head, sub_logits

VOCAB, HID, K = 1000, 256, 300


def _heads():
    lin = nn.Linear(HID, VOCAB, bias=False)
    return {"linear": lin, "quantized": nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)}


@pytest.mark.parametrize("kind", ["linear", "quantized"])
def test_sub_head_logits_equal_full_head_on_kept_ids(rng, kind):
    head = _heads()[kind]
    x = rng.normal((1, 1, HID))
    full = head(x)
    part = sub_logits(sub_head(head, K), x)
    assert part.shape[-1] == K
    assert mx.allclose(part, full[..., :K], atol=1e-4).item()


def test_expanded_logits_never_pick_an_id_outside_the_subset(rng):
    head = _heads()["quantized"]
    x = rng.normal((8, 1, HID))
    out = expand(sub_logits(sub_head(head, K), x), VOCAB)
    assert out.shape[-1] == VOCAB
    assert (out.argmax(-1) < K).all().item()
    assert (out[..., K:] <= -1e29).all().item()


class _StandInLM:
    """Stand-in for the language model (allowed: tests stand in for the LLM). Real heads, fixed MTP core."""
    def __init__(self, head):
        self.lm_head, self.args = head, type("A", (), {"tie_word_embeddings": False})()
        self.calls = 0

    def _mtp_core(self, h, ids, mtp_cache=None, emit_logits=True, position_offset=None):
        self.calls += 1
        return None, h * (1.0 + 0.1 * self.calls)

    def make_mtp_cache(self):
        return [None]


def test_fast_mtp_drafter_drafts_the_best_id_within_the_subset(rng, monkeypatch):
    """Head built so the best id over the FULL vocab is outside the subset: the drafter must still stay inside."""
    from optiq.runtime import ngram_spec
    from mlx_fastpath.draft import _fast_mtp_drafter_class
    monkeypatch.setattr(ngram_spec, "_mtp_supported", lambda m: True)
    lin = nn.Linear(HID, VOCAB, bias=False)
    w = rng.normal((VOCAB, HID)) * 0.01
    w[K + 7] = mx.ones((HID,))          # an id outside the subset that wins on any positive hidden state
    lin.weight = w
    head = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
    hidden = mx.abs(rng.normal((1, 1, HID)))
    assert head(hidden)[0, -1].argmax().item() == K + 7      # precondition: full head picks outside the subset
    drafts = _fast_mtp_drafter_class(K)(_StandInLM(head), depth=2).draft(hidden, token=5)
    expect = [int(head(hidden * 1.1)[0, -1, :K].argmax().item()), int(head(hidden * 1.1 * 1.2)[0, -1, :K].argmax().item())]
    assert drafts == expect and all(d < K for d in drafts)


# ---------- id-set draft head: the first K ids plus the prompt's own ids (default off) ----------
def _head_preferring(rng, winner):
    """A quantized head where `winner` beats every other id on any positive hidden state."""
    lin = nn.Linear(HID, VOCAB, bias=False)
    w = rng.normal((VOCAB, HID)) * 0.01
    w[winner] = mx.ones((HID,))
    lin.weight = w
    return nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)


class _StandInEngine:
    """Stand-in for optiq's engine (allowed: the LLM). Its generate() makes one MTP draft through the real,
    patched OptiqEngine._mtp_draft and returns the drafted id."""
    def __init__(self, head, hidden):
        self._lm, self.hidden = _StandInLM(head), hidden

    def generate(self, prompt, prompt_ids, **kw):
        from optiq.runtime.engine import OptiqEngine
        logits, _ = OptiqEngine._mtp_draft(self, self.hidden, 5, None)
        return int(logits[0, -1].argmax().item())


@pytest.fixture
def draft_patch():
    from mlx_fastpath import draft
    yield draft
    draft.disable_reduced_vocab()


def test_id_set_is_the_first_k_ids_plus_the_prompt_ids():
    from mlx_fastpath.draft import id_set
    assert id_set([3, K + 7, K + 7, VOCAB - 1], K) == list(range(K)) + [K + 7, VOCAB - 1]


def test_id_set_head_scores_kept_ids_exactly_and_never_drafts_others(rng):
    from mlx_fastpath.draft import id_set, rows_head, scatter
    head = _heads()["quantized"]
    x = rng.normal((4, 1, HID))
    idx = mx.array(id_set([K + 7, VOCAB - 1], K), dtype=mx.int32)
    out = scatter(sub_logits(rows_head(head, idx), x), idx, VOCAB)
    full = head(x)
    keep = mx.zeros((VOCAB,), dtype=mx.bool_)
    keep[idx] = True
    assert mx.allclose(mx.where(keep, out, 0), mx.where(keep, full, 0), atol=1e-4).item()
    assert (mx.where(keep, -1e30, out) <= -1e29).all().item()


@pytest.mark.parametrize("prompt_ids, can_draft_it", [(False, False), (True, True)])
def test_generate_drafts_a_prompt_id_beyond_the_first_k_only_with_the_switch(rng, draft_patch, prompt_ids, can_draft_it):
    import mlx_fastpath as fp
    head = _head_preferring(rng, K + 7)
    hidden = mx.abs(rng.normal((1, 1, HID)))
    draft_patch.enable_reduced_vocab(K, prompt_ids=prompt_ids)
    eng = _StandInEngine(head, hidden)
    drafted = fp.generate(eng, [1, 2, K + 7, 4])
    assert (drafted == K + 7) is can_draft_it
    assert drafted == K + 7 or drafted < K


def test_the_prompt_ids_do_not_outlive_their_generate_call(rng, draft_patch):
    import mlx_fastpath as fp
    from optiq.runtime.engine import OptiqEngine
    head = _head_preferring(rng, K + 7)
    hidden = mx.abs(rng.normal((1, 1, HID)))
    draft_patch.enable_reduced_vocab(K, prompt_ids=True)
    eng = _StandInEngine(head, hidden)
    assert fp.generate(eng, [K + 7]) == K + 7
    logits, _ = OptiqEngine._mtp_draft(eng, hidden, 5, None)     # outside generate(): the first K ids only
    assert int(logits[0, -1].argmax().item()) < K


def test_each_generate_call_drafts_from_its_own_prompt(rng, draft_patch):
    import mlx_fastpath as fp
    head = _head_preferring(rng, K + 7)
    hidden = mx.abs(rng.normal((1, 1, HID)))
    draft_patch.enable_reduced_vocab(K, prompt_ids=True)
    eng = _StandInEngine(head, hidden)
    assert fp.generate(eng, [1, K + 7]) == K + 7
    assert fp.generate(eng, [1, 2]) < K                          # a new prompt without K+7: no stale id set


class _FailingEngine(_StandInEngine):
    def generate(self, prompt, prompt_ids, **kw):
        raise RuntimeError("engine failed")


def test_the_prompt_ids_are_cleared_when_generate_raises(rng, draft_patch):
    import mlx_fastpath as fp
    from optiq.runtime.engine import OptiqEngine
    head = _head_preferring(rng, K + 7)
    hidden = mx.abs(rng.normal((1, 1, HID)))
    draft_patch.enable_reduced_vocab(K, prompt_ids=True)
    with pytest.raises(RuntimeError, match="engine failed"):
        fp.generate(_FailingEngine(head, hidden), [K + 7])
    logits, _ = OptiqEngine._mtp_draft(_StandInEngine(head, hidden), hidden, 5, None)
    assert int(logits[0, -1].argmax().item()) < K


@pytest.mark.parametrize("switch, can_draft_it", [(False, False), (True, True)])
def test_apply_passes_the_prompt_ids_switch_to_the_draft_head(rng, switch, can_draft_it):
    import mlx_fastpath as fp
    head = _head_preferring(rng, K + 7)
    hidden = mx.abs(rng.normal((1, 1, HID)))
    try:
        done = fp.apply(attention_dispatch=False, draft_vocab=K, draft_prompt_ids=switch, fused_capture=False,
                        prefill_step=None)
        assert done["draft_vocab"] == K                          # precondition: apply did not fall back to stock
        assert (fp.generate(_StandInEngine(head, hidden), [1, K + 7]) == K + 7) is can_draft_it
    finally:
        fp.remove()
