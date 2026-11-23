"""Pricing the MTP head before an n-gram run (optiq's ngram_spec.measure_draft_cost, swapped by the fast drafter switch)
leaves the request's own cache as it was, and drafts from a hidden state of the model's real width.
Tiny random-weight Qwen3.5 with an 8-bit embedding: the same hybrid cache (GDN ArraysCache + KVCache, not trimmable)
and the same packed embedding width as the real quantized models. A few MB."""
import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import make_prompt_cache

PROMPT = [5, 17, 42, 9, 101, 33, 7]


def tiny_qwen35():
    mx.random.seed(0)
    args = qwen3_5.ModelArgs.from_dict({"model_type": "qwen3_5", "text_config": dict(
        model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=1, head_dim=16, vocab_size=256, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16, full_attention_interval=4)})
    model = qwen3_5.Model(args)
    nn.quantize(model, group_size=32, bits=8)
    mx.eval(model.parameters())
    return model


class StandInHead:
    """Stands in for the MTP head (part of the LLM): records the hidden widths it is asked to draft from. Like optiq's
    own MtpDrafter, a draft that fails returns no tokens instead of raising, so pricing carries on to the target
    steps (the path that touched the live cache)."""
    def __init__(self, model):
        self.model, self.ok, self.depth, self.widths = model.language_model, True, 1, []

    def draft(self, hidden, token, position=None):
        self.widths.append(hidden.shape[-1])
        return [0] if hidden.shape[-1] == self.model.args.hidden_size else []


@pytest.fixture
def fast_drafter_switch():
    from mlx_fastpath import draft
    draft.enable_fast_mtp_drafter()
    yield
    draft.disable_fast_mtp_drafter()


def _next_token_logits(model, cache):
    return model(mx.array([[PROMPT[-1]]]), cache=cache)[0, -1]


def test_pricing_the_head_leaves_the_live_cache_as_it_was(fast_drafter_switch):
    from optiq.runtime import ngram_spec
    model = tiny_qwen35()
    untouched, priced = make_prompt_cache(model), make_prompt_cache(model)
    for c in (untouched, priced):
        model(mx.array([PROMPT[:-1]]), cache=c)
    head = StandInHead(model)
    ngram_spec.measure_draft_cost(head, model, cache=priced)
    assert head.widths, "precondition: the head was priced (drafted at least once)"
    assert [c.offset for c in priced if hasattr(c, "offset")] == [c.offset for c in untouched if hasattr(c, "offset")]
    assert mx.array_equal(_next_token_logits(model, priced), _next_token_logits(model, untouched)).item()


def test_pricing_the_head_drafts_from_the_models_hidden_width(fast_drafter_switch):
    from optiq.runtime import ngram_spec
    model = tiny_qwen35()
    cache = make_prompt_cache(model)
    model(mx.array([PROMPT[:-1]]), cache=cache)
    head = StandInHead(model)
    cost = ngram_spec.measure_draft_cost(head, model, cache=cache)
    assert head.widths and set(head.widths) == {model.language_model.args.hidden_size}
    assert cost > 0.0


def test_switching_the_drafter_off_gives_optiq_its_own_pricing_back():
    from optiq.runtime import ngram_spec
    from mlx_fastpath import draft
    original = ngram_spec.measure_draft_cost
    draft.enable_fast_mtp_drafter()
    draft.disable_fast_mtp_drafter()
    assert ngram_spec.measure_draft_cost is original
