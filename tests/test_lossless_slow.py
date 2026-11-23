"""Real small models, opt-in (MLX_FASTPATH_SLOW=1; run one model per process: -k q08 / -k q2b / -k gemma; ≤ 6 GB):
the speedups keep greedy output (bf16 KV cache) and are no less accurate than stock (4-bit KV cache).

bf16 KV cache — same prompts, greedy, speedups off then on. Both bounds are measured, not copied:
- Shape shift: how far stock's own logits move when the text is fed in blocks of the decoder's verify width instead of
  one token at a time (summation order; optiq's OptiqEngine.generate docstring puts it at about 0.25).
- Drift: fed in those blocks, the speedups move no logit further from stock than that shape shift. They also only
  reorder sums, so a larger move is a bug, split or no split. At width 1 stock has no shift, so the drift must be 0.
- Splits: the texts match, or where they split the stock logits of the two tokens differ by at most drift + shape
  shift: a gap that small resolves either way depending on the verify shapes a run happens to use.
- MTP depth 2 verifies 3 tokens: with 4 query heads per KV head that is 12 fold rows, under attention.FOLD_MIN_ROWS, so
  the GQA fold never runs there. generate_ngram may verify up to 9 tokens, but on these prompts and models optiq's
  gate keeps it at <= 3 (its 5- and 9-wide passes are one-off cost calibration). So the n-gram check measures drift at
  width 9 (36 rows), where the fold does run; a counter shows it ran.
- optiq keeps learned state across generate_ngram calls (pass costs, acceptance priors, draft costs, per model), so
  each run starts from a reset state: otherwise the second run inherits what the first learned.
4-bit KV cache — identical text is not a fair test: stock's own 4-bit attention is up to ~3.4 logits from exact fp32
attention on the same cache and changes the top token on up to ~6% of tokens (research/batch5/results.md). So the
check is accuracy against that exact reference, along the stock text, one token at a time (the fused-kernel path):
fastpath's mean |error| <= stock's x MEAN_RATIO, and its top-token agreement >= stock's - TOP1_SLACK.
At least MIN_AGREED tokens must agree before any split, so a split on the first token cannot pass for coverage.

Qwen3.5 0.8B / 2B (MTP head, same layer types as the 9B) run the MTP and n-gram paths; Gemma 4 E2B (no MTP, attention
not in attention.MODULES) runs the plain AR path, where only the model-agnostic parts apply.
Models: OptiQ 4/8-bit copies in MLX_FASTPATH_MODELS (vision sidecars not needed).
"""
import gc

import mlx.core as mx
import pytest

from real_model import LLM, caps, slow_enabled

MTP_MODELS = ["Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed",      # built here with `optiq convert` from Qwen/Qwen3.5-0.8B
              "Qwen3.5-2B-OptiQ-4bit"]                    # mlx-community/Qwen3.5-2B-OptiQ-4bit
AR_MODELS = ["gemma-4-e2b-it-OptiQ-4bit"]                 # mlx-community/gemma-4-e2b-it-OptiQ-4bit
PROMPTS = ["Explain in five sentences why the sky is blue.",
           "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
           "A train leaves at 9:40 and arrives at 13:15. How long is the trip? Show the steps."]
MAX_TOKENS = 96
MIN_AGREED = 100
MEAN_RATIO = 1.1              # 4-bit KV: fastpath's mean error vs exact may exceed stock's by at most 10%
TOP1_SLACK = 0.02             # ... and its top-token agreement with exact may trail stock's by at most 2 points
MTP_WIDTH, NGRAM_WIDTH = 3, 9  # verify widths: MTP depth 2, generate_ngram's 8 drafts

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not slow_enabled(), reason="opt-in: MLX_FASTPATH_SLOW=1")]


def _engine(name):
    import mlx_fastpath as fp
    path = LLM / name
    if not path.exists():
        pytest.skip(f"{name} not on disk")
    mlx_limit, max_footprint, min_free = caps(path)
    guard = fp.MemoryGuard(min_free_gb=min_free, max_footprint_gb=max_footprint).__enter__()
    return fp.load(path, memory_limit_gb=mlx_limit, n2=False), guard


def _release(engine, guard):
    report = guard.report()
    print(f"\n[memory] {report}", end="")
    try:
        assert report["swap_growth_gb"] <= 0.0625, "swap grew"
    finally:
        guard.__exit__(None, None, None)
        del engine
        gc.collect()
        mx.clear_cache()


@pytest.fixture(scope="module", params=MTP_MODELS, ids=["q08", "q2b"])
def mtp_engine(request):
    """One load per model, shared by read-only checks (generation never changes the weights)."""
    engine, guard = _engine(request.param)
    yield engine
    _release(engine, guard)


@pytest.fixture(scope="module", params=AR_MODELS, ids=["gemma"])
def ar_engine(request):
    engine, guard = _engine(request.param)
    yield engine
    _release(engine, guard)


def _prompt_ids(engine, text):
    tok = engine.tokenizer
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False))


def _exact_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    """fp32 attention on the dequantized 4-bit cache: the reference both 4-bit paths are measured against."""
    from mlx_lm.models.cache import QuantizedKVCache
    from mlx_lm.models.base import scaled_dot_product_attention as stock
    if not isinstance(cache, QuantizedKVCache):
        return stock(queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks)
    k = mx.dequantize(*keys, group_size=cache.group_size, bits=cache.bits).astype(mx.float32)
    v = mx.dequantize(*values, group_size=cache.group_size, bits=cache.bits).astype(mx.float32)
    qL, g, n = queries.shape[2], queries.shape[1] // k.shape[1], k.shape[2]
    k, v = mx.repeat(k, g, axis=1), mx.repeat(v, g, axis=1)
    s = (queries.astype(mx.float32) * scale) @ k.swapaxes(-1, -2)
    s = mx.where(mx.arange(n)[None] <= (n - qL + mx.arange(qL))[:, None], s, -mx.inf)
    return (mx.softmax(s, axis=-1) @ v).astype(queries.dtype)


def _logits_along(engine, prompt_ids, tokens, *, kv_bits=None, block=1, mode="stock"):
    """Next-token logits (fp32) for each of `tokens`, fed after the prompt in blocks of `block` (the decoder's verify
    shape), with the run's KV cache. mode: stock | fast (apply()) | exact (fp32 attention on the cache). Row k scores
    token k."""
    import mlx_fastpath as fp
    import mlx_lm.models.qwen3_next as qn
    from mlx_lm.models.cache import make_prompt_cache
    from optiq.runtime.engine import _maybe_quantize_kv
    fp.remove()                                          # first, so `saved` is stock even if an apply() leaked
    model, saved = engine._model, qn.scaled_dot_product_attention
    if mode == "fast":
        assert fp.apply(), "apply() ran stock: installed versions differ from apply.TESTED"
    elif mode == "exact":
        qn.scaled_dot_product_attention = _exact_sdpa
    try:
        cache = make_prompt_cache(model)
        rows = [model(mx.array([list(prompt_ids)]), cache=cache)[0, -1:]]
        if kv_bits is not None:
            _maybe_quantize_kv(cache, kv_bits, 64)        # as OptiqEngine.generate does after prefill
        feed = tokens[:-1]
        for i in range(0, len(feed), block):
            rows.append(model(mx.array([feed[i:i + block]]), cache=cache)[0])
            mx.eval(rows[-1])
        return mx.concatenate(rows).astype(mx.float32)
    finally:
        fp.remove()
        qn.scaled_dot_product_attention = saved


def _assert_same_or_near_tie(engine, prompt_ids, stock, fast, block):
    """stock / fast: generated token ids per prompt (EOS excluded), from decoded text re-encoded (_ids). A tokenizer
    round-trip difference could show as a split, so it can only fail a check, never pass one falsely."""
    eos = min(engine._eos_ids())                            # a set of ids; any one stands for "ended"
    agreed = 0
    for ids, s, f in zip(prompt_ids, stock, fast):
        a, b = list(s) + [eos], list(f) + [eos]               # a text ends on EOS: one more position to score
        ref = _logits_along(engine, ids, a, block=block, mode="stock")
        shape = mx.abs(ref - _logits_along(engine, ids, a, block=1, mode="stock")).max().item() if block > 1 else 0.0
        drift = mx.abs(ref - _logits_along(engine, ids, a, block=block, mode="fast")).max().item()
        print(f"\n[lossless] width {block}: drift {drift:.3f}, stock shape shift {shape:.3f}", end="")
        assert drift <= shape, f"speedups move logits by {drift:.3f}, more than stock's own shape shift {shape:.3f}"
        if a == b:
            agreed += len(a) - 1
            continue
        i = next(k for k, (x, y) in enumerate(zip(a, b)) if x != y)   # exists: they differ and both end on EOS
        gap = abs((ref[i, a[i]] - ref[i, b[i]]).item())
        print(f", split at token {i}, stock logit gap {gap:.3f}", end="")
        assert gap <= drift + shape, f"split at token {i} with logit gap {gap:.3f} > drift {drift:.3f} + shift {shape:.3f}"
        agreed += i
    assert agreed >= MIN_AGREED, f"only {agreed} tokens agreed before the splits"


@pytest.fixture
def attention_calls(monkeypatch):
    """Counts the calls into the patched attention paths (they still run; the count only shows they were taken)."""
    import mlx_fastpath.attention as att
    calls = {"fused_qkv_sdpa": 0, "folded_qsdpa": 0, "folded_sdpa": 0}
    for name in calls:
        real = getattr(att, name)

        def counted(*a, _real=real, _name=name, **k):
            calls[_name] += 1
            return _real(*a, **k)
        monkeypatch.setattr(att, name, counted)
    return calls


def _ids(engine, text):
    return engine.tokenizer.encode(text)


def test_speedups_keep_greedy_mtp_output_with_a_bf16_cache(mtp_engine):
    import mlx_fastpath as fp
    ids = [_prompt_ids(mtp_engine, p) for p in PROMPTS]
    fp.remove()
    stock = [_ids(mtp_engine, fp.generate(mtp_engine, x, max_tokens=MAX_TOKENS).text) for x in ids]
    try:
        assert fp.apply(), "apply() ran stock: installed versions differ from apply.TESTED"
        fast = [_ids(mtp_engine, fp.generate(mtp_engine, x, max_tokens=MAX_TOKENS).text) for x in ids]
    finally:
        fp.remove()
    _assert_same_or_near_tie(mtp_engine, ids, stock, fast, MTP_WIDTH)


def _ngram_from_clean_state(engine, prompt_ids):
    """generate_ngram after clearing what optiq's n-gram path learned in earlier calls."""
    import mlx_fastpath as fp
    from optiq.runtime import ngram_spec
    for state in (ngram_spec._COST_MODELS, ngram_spec._PASS_COSTS, ngram_spec._ACCEPTANCE_PRIOR,
                  ngram_spec._MTP_DRAFT_COSTS):
        state.clear()
    return fp.generate_ngram(engine, prompt_ids, max_tokens=MAX_TOKENS)


def test_speedups_keep_greedy_ngram_output_and_the_fold_stays_close(mtp_engine, attention_calls):
    """Reference: plain greedy decoding (optiq's AR loop), not stock n-gram. Stock optiq 0.8.5's n-gram + MTP path is
    not greedy on quantized Qwen3.5: its head pricing runs 5 target steps on the live, untrimmable hybrid cache
    (research/batch5/results.md); fastpath swaps that pricing out. Stock's own distance is printed, not asserted."""
    import mlx_fastpath as fp
    ids = [_prompt_ids(mtp_engine, p) for p in PROMPTS]
    fp.remove()
    greedy = [_ids(mtp_engine, mtp_engine.generate("", prompt_ids=x, max_tokens=MAX_TOKENS, depth=0).text) for x in ids]
    stock = [_ngram_from_clean_state(mtp_engine, x) for x in ids]
    try:
        assert fp.apply(), "apply() ran stock: installed versions differ from apply.TESTED"
        fast = [_ngram_from_clean_state(mtp_engine, x) for x in ids]
    finally:
        fp.remove()
    first_split = lambda a, b: next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    print(f"\n[ngram] first split vs greedy: stock {[first_split(g, s) for g, s in zip(greedy, stock)]}, "
          f"fastpath {[first_split(g, f) for g, f in zip(greedy, fast)]}", end="")
    _assert_same_or_near_tie(mtp_engine, ids, greedy, fast, NGRAM_WIDTH)
    assert attention_calls["folded_sdpa"] > 0, attention_calls      # precondition: the drift check ran the fold


def test_speedups_are_no_less_accurate_than_stock_with_a_4bit_cache(mtp_engine, attention_calls):
    import mlx_fastpath as fp
    ids = [_prompt_ids(mtp_engine, p) for p in PROMPTS]
    fp.remove()
    texts = [_ids(mtp_engine, fp.generate(mtp_engine, x, max_tokens=MAX_TOKENS, kv_bits=4).text) for x in ids]
    err, top1 = {"stock": [], "fast": []}, {"stock": [], "fast": []}
    for x, t in zip(ids, texts):
        exact = _logits_along(mtp_engine, x, t, kv_bits=4, mode="exact")
        for mode in ("stock", "fast"):
            got = _logits_along(mtp_engine, x, t, kv_bits=4, mode=mode)
            err[mode].append(mx.abs(got - exact).mean().item())
            top1[mode].append((got.argmax(-1) == exact.argmax(-1)).mean().item())
    assert attention_calls["fused_qkv_sdpa"] > 0, attention_calls   # precondition: the fused 4-bit kernel ran
    mean = {m: sum(v) / len(v) for m, v in err.items()}
    agree = {m: sum(v) / len(v) for m, v in top1.items()}
    print(f"\n[kv4] mean |err| vs exact: stock {mean['stock']:.4f}, fast {mean['fast']:.4f}; "
          f"top-1 agreement: stock {agree['stock']:.3f}, fast {agree['fast']:.3f}; per prompt {err} {top1}")
    assert sum(len(t) for t in texts) >= MIN_AGREED, "too few tokens measured"
    assert mean["fast"] <= mean["stock"] * MEAN_RATIO, mean
    assert agree["fast"] >= agree["stock"] - TOP1_SLACK, agree


def test_the_prefill_swap_keeps_plain_decoding_output_on_a_model_without_mtp(ar_engine):
    """Gemma: its attention is not in attention.MODULES and it has no MTP head, so of apply()'s swaps only the
    model-agnostic prefill (prefill.prefill_chunked) runs. This checks that one: same greedy text with it on.
    (Its teacher-forced drift is 0 by construction, since it feeds the model directly; the text comparison is the
    check.)"""
    import mlx_fastpath as fp
    from mlx_fastpath import prefill
    from optiq.runtime.engine import OptiqEngine
    ids = [_prompt_ids(ar_engine, p) for p in PROMPTS]
    fp.remove()
    stock = [_ids(ar_engine, ar_engine.generate("", prompt_ids=x, max_tokens=MAX_TOKENS, depth=0).text) for x in ids]
    try:
        assert fp.apply(), "apply() ran stock: installed versions differ from apply.TESTED"
        assert OptiqEngine._prefill_chunked is prefill.prefill_chunked, "precondition: the prefill swap is active"
        fast = [_ids(ar_engine, ar_engine.generate("", prompt_ids=x, max_tokens=MAX_TOKENS, depth=0).text) for x in ids]
    finally:
        fp.remove()
    _assert_same_or_near_tie(ar_engine, ids, stock, fast, 1)
