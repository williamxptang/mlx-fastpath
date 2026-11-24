"""Gemma 4 assistant drafting with an adaptive gamma: the output is plain greedy decoding whatever gamma each pass
picks, and the gamma choice follows step costs and acceptance.

Tiny random-weight Gemma 4 (sliding window 8, so the ring-buffer caches wrap and get rolled back) and a stand-in
drafter that proposes the true next token except on a fixed pattern of positions: the same cache types, KV sharing
and rollback as the real E2B, a few MB."""
import mlx.core as mx
import pytest
from mlx_lm.generate import generate_step
from mlx_lm.models import gemma4_text
from mlx_lm.models.cache import make_prompt_cache

from mlx_fastpath import assistant
from mlx_fastpath.assistant import GammaChooser

PROMPT = [5, 17, 42, 9, 101, 33, 7, 64, 2, 88, 19]
GEN = 40                                       # > 4 sliding windows


def tiny_gemma4():
    mx.random.seed(0)
    args = gemma4_text.ModelArgs(
        hidden_size=64, num_hidden_layers=6, intermediate_size=128, num_attention_heads=2, head_dim=16,
        global_head_dim=32, num_key_value_heads=1, num_kv_shared_layers=2, vocab_size=256,
        vocab_size_per_layer_input=256, hidden_size_per_layer_input=16, sliding_window=8, sliding_window_pattern=3,
        max_position_embeddings=512, use_double_wide_mlp=False)
    model = gemma4_text.Model(args)
    mx.eval(model.parameters())
    return model


def plain_greedy(model, n=GEN):
    out = []
    for tok, _ in generate_step(mx.array(PROMPT), model, max_tokens=n, prompt_cache=make_prompt_cache(model)):
        out.append(int(tok))
    return out


class StandInDrafter:
    """Proposes the true next token (from plain greedy decoding) unless `wrong(position)`; then a different one."""

    def __init__(self, sequence, wrong):
        self.seq, self.wrong = sequence, wrong

    def forward(self, last_token_emb, target_hidden, shared_kv, position):
        # cache holds seq[:position]; `cur` = seq[position] is fed next, so the draft for `position` is seq[position+1]
        true = self.seq[position + 1] if position + 1 < len(self.seq) else 0
        tok = (true + 1) % 256 if self.wrong(position) else true
        return mx.eye(256)[tok][None, None], target_hidden


@pytest.fixture(scope="module")
def setup():
    model = tiny_gemma4()
    ref = plain_greedy(model)
    assert len(ref) == GEN
    return model, ref


def generate(model, drafter, **kw):
    from optiq.runtime.spec import SpecStats
    stats, out = SpecStats(), []
    for tok, _ in assistant.assistant_generate_step(mx.array(PROMPT), model, drafter=drafter, max_tokens=GEN,
                                                    prompt_cache=make_prompt_cache(model), stats=stats, **kw):
        out.append(int(tok))
    return out, stats


FLAT = {"verify": {1: 30.0, 2: 30.0, 3: 30.0, 4: 30.0}, "draft": {1: 1.0, 2: 2.0, 3: 3.0}}   # gamma 3 is best
CLIFF = {"verify": {1: 30.8, 2: 32.0, 3: 52.1, 4: 65.1}, "draft": {1: 2.3, 2: 3.6, 3: 4.9}}  # E2B int4 on M1


@pytest.mark.parametrize("wrong", [lambda p: False, lambda p: True, lambda p: p % 3 == 0, lambda p: p % 5 in (1, 2)],
                         ids=["always-right", "always-wrong", "every-third-wrong", "two-in-five-wrong"])
@pytest.mark.parametrize("costs", [FLAT, CLIFF, None], ids=["flat-costs", "cliff-costs", "measured-costs"])
def test_adaptive_output_is_plain_greedy(setup, wrong, costs):
    model, ref = setup
    seq = PROMPT + ref
    chooser = GammaChooser(costs) if costs else None
    out, stats = generate(model, StandInDrafter(seq, wrong), gamma="auto", chooser=chooser)
    assert out == ref
    assert stats.n_emitted == GEN


def test_drafts_are_used_when_they_are_right(setup):
    model, ref = setup
    out, stats = generate(model, StandInDrafter(PROMPT + ref, lambda p: False), gamma="auto", chooser=GammaChooser(FLAT))
    assert out == ref
    assert stats.n_accepted == stats.n_drafted > 0
    assert stats.n_target_calls < GEN / 2           # several tokens per pass


@pytest.mark.parametrize("gamma", [1, 2, 3])
def test_fixed_gamma_matches_optiqs_loop(setup, gamma):
    from optiq.runtime.spec import SpecStats, assistant_generate_step as optiq_step
    model, ref = setup
    drafter = StandInDrafter(PROMPT + ref, lambda p: p % 3 == 0)
    ours, s1 = generate(model, drafter, gamma=gamma)
    s2, theirs = SpecStats(), []
    for tok, _ in optiq_step(mx.array(PROMPT), model, drafter=drafter, gamma=gamma, max_tokens=GEN,
                             prompt_cache=make_prompt_cache(model), stats=s2):
        theirs.append(int(tok))
    assert ours == theirs == ref
    assert (s1.n_drafted, s1.n_accepted, s1.n_target_calls) == (s2.n_drafted, s2.n_accepted, s2.n_target_calls)


@pytest.mark.parametrize("costs,p,expected", [
    (CLIFF, 0.9, 1),                                                                 # 3-token verify too dear
    ({"verify": {1: 40.0, 2: 44.1, 3: 48.2, 4: 67.0}, "draft": {1: 1.9, 2: 3.7, 3: 5.4}}, 0.7, 2),  # E2B OptiQ
    (FLAT, 0.9, 3),                                                                  # verify width is free
    (CLIFF, 0.1, 0),                                                                 # drafts rarely right
], ids=["int4-cliff", "optiq-cliff-at-4", "flat", "low-acceptance"])
def test_gamma_maximises_expected_tokens_per_ms(costs, p, expected):
    prior = (10 * p, 10 * (1 - p))
    assert GammaChooser(costs, prior=prior).best(limit=3) == expected


def test_gamma_never_exceeds_the_tokens_left():
    assert GammaChooser(FLAT, prior=(9, 1)).best(limit=1) == 1
    assert GammaChooser(FLAT, prior=(9, 1)).best(limit=0) == 0


def test_rejections_drive_gamma_down_and_probes_keep_measuring():
    c = GammaChooser(CLIFF)
    assert c.best(3) == 1
    for _ in range(60):
        c.update(1, 0)                                 # every draft rejected
    assert c.best(3) == 0
    picks = [c.choose(3) for _ in range(assistant.PROBE_EVERY)]
    assert picks.count(1) == 1 and picks.count(0) == assistant.PROBE_EVERY - 1


NEEDLE_OPTIQ = {"verify": {1: 37.06, 2: 43.06, 3: 57.02, 4: 70.37}, "draft": {1: 3.15, 2: 5.87, 3: 8.38}}


def test_probes_find_a_longer_gamma_that_pays():
    """Measured case (E2B OptiQ, 4k context, M1): the prior alone picks gamma 1, but drafts are mostly right, so a
    longer gamma pays. The probes must measure the deeper positions and move gamma up."""
    c = GammaChooser(NEEDLE_OPTIQ)
    assert c.best(3) == 1
    for _ in range(200):
        g = c.choose(3)
        c.update(g, g)                                 # every draft accepted
    assert c.best(3) >= 2


def test_acceptance_drives_gamma_up():
    c = GammaChooser(FLAT, prior=(1, 9))
    assert c.best(3) == 1
    for _ in range(60):
        c.update(3, 3)                                 # every draft accepted
    assert c.best(3) == 3


def _cache_states(caches):
    from mlx.utils import tree_flatten
    return [[(k, v.tolist() if isinstance(v, mx.array) else v) for k, v in tree_flatten(c.state)] for c in caches]


@pytest.mark.parametrize("prompt_len,steady_steps", [(4, 0), (len(PROMPT) - 1, 0), (len(PROMPT) - 1, 20)],
                         ids=["partly-filled-ring", "fresh-prompt", "ring-at-steady-state"])
def test_measuring_costs_leaves_the_cache_exactly_as_it_was(setup, prompt_len, steady_steps):
    """The cost measurement writes verify steps into the live cache. A full sliding-window ring buffer is written in
    place, so it must be put back from copies, not rolled back."""
    from optiq.runtime.spec.runtime import _target_step
    model, ref = setup
    caches = make_prompt_cache(model)
    model(mx.array(PROMPT[:prompt_len])[None], cache=caches)
    tok = PROMPT[prompt_len]
    for _ in range(steady_steps):                      # single-token decoding: the ring stays at its window size
        _, logits = _target_step(model, mx.array([[tok]]), caches)
        tok = int(mx.argmax(logits[0, -1]))
    hidden, logits = _target_step(model, mx.array([[tok]]), caches)
    mx.eval(hidden, logits)
    cur, h_prev = int(mx.argmax(logits[0, -1])), hidden[:, -1:, :]
    before = _cache_states(caches)
    assert before
    assistant._measure_costs(model, StandInDrafter(PROMPT + ref, lambda p: False), caches, cur, h_prev, 3)
    assert _cache_states(caches) == before


@pytest.mark.parametrize("n", [1, 2, 3])
def test_very_short_generations_are_plain_greedy(setup, n):
    model, ref = setup
    out = [int(t) for t, _ in assistant.assistant_generate_step(
        mx.array(PROMPT), model, drafter=StandInDrafter(PROMPT + ref, lambda p: p % 3 == 0), gamma="auto",
        max_tokens=n, prompt_cache=make_prompt_cache(model), chooser=GammaChooser(FLAT))]
    assert out == ref[:n]


@pytest.mark.parametrize("stop_after", [1, 5, 17])
def test_stopping_early_leaves_the_cache_like_optiqs_loop(setup, stop_after):
    """The server's contract: a consumer that stops early leaves the cache holding exactly what it received."""
    from optiq.runtime.spec import assistant_generate_step as optiq_step
    model, ref = setup
    drafter = StandInDrafter(PROMPT + ref, lambda p: p % 3 == 0)

    def run(step, **kw):
        cache = make_prompt_cache(model)
        gen = step(mx.array(PROMPT), model, drafter=drafter, max_tokens=GEN, prompt_cache=cache, **kw)
        got = [int(t) for t, _ in (next(gen) for _ in range(stop_after))]
        gen.close()
        return got, _cache_states(cache)

    ours, theirs = run(assistant.assistant_generate_step, gamma=2), run(optiq_step, gamma=2)
    assert ours[0] == theirs[0] == ref[:stop_after]
    assert ours[1] == theirs[1]


def test_a_prefilled_server_cache_continues_the_conversation(setup):
    """mlx-lm's server passes only the uncached rest of the prompt plus the cache holding the part before it."""
    model, ref = setup
    cache = make_prompt_cache(model)
    model(mx.array(PROMPT[:6])[None], cache=cache)
    out = [int(t) for t, _ in assistant.assistant_generate_step(
        mx.array(PROMPT[6:]), model, drafter=StandInDrafter(PROMPT + ref, lambda p: p % 4 == 1), gamma="auto",
        max_tokens=GEN, prompt_cache=cache, chooser=GammaChooser(FLAT))]
    assert out == ref


def test_without_a_drafter_it_decodes_plainly(setup):
    model, ref = setup
    out = [int(t) for t, _ in assistant.assistant_generate_step(mx.array(PROMPT), model, drafter=None, max_tokens=GEN,
                                                                prompt_cache=make_prompt_cache(model))]
    assert out == ref


def test_the_swapped_in_loop_ignores_a_fixed_gamma_and_adapts(setup):
    """`optiq serve --drafter` calls optiq.runtime.spec.assistant_generate_step(..., gamma=2); after enable() that
    call must run the adaptive loop, and disable() must give optiq's back."""
    import optiq.runtime.spec as spec
    from optiq.runtime.spec import SpecStats
    model, ref = setup
    stock = spec.assistant_generate_step
    assistant.enable()
    try:
        stats = SpecStats()
        out = [int(t) for t, _ in spec.assistant_generate_step(
            mx.array(PROMPT), model, drafter=StandInDrafter(PROMPT + ref, lambda p: False), gamma=2, max_tokens=GEN,
            prompt_cache=make_prompt_cache(model), stats=stats)]
        assert out == ref
        assert stats.chooser is not None and sum(stats.chooser.picks.values()) > 0
    finally:
        assistant.disable()
    assert spec.assistant_generate_step is stock


def test_costs_belong_to_the_model_and_follow_the_context(setup, monkeypatch):
    """Costs are measured once per model + drafter + 4k context bucket, kept on the model: a second model gets its own
    measurement, and a cache that crosses a bucket boundary is measured again."""
    model, ref = setup
    calls = []
    monkeypatch.setattr(assistant, "_measure_costs", lambda *a, **k: calls.append(a[2][0].offset) or FLAT)
    drafter = StandInDrafter(PROMPT + ref, lambda p: False)
    other = tiny_gemma4()
    for m in (model, other, model):
        caches = make_prompt_cache(m)
        assistant._costs_for(m, drafter, caches, 0, None)
    assert len(calls) == 2                              # model, other; the third call reuses model's costs
    monkeypatch.setattr(assistant, "CTX_BUCKET", 4)
    caches = make_prompt_cache(model)
    model(mx.array(PROMPT[:5])[None], cache=caches)     # offset 5 -> bucket 1 at bucket size 4
    assistant._costs_for(model, drafter, caches, 0, None)
    assert len(calls) == 3


def test_a_measurement_that_fails_midway_leaves_the_cache_as_it_was(setup, monkeypatch):
    """A verify step that raises after writing into the cache (an OOM, an interrupt) must not leave it advanced."""
    import optiq.runtime.spec.runtime as rt
    model, ref = setup
    caches = make_prompt_cache(model)
    model(mx.array(PROMPT[:-1])[None], cache=caches)
    hidden = model.model(mx.array([[PROMPT[-1]]]), cache=caches)
    mx.eval(hidden)
    before = _cache_states(caches)
    real, calls = rt._target_step, []

    def failing_step(*a, **k):
        out = real(*a, **k)
        mx.eval(out)
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("stand-in for an out-of-memory error mid-measurement")
        return out

    monkeypatch.setattr(rt, "_target_step", failing_step)
    with pytest.raises(RuntimeError, match="mid-measurement"):
        assistant._measure_costs(model, StandInDrafter(PROMPT + ref, lambda p: False), caches, 0,
                                 hidden[:, -1:, :], 3)
    assert len(calls) == 3
    assert _cache_states(caches) == before


def test_costs_measured_for_another_object_at_a_reused_address_are_not_used(setup, monkeypatch):
    """Costs are kept per drafter OBJECT: an entry left by a freed drafter whose id() a new drafter now has must be
    measured again, not reused."""
    model, ref = setup
    calls = []
    monkeypatch.setattr(assistant, "_measure_costs", lambda *a, **k: calls.append(1) or FLAT)
    drafter = StandInDrafter(PROMPT + ref, lambda p: False)
    store = model.__dict__.setdefault("_fastpath_assistant_costs", {})
    store[id(drafter)] = (object(), {0: CLIFF})        # what a freed drafter at the same address would have left
    try:
        assert assistant._costs_for(model, drafter, make_prompt_cache(model), 0, None) is FLAT
        assert calls == [1]
    finally:
        store.pop(id(drafter), None)


def test_crossing_into_a_new_context_bucket_during_a_run_switches_the_costs(setup, monkeypatch):
    """A generation that grows past a bucket boundary must decide with the new bucket's costs from then on."""
    model, ref = setup
    measured = []

    def fake_measure(*a, **k):
        costs = {"verify": dict(FLAT["verify"]), "draft": dict(FLAT["draft"])}
        measured.append(costs)
        return costs

    monkeypatch.setattr(assistant, "_measure_costs", fake_measure)
    monkeypatch.setattr(assistant, "CTX_BUCKET", 16)    # prompt 11 tokens + 40 generated: crosses 16, 32, 48
    from optiq.runtime.spec import SpecStats
    stats = SpecStats()
    out = [int(t) for t, _ in assistant.assistant_generate_step(
        mx.array(PROMPT), model, drafter=StandInDrafter(PROMPT + ref, lambda p: p % 3 == 0), gamma="auto",
        max_tokens=GEN, prompt_cache=make_prompt_cache(model), stats=stats)]
    assert out == ref
    assert len(measured) >= 3
    assert stats.chooser.costs is measured[-1]


@pytest.mark.parametrize("stop_after", [1, 4, 9, 23])
def test_stopping_early_with_adaptive_gamma_leaves_the_cache_like_plain_decoding(setup, stop_after):
    """With gamma varying per pass, a consumer that stops early must leave the cache at the same position as plain
    decoding stopped at the same token: same offset, same number of kept tokens (values differ at rounding level and
    the ring buffer's internal layout differs, since verify writes several tokens at once)."""
    model, ref = setup

    def offsets(step, **kw):
        cache = make_prompt_cache(model)
        gen = step(mx.array(PROMPT), model, max_tokens=GEN, prompt_cache=cache, **kw)
        got = [int(t) for t, _ in (next(gen) for _ in range(stop_after))]
        gen.close()
        return got, [(c.offset, c.size()) for c in cache]   # the ring's write index (_idx) is layout, not content

    ours = offsets(assistant.assistant_generate_step, drafter=StandInDrafter(PROMPT + ref, lambda p: p % 3 == 0),
                   gamma="auto", chooser=GammaChooser(FLAT))
    plain = offsets(generate_step)
    assert ours[0] == plain[0] == ref[:stop_after]
    assert ours[1] == plain[1]


def test_putting_the_cache_back_is_not_counted_as_verify_cost(setup, monkeypatch):
    """A real pass never restores the cache, so the measured verify cost must exclude the restore: timing it would add
    the same amount to every width and push gamma up."""
    import time as _time
    model, ref = setup
    real_restore = assistant._restore

    def slow_restore(caches, snap):
        _time.sleep(0.2)
        real_restore(caches, snap)

    monkeypatch.setattr(assistant, "_restore", slow_restore)
    caches = make_prompt_cache(model)
    model(mx.array(PROMPT[:-1])[None], cache=caches)
    hidden = model.model(mx.array([[PROMPT[-1]]]), cache=caches)
    mx.eval(hidden)
    costs = assistant._measure_costs(model, StandInDrafter(PROMPT + ref, lambda p: False), caches, 0,
                                     hidden[:, -1:, :], 3, reps=1)
    assert len(costs["verify"]) == 4
    assert max(costs["verify"].values()) < 100          # ms; a timed restore alone would be >= 200


def test_real_pass_times_override_a_calibration_that_was_too_optimistic():
    """Measured on E2B OptiQ (M1): the start-up calibration timed a 4-token verify at 57 ms, real passes at gamma 3
    took about 67 ms. Once real passes are timed, gamma must follow them."""
    optimistic = {"verify": {1: 39.5, 2: 41.5, 3: 51.7, 4: 56.9}, "draft": {1: 2.0, 2: 3.7, 3: 5.4}}
    c = GammaChooser(optimistic, prior=(8, 2))           # p = 0.8 per position
    assert c.best(3) == 3
    for g, ms in ((1, 46.0), (2, 55.0), (3, 75.0)):
        for _ in range(assistant.OBS_MIN):
            c.observe(g, ms)
    assert c.best(3) == 2


def test_untimed_gammas_are_scaled_like_the_timed_ones():
    c = GammaChooser(FLAT)
    for _ in range(assistant.OBS_MIN):
        c.observe(1, 2 * c.calibrated(1))                 # real passes take twice the calibrated time
    assert c.cost(1) == pytest.approx(2 * c.calibrated(1))
    assert c.cost(3) == pytest.approx(2 * c.calibrated(3))


def test_a_new_context_bucket_drops_the_old_timings():
    c = GammaChooser(FLAT)
    for _ in range(assistant.OBS_MIN):
        c.observe(1, 99.0)
    assert c.cost(1) == pytest.approx(99.0)
    c.set_costs(CLIFF)
    assert c.cost(1) == pytest.approx(c.calibrated(1))


def test_the_loop_times_its_real_passes(setup):
    model, ref = setup
    from optiq.runtime.spec import SpecStats
    stats = SpecStats()
    out = [int(t) for t, _ in assistant.assistant_generate_step(
        mx.array(PROMPT), model, drafter=StandInDrafter(PROMPT + ref, lambda p: p % 3 == 0), gamma="auto",
        max_tokens=GEN, prompt_cache=make_prompt_cache(model), stats=stats, chooser=GammaChooser(FLAT))]
    assert out == ref
    obs = stats.chooser.observed
    timed = sum(v[1] for v in obs.values())
    assert obs and stats.n_target_calls - 2 <= timed <= stats.n_target_calls - 1   # all but the first (no hidden
    # state yet) and possibly the last (it can end the run before its tokens are read back)
    assert all(v[0] > 0 for v in obs.values())



def test_a_fixed_gamma_replaces_the_one_optiq_serve_passes(setup):
    """`optiq serve --drafter` always passes gamma=2; enable(1) must make it draft one token per pass, exactly as our
    loop at gamma 1 does."""
    import optiq.runtime.spec as spec
    from optiq.runtime.spec import SpecStats
    model, ref = setup
    drafter = StandInDrafter(PROMPT + ref, lambda p: p % 3 == 0)
    direct = generate(model, drafter, gamma=1)
    stock = spec.assistant_generate_step
    assistant.enable(1)
    try:
        stats = SpecStats()
        out = [int(t) for t, _ in spec.assistant_generate_step(
            mx.array(PROMPT), model, drafter=drafter, gamma=2, max_tokens=GEN, prompt_cache=make_prompt_cache(model),
            stats=stats)]
    finally:
        assistant.disable()
    assert spec.assistant_generate_step is stock
    assert out == direct[0] == ref
    assert (stats.n_drafted, stats.n_target_calls) == (direct[1].n_drafted, direct[1].n_target_calls)
    assert stats.n_drafted <= stats.n_target_calls


@pytest.mark.parametrize("bad", ["fast", -1, 1.5, True, False])
def test_a_bad_gamma_setting_is_refused_before_anything_is_swapped_in(bad):
    import mlx_lm.models.qwen3_next as qn
    import optiq.runtime.spec as spec
    import mlx_fastpath
    before = (spec.assistant_generate_step, qn.scaled_dot_product_attention)
    with pytest.raises(ValueError):
        mlx_fastpath.apply(assistant_gamma=bad)
    assert (spec.assistant_generate_step, qn.scaled_dot_product_attention) == before
    mlx_fastpath.remove()


def test_pass_timings_leave_out_the_consumers_time(setup):
    """A slow consumer (a server detokenizing, a network write) must not make the passes look expensive."""
    import time as _time
    from optiq.runtime.spec import SpecStats
    model, ref = setup
    stats = SpecStats()
    out = []
    for t, _ in assistant.assistant_generate_step(
            mx.array(PROMPT), model, drafter=StandInDrafter(PROMPT + ref, lambda p: p % 3 == 0), gamma="auto",
            max_tokens=GEN, prompt_cache=make_prompt_cache(model), stats=stats, chooser=GammaChooser(FLAT)):
        out.append(int(t))
        _time.sleep(0.05)
    assert out == ref
    timed = [v[0] for v in stats.chooser.observed.values()]
    assert timed and max(timed) < 40                    # ms; a pass that counted one 50 ms consumer pause would exceed it


def test_apply_with_a_fixed_gamma_reaches_optiq_serves_call(setup):
    import optiq.runtime.spec as spec
    from optiq.runtime.spec import SpecStats
    import mlx_fastpath
    model, ref = setup
    drafter = StandInDrafter(PROMPT + ref, lambda p: False)
    try:
        assert mlx_fastpath.apply(assistant_gamma=3)["assistant_gamma"] == 3
        stats = SpecStats()
        out = [int(t) for t, _ in spec.assistant_generate_step(
            mx.array(PROMPT), model, drafter=drafter, gamma=2, max_tokens=GEN, prompt_cache=make_prompt_cache(model),
            stats=stats)]
    finally:
        mlx_fastpath.remove()
    assert out == ref
    assert stats.n_drafted > 2 * (stats.n_target_calls - 2)           # 3 drafts a pass, not serve's 2


def test_applying_twice_then_removing_gives_optiqs_loop_back():
    import optiq.runtime.spec as spec
    import mlx_fastpath
    stock = spec.assistant_generate_step
    try:
        mlx_fastpath.apply(assistant_gamma=1)
        mlx_fastpath.apply(assistant_gamma="auto")
        assert spec.assistant_generate_step is not stock
    finally:
        mlx_fastpath.remove()
    assert spec.assistant_generate_step is stock


def test_gamma_zero_is_plain_decoding_through_the_loop(setup):
    model, ref = setup
    out, stats = generate(model, StandInDrafter(PROMPT + ref, lambda p: False), gamma=0)
    assert out == ref
    assert stats.n_drafted == 0


def test_the_latest_apply_setting_is_the_one_in_force(setup):
    import optiq.runtime.spec as spec
    from optiq.runtime.spec import SpecStats
    import mlx_fastpath
    model, ref = setup
    drafter = StandInDrafter(PROMPT + ref, lambda p: False)
    try:
        mlx_fastpath.apply(assistant_gamma=1)
        mlx_fastpath.apply(assistant_gamma=3)
        stats = SpecStats()
        out = [int(t) for t, _ in spec.assistant_generate_step(
            mx.array(PROMPT), model, drafter=drafter, gamma=2, max_tokens=GEN, prompt_cache=make_prompt_cache(model),
            stats=stats)]
    finally:
        mlx_fastpath.remove()
    assert out == ref
    assert stats.n_drafted > 2 * (stats.n_target_calls - 2)           # 3 drafts a pass: the second setting won


def test_logits_processors_fall_back_to_plain_decoding(setup):
    """A caller with logits processors (repetition penalty etc.) gets mlx-lm's plain loop, as with optiq's."""
    model, ref = setup
    seen = []

    def proc(tokens, logits):
        seen.append(1)
        return logits

    out = [int(t) for t, _ in assistant.assistant_generate_step(
        mx.array(PROMPT), model, drafter=StandInDrafter(PROMPT + ref, lambda p: False), gamma="auto",
        max_tokens=GEN, prompt_cache=make_prompt_cache(model), logits_processors=[proc])]
    assert out == ref
    assert len(seen) >= GEN - 1
