"""Gemma 4 `-assistant` drafter with an adaptive draft length (gamma) per pass.

optiq 0.8.5 drafts a fixed gamma per pass (`optiq serve --drafter` hardcodes 2). The best gamma depends on the
weights and the chip: on an M1, E2B int4 is fastest at gamma 1 (verifying 3 tokens costs 1.26-1.63x verifying 2) and
E2B OptiQ at gamma 2 (research/gemma/results.md). A pass costs about draft(gamma) + verify(gamma + 1).
The per-pass timing below leaves out the consumer's time between yields; the GPU may keep working then (async eval),
so with a very slow consumer passes read slightly cheap.

Here each pass picks gamma in 0..GAMMA_MAX to maximise expected tokens per second:
    E(g) = 1 + sum_{k<=g} prod_{j<=k} p_j        (p_j: chance draft j is accepted given j-1 were)
    C(g) = verify(g + 1) + draft(g)              (ms, measured on the live cache once per model, drafter and 4k
                                                  context bucket; the bucket is checked every pass)
p_j comes from decayed counts with a prior. Every PROBE_EVERY passes gamma is one more than the best, so positions
the best gamma never reaches are still measured (without it a cautious prior keeps gamma low for good).
Output is plain greedy decoding either way: every draft is checked against the target's own token.

The loop is optiq's `assistant_generate_step` (optiq.runtime.spec.runtime, MIT, itself after Ollama's mtp.go, MIT)
with the per-pass gamma choice added; its cache, rollback and yield contract are unchanged.
"""
import time

import mlx.core as mx

GAMMA_MAX = 3
PROBE_EVERY = 8              # every Nth pass drafts one deeper than the best, so deeper positions keep being measured
DECAY = 0.97                 # per-pass decay of the acceptance counts (adapts within a request)
OBS_WEIGHT = 0.3             # weight of the newest real pass time in its gamma's running average
OBS_MIN = 2                  # real passes needed before a gamma's own timing replaces the calibration
PRIOR = (2.0, 1.0)           # accepted, rejected pseudo-counts per draft position (p ~ 0.67 before any data)
CTX_BUCKET = 4096            # step costs are re-measured per this many tokens of context (checked every pass)
_state = {}


class GammaChooser:
    """Picks gamma per pass from step costs and the running per-position acceptance."""

    def __init__(self, costs, gamma_max=GAMMA_MAX, prior=PRIOR):
        self.costs, self.gamma_max, self.prior = costs, gamma_max, prior
        self.acc = [0.0] * (gamma_max + 1)        # index j = draft position (1-based)
        self.tried = [0.0] * (gamma_max + 1)
        self.passes = 0
        self.picks = {}                           # gamma -> passes (diagnostics)
        self.observed = {}                        # gamma -> [running ms of real passes, count]

    def p(self, j):
        a, b = self.prior
        return (self.acc[j] + a) / (self.tried[j] + a + b)

    def expected_tokens(self, g):
        e, run = 1.0, 1.0
        for j in range(1, g + 1):
            run *= self.p(j)
            e += run
        return e

    def calibrated(self, g):
        return self.costs["verify"][g + 1] + (self.costs["draft"][g] if g else 0.0)

    def cost(self, g):
        """ms per pass: the real passes' running average once a gamma has OBS_MIN of them; otherwise the start-up
        calibration, scaled by how real passes compared with it so far (the calibration times a bare verify; a real
        pass also normalises the verified positions over the vocabulary and reads the tokens back)."""
        seen = {k: v[0] for k, v in self.observed.items() if v[1] >= OBS_MIN}
        if g in seen:
            return seen[g]
        if not seen:
            return self.calibrated(g)
        scale = sum(seen[k] / self.calibrated(k) for k in seen) / len(seen)
        return self.calibrated(g) * scale

    def observe(self, g, ms):
        """One real pass at gamma g took ms (excluding time spent in the consumer)."""
        if g not in self.observed:
            self.observed[g] = [ms, 1]
        else:
            o = self.observed[g]
            o[0], o[1] = (1 - OBS_WEIGHT) * o[0] + OBS_WEIGHT * ms, o[1] + 1

    def set_costs(self, costs):
        """New calibration (the context crossed into a new bucket): real-pass timings from the old context go."""
        if costs is not self.costs:
            self.costs, self.observed = costs, {}

    def best(self, limit):
        top = max(0, min(self.gamma_max, limit))
        return max(range(top + 1), key=lambda g: self.expected_tokens(g) / self.cost(g))

    def choose(self, limit):
        g = self.best(limit)
        self.passes += 1
        if self.passes % PROBE_EVERY == 0 and g < min(self.gamma_max, limit):
            g += 1
        self.picks[g] = self.picks.get(g, 0) + 1
        return g

    def update(self, gamma, n_accepted):
        """gamma drafts were made, the first n_accepted were accepted (the next one, if any, was rejected)."""
        for j in range(1, self.gamma_max + 1):
            self.acc[j] *= DECAY
            self.tried[j] *= DECAY
        for j in range(1, min(gamma, n_accepted + 1) + 1):
            self.tried[j] += 1.0
            if j <= n_accepted:
                self.acc[j] += 1.0


def _copy_states(states):
    from mlx.utils import tree_map
    out = [tree_map(lambda a: mx.array(a) + 0 if isinstance(a, mx.array) else a, st) for st in states]
    mx.eval(out)
    return out


def _snapshot(caches):
    """Copies of every cache's state, so a measurement that writes into the cache can be undone exactly. Rolling back
    is not enough: a 1-token write into a full sliding-window ring buffer overwrites its oldest entry in place."""
    return _copy_states([c.state for c in caches])


def _restore(caches, snap):
    """Put the snapshot back as FRESH copies: the cache writes in place into whatever arrays it holds, so handing it
    the snapshot's own arrays would let the next write corrupt the snapshot."""
    for c, state in zip(caches, _copy_states(snap)):
        c.state = state                       # mlx-lm 0.32: offset, ring index etc. are part of state


def _costs_for(model, drafter, caches, cur, h_prev):
    """Step costs for this model + drafter at the cache's context bucket, measured once and kept ON the model object
    (a plain attribute, not a parameter), keyed by the drafter object itself: a freed model takes its costs with it,
    so a new model can never inherit them through a reused id()."""
    store = model.__dict__.setdefault("_fastpath_assistant_costs", {})
    entry = store.get(id(drafter))
    if entry is None or entry[0] is not drafter:
        entry = store[id(drafter)] = (drafter, {})
    bucket = caches[0].offset // CTX_BUCKET
    if bucket not in entry[1]:
        entry[1][bucket] = _measure_costs(model, drafter, caches, cur, h_prev, GAMMA_MAX)
    return entry[1][bucket]


def _measure_costs(model, drafter, caches, cur, h_prev, gamma_max, reps=2):
    """verify(n) for n = 1..gamma_max+1 and draft(g) for g = 1..gamma_max, in ms, on the live cache. The cache is
    snapshotted first and restored after every verify, outside the timed region: a real pass never pays the restore,
    and timing it would add the same cost to every width and bias gamma up."""
    from optiq.runtime.spec.kv_view import extract_typed_kv
    from optiq.runtime.spec.runtime import _target_embed, _target_step

    def timed(fn, after=None):
        best = float("inf")
        for _ in range(reps + 1):                 # first call warms up
            mx.synchronize()
            t = time.perf_counter()
            fn()
            mx.synchronize()
            best = min(best, (time.perf_counter() - t) * 1000)
            if after is not None:
                after()                           # not timed
        return best

    snap = _snapshot(caches)

    def verify(n):
        _, logits = _target_step(model, mx.array([[cur] * n]), caches)
        mx.argmax(logits[0], axis=-1).tolist()

    def restore():
        _restore(caches, snap)

    def draft(g):
        shared_kv, position = extract_typed_kv(model, caches), caches[0].offset
        tok_in, h_in = cur, h_prev
        for k in range(g):
            d_logits, h_in = drafter.forward(last_token_emb=_target_embed(model, tok_in), target_hidden=h_in,
                                             shared_kv=shared_kv, position=position + k)
            tok_in = int(mx.argmax(d_logits))

    try:
        return {"verify": {n: timed(lambda: verify(n), after=restore) for n in range(1, gamma_max + 2)},
                "draft": {g: timed(lambda: draft(g)) for g in range(1, gamma_max + 1)}}
    finally:
        _restore(caches, snap)                    # also when a measurement raises


def assistant_generate_step(prompt, model, *, drafter, gamma="auto", max_tokens=256, sampler=None,
                            logits_processors=None, prompt_cache=None, prefill_step_size=2048,
                            prompt_progress_callback=None, stats=None, chooser=None, **_unused):
    """Drop-in for optiq's `assistant_generate_step` (and so for mlx-lm's `generate_step`): yields
    `(token, logprobs)`. `gamma="auto"` picks the draft length per pass; an int fixes it (optiq's behaviour).
    `chooser` may be passed to fix the costs (tests) or to read the chooser's state afterwards."""
    from mlx_lm.models.cache import make_prompt_cache
    from optiq.runtime.ngram_spec import _PLAIN_GENERATE_STEP, eval_cache_state
    from optiq.runtime.spec import SpecStats
    from optiq.runtime.spec.kv_view import extract_typed_kv
    from optiq.runtime.spec.runtime import _target_embed, _target_step, _trim_cache

    if drafter is None or logits_processors or prompt.size == 0:
        yield from _PLAIN_GENERATE_STEP(prompt, model, max_tokens=max_tokens, sampler=sampler,
                                        logits_processors=logits_processors, prompt_cache=prompt_cache,
                                        prefill_step_size=prefill_step_size,
                                        prompt_progress_callback=prompt_progress_callback)
        return
    adaptive = gamma == "auto"
    fixed_costs = chooser is not None                 # a caller-supplied chooser keeps its costs
    stats = stats if stats is not None else SpecStats()
    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
    caches = prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    t0 = time.time()

    y = prompt.astype(mx.uint32)
    total = y.size
    while y.size > 1:
        n = min(prefill_step_size, y.size - 1)
        model(y[:n][None], cache=caches)
        eval_cache_state(caches)
        if prompt_progress_callback is not None:
            prompt_progress_callback(total - y.size + n, total)
        y = y[n:]
        mx.clear_cache()

    cur = int(y[-1].item())
    cur_is_output = False
    h_prev = None
    prev_lp = None
    ntoks = 0
    pending = None
    try:
        while True:
            remaining = max_tokens - ntoks - (1 if cur_is_output else 0)
            limit = remaining - 1
            if h_prev is None or limit <= 0:
                budget = 0
            elif not adaptive:
                budget = min(gamma, limit)
            else:
                if not fixed_costs:
                    costs = _costs_for(model, drafter, caches, cur, h_prev)
                    if chooser is None:
                        chooser = GammaChooser(costs)
                    chooser.set_costs(costs)              # the context may have crossed into a new bucket
                budget = chooser.choose(limit)
            drafts = []
            t_pass, t_away = time.perf_counter(), 0.0
            if budget > 0:
                position = caches[0].offset
                shared_kv = extract_typed_kv(model, caches)
                tok_in, h_in = cur, h_prev
                for k in range(budget):
                    d_logits, h_next = drafter.forward(
                        last_token_emb=_target_embed(model, tok_in), target_hidden=h_in,
                        shared_kv=shared_kv, position=position + k)
                    d = int(mx.argmax(d_logits))
                    drafts.append(d)
                    tok_in, h_in = d, h_next
            inp = [cur] + drafts
            hidden, logits = _target_step(model, mx.array([inp]), caches)
            stats.n_target_calls += 1
            lp = logits[0] - mx.logsumexp(logits[0], axis=-1, keepdims=True)
            toks = sampler(lp)
            mx.async_eval(toks)
            pending = [len(inp), 1]
            if cur_is_output:
                ntoks += 1
                stats.n_emitted += 1
                t_yield = time.perf_counter()
                yield cur, prev_lp
                t_away += time.perf_counter() - t_yield
                if ntoks >= max_tokens:
                    break
            toks = toks.tolist()
            if adaptive and chooser is not None and h_prev is not None:
                chooser.observe(len(drafts), (time.perf_counter() - t_pass - t_away) * 1000)
            n = 0
            while n < len(drafts) and toks[n] == drafts[n]:
                n += 1
            stats.n_drafted += len(drafts)
            stats.n_accepted += n
            if adaptive and chooser is not None and drafts:
                chooser.update(len(drafts), n)
            stopped = False
            for j in range(n):
                pending[1] = j + 2
                ntoks += 1
                stats.n_emitted += 1
                yield drafts[j], lp[j]
                if ntoks >= max_tokens:
                    stopped = True
                    break
            if stopped:
                break
            _trim_cache(caches, len(inp) - (n + 1))
            pending = None
            cur, prev_lp, h_prev = toks[n], lp[n], hidden[:, n:n + 1, :]
            cur_is_output = True
    finally:
        if pending is not None:
            _trim_cache(caches, pending[0] - pending[1])
        stats.elapsed_s = time.time() - t0
        stats.chooser = chooser                  # diagnostics: picks, costs, acceptance


def check_gamma(gamma):
    if gamma != "auto" and (isinstance(gamma, bool) or not isinstance(gamma, int) or gamma < 0):
        raise ValueError(f"assistant gamma must be 'auto' or an int >= 0, got {gamma!r}")


def enable(gamma="auto"):
    """Swap optiq's assistant loop for this one, so `optiq serve --drafter` (which hardcodes gamma=2) and direct callers
    use `gamma` instead: an int fixes the draft length (measured best on M1: 1 for E2B int4, 2 for E2B OptiQ), "auto"
    picks it per pass (opt-in: on M1 it matched optiq's gamma 2 on average, not the best fixed gamma;
    research/gemma)."""
    import optiq.runtime.spec as spec

    check_gamma(gamma)
    _state.setdefault("orig", spec.assistant_generate_step)

    chosen = gamma

    def swapped(prompt, model, *, drafter, gamma=None, **kw):       # the caller's gamma (serve: 2) is replaced
        return assistant_generate_step(prompt, model, drafter=drafter, gamma=chosen, **kw)

    spec.assistant_generate_step = swapped


def disable():
    import optiq.runtime.spec as spec
    if "orig" in _state:
        spec.assistant_generate_step = _state.pop("orig")
