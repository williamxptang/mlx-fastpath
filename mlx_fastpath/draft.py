"""Faster MTP drafting for optiq's engine (lossless: the full model still verifies every token).

reduced-vocab draft head: drafts score only token ids < K (rows of the LM head). On Qwen3.5-9B the
  full head is 1 GB (8-bit, 248k rows) per draft step; K = 32768 reads 1/8 of it. The lowest 32k ids
  covered 88-98% of the tokens the model produced. Measured: +11-24% tok/s.
id-set draft head (switch, default off): the first K ids plus the prompt's own ids, so copied names, code and
  numbers above id K can be drafted too. With K = 32768, 10-14% of the tokens of code edits and long prompts are
  above K; with the prompt's ids 0-5% are (research/batch2/coverage.py). Measured on DFlash drafts: +5-12% tok/s,
  same tokens (results.md §16, §17.2). Not yet measured on MTP drafts.
fused GDN capture: use optiq's own linear_gdn_from_conv kernel instead of its per-token Python loop
  when verifying on Qwen3.5/3.6 hybrid models. Measured: +3-12% tok/s.
safe head pricing (with the fast drafter): optiq 0.8.5 prices the MTP head before an n-gram run with the packed width
  of a quantized embedding (its own drafter then turns itself off) and times 5 target steps on the live cache, which a
  hybrid GDN cache cannot trim back (the first token is then not greedy). safe_draft_cost does neither
  (research/batch5/results.md).
"""
import itertools

import mlx.core as mx
import mlx.nn as nn

DEFAULT_VOCAB = 32768
_state = {}
_serial = itertools.count(1)


def sub_head(head, k):
    """First k output rows of an LM head (nn.Linear, nn.QuantizedLinear, or an embedding used as_linear)."""
    if hasattr(head, "scales"):
        return ("q", head.weight[:k], head.scales[:k], head.biases[:k], head.group_size, head.bits)
    return ("f", head.weight[:k])


def sub_logits(sub, x):
    if sub[0] == "q":
        _, w, s, b, gs, bits = sub
        return mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)
    return x @ sub[1].T


def expand(sub_out, vocab):
    """Scatter the first-k logits into a full-vocab vector; other ids get -1e30 so they are never drafted."""
    k = sub_out.shape[-1]
    full = mx.full(sub_out.shape[:-1] + (vocab,), -1e30, dtype=sub_out.dtype)
    full[..., :k] = sub_out
    return full


def id_set(prompt_ids, k):
    """Sorted draft ids: 0..k-1 plus every prompt id (each once)."""
    return list(range(k)) + sorted({int(t) for t in prompt_ids if int(t) >= k})


def rows_head(head, idx):
    """The LM head's rows at the ids idx (an int32 array), in the same form as sub_head."""
    if hasattr(head, "scales"):
        return ("q", head.weight[idx], head.scales[idx], head.biases[idx], head.group_size, head.bits)
    return ("f", head.weight[idx])


def scatter(sub_out, idx, vocab):
    """Put the logits of the ids idx into a full-vocab vector; other ids get -1e30 so they are never drafted."""
    full = mx.full(sub_out.shape[:-1] + (vocab,), -1e30, dtype=sub_out.dtype)
    return mx.put_along_axis(full, mx.broadcast_to(idx, sub_out.shape), sub_out, axis=-1)


def set_prompt(prompt_ids):
    """generate() calls this around each request (None afterwards), so drafts may use the prompt's ids."""
    _state["prompt"] = None if prompt_ids is None else (next(_serial), list(prompt_ids))


def _head_of(lm):
    return lm.model.embed_tokens if getattr(lm.args, "tie_word_embeddings", False) else lm.lm_head


def _draft_reduced(self, hidden_last, next_token_id, mtp_cache):
    k = _state["vocab"]
    prompt = _state.get("prompt") if _state.get("use_prompt_ids") else None
    cache_key = ("_mlx_fastpath_subhead", k, prompt[0] if prompt else None)
    sub = getattr(self, "_mlx_fastpath_subhead", None)
    if sub is None or sub[0] != cache_key:                 # built once per request
        head = _head_of(self._lm)
        if prompt:
            idx = mx.array(id_set(prompt[1], k), dtype=mx.int32)
            sub = (cache_key, rows_head(head, idx), head.weight.shape[0], idx)
        else:
            sub = (cache_key, sub_head(head, k), head.weight.shape[0], None)
        self._mlx_fastpath_subhead = sub
    _, sh, vocab, idx = sub
    _, post = self._lm._mtp_core(hidden_last, mx.array([[next_token_id]]), mtp_cache=mtp_cache,
                                 emit_logits=False)
    logits = sub_logits(sh, post)
    return (expand(logits, vocab) if idx is None else scatter(logits, idx, vocab)), post


def enable_reduced_vocab(vocab=DEFAULT_VOCAB, prompt_ids=False):
    from optiq.runtime.engine import OptiqEngine
    _state.setdefault("orig_draft", OptiqEngine._mtp_draft)
    _state["vocab"] = vocab
    _state["use_prompt_ids"] = prompt_ids
    OptiqEngine._mtp_draft = _draft_reduced


def disable_reduced_vocab():
    from optiq.runtime.engine import OptiqEngine
    if "orig_draft" in _state:
        OptiqEngine._mtp_draft = _state["orig_draft"]
    _state["use_prompt_ids"] = False


def enable_fused_capture(backend="linear_gdn_from_conv"):
    import optiq.runtime.mtp.gdn_capture as gc
    orig = _state.setdefault("orig_capture", gc.forward_with_gdn_capture)

    def forward_with_gdn_capture(*a, **k):
        k.setdefault("capture_backend", backend)
        return orig(*a, **k)

    gc.forward_with_gdn_capture = forward_with_gdn_capture


def disable_fused_capture():
    import optiq.runtime.mtp.gdn_capture as gc
    if "orig_capture" in _state:
        gc.forward_with_gdn_capture = _state["orig_capture"]


def _fast_mtp_drafter_class(vocab):
    """optiq's MtpDrafter (used by its n-gram speculator, incl. `optiq serve --ngram-draft N --mtp`) with the
    reduced-vocab head. Measured with n-gram drafting on Qwen3.5-9B: 27-71 tok/s (code edit 71, copy 59, chat 27)."""
    from optiq.runtime import ngram_spec

    base = _state.get("orig_mtp_drafter", ngram_spec.MtpDrafter)

    class FastMtpDrafter(base):
        def __init__(self, model, *, depth=2):
            super().__init__(model, depth=depth)
            self.sub = sub_head(_head_of(self.model), vocab) if self.ok else None

        def draft(self, hidden, token, position=None):
            if not self.ok:
                return []
            out, h, nxt = [], hidden.reshape(1, 1, -1), token
            for i in range(self.depth):
                _, h = self.model._mtp_core(h, mx.array([[nxt]], mx.uint32), mtp_cache=self.model.make_mtp_cache(),
                                            emit_logits=False,
                                            position_offset=None if position is None else position + i)
                nxt = int(mx.argmax(sub_logits(self.sub, h)[0, -1]).item())
                out.append(nxt)
                h = h[:, -1:, :]
            return out

    return FastMtpDrafter


COST_CONTEXT = 1024   # tokens of context on the scratch cache the target step is timed on


def _context_len(cache):
    return max((getattr(c, "offset", 0) or 0 for c in cache or []), default=0)


def safe_draft_cost(drafter, model, hidden=None, token=0, *, reps=5, cache=None):
    """optiq's ngram_spec.measure_draft_cost (one head draft, in plain decode steps) without its two faults
    (optiq 0.8.5, research/batch5/results.md):
    - its fake hidden state takes the width of `embed_tokens.weight`, which on a quantized embedding is the PACKED
      width (256 for 1024 at 8 bits): the head's draft fails, and optiq's own drafter then turns itself off;
    - it times 5 target steps on the LIVE cache and undoes them with a trim, which does nothing on a hybrid GDN cache
      (ArraysCache cannot be trimmed): the 5 fake tokens stay in the state and shift the next real token.
    Here the width is the model's hidden size, and the target step is timed on a scratch cache holding up to
    COST_CONTEXT tokens of context (as long as the live one, capped), so the live cache is never touched.
    The scratch cache is the model's default (bf16 KV) even when the live one is 4-bit, so the step cost is an
    approximation; optiq keeps the result per model, so the scratch prefill is paid once."""
    import time
    from mlx_lm.models.cache import make_prompt_cache
    if not drafter.ok:
        return 0.0
    try:
        if hidden is None:
            hidden = mx.zeros((1, 1, drafter.model.args.hidden_size), mx.bfloat16)
        scratch = make_prompt_cache(model)
        n = min(_context_len(cache), COST_CONTEXT)
        if n:
            model(mx.zeros((1, n), mx.uint32), cache=scratch)
            mx.eval([c.state for c in scratch])
        drafter.draft(hidden, token)                     # warm up
        draft_t, step_t = [], []
        for _ in range(reps):
            t0 = time.perf_counter(); drafter.draft(hidden, token); draft_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            mx.eval(model(mx.array([[token]], mx.uint32), cache=scratch))
            step_t.append(time.perf_counter() - t0)
        draft_t.sort(); step_t.sort()
        step = step_t[len(step_t) // 2]
        return draft_t[len(draft_t) // 2] / step if step > 0 else 0.0
    except Exception:                                    # noqa: BLE001 - costing must not raise (as optiq's)
        return 0.0


def enable_fast_mtp_drafter(vocab=DEFAULT_VOCAB):
    from optiq.runtime import ngram_spec
    _state.setdefault("orig_mtp_drafter", ngram_spec.MtpDrafter)
    _state.setdefault("orig_draft_cost", ngram_spec.measure_draft_cost)
    ngram_spec.MtpDrafter = _fast_mtp_drafter_class(vocab)
    ngram_spec.measure_draft_cost = safe_draft_cost


def disable_fast_mtp_drafter():
    from optiq.runtime import ngram_spec
    if "orig_mtp_drafter" in _state:
        ngram_spec.MtpDrafter = _state.pop("orig_mtp_drafter")
    if "orig_draft_cost" in _state:
        ngram_spec.measure_draft_cost = _state.pop("orig_draft_cost")
