"""Faster prompt processing for optiq's engine (lossless).

Stock optiq (and mlx-lm) run the LM head on every prompt position, then keep only the last one:
on Qwen3.5-9B that is a 248k-row head over each chunk, about 1 TFLOP per 512 tokens, thrown away.
Here the head runs on the last position only, and chunks are 1024 tokens (MLX's fused NAX attention for
head dim 256 starts at 1024 query rows). Measured on M5, 9B: 1.36x (4k), 1.23x (16k), 1.37x (32k)
faster prefill, final argmax identical, peak memory equal or lower (research/prefill).
"""
import mlx.core as mx

DEFAULT_STEP = 1024
_state = {}


def prefill_chunked(self, prompt_ids, cache, step=None):
    """Same contract as optiq's OptiqEngine._prefill_chunked: returns (hidden of the last chunk,
    logits whose [0, -1] is the next-token distribution)."""
    step = step or _state.get("step", DEFAULT_STEP)
    hidden = logits = None
    n = len(prompt_ids)
    for i in range(0, n, step):
        hidden = self._inner(mx.array([prompt_ids[i:i + step]]), cache=cache)
        if i + step >= n:
            logits = self._lm_head(hidden[:, -1:, :])
            mx.eval(logits, hidden)
        else:
            mx.eval([c.state for c in cache if c is not None])
    return hidden, logits


def enable(step=DEFAULT_STEP):
    from optiq.runtime.engine import OptiqEngine
    _state.setdefault("orig", OptiqEngine._prefill_chunked)
    _state["step"] = step
    OptiqEngine._prefill_chunked = prefill_chunked


def disable():
    from optiq.runtime.engine import OptiqEngine
    if "orig" in _state:
        OptiqEngine._prefill_chunked = _state["orig"]
