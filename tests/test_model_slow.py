"""Real model, opt-in (MLX_FASTPATH_SLOW=1; MLX_FASTPATH_MODEL picks the model, default Qwen3.5-9B-OptiQ-4bit):
with the speedups applied, the model answers a 4k-context task with ground truth, without loops, and never swaps.
Memory caps come from the model's size (real_model.caps). The speed floor applies only to models that declare one."""
import os
from pathlib import Path

import pytest

from real_model import LLM, caps, slow_enabled

MODEL = os.environ.get("MLX_FASTPATH_MODEL", str(LLM / "Qwen3.5-9B-OptiQ-4bit"))
SPEED_FLOOR_TOK_S = {"Qwen3.5-9B-OptiQ-4bit": 20.0}     # decode tok/s at 4k context, measured on M5
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not slow_enabled() or not Path(MODEL).exists(),
                                 reason="opt-in: MLX_FASTPATH_SLOW=1 and the model on disk")]
CODES = ["KESTREL-4471", "MAGENTA-0928", "OBSIDIAN-3365"]


def test_fast_decode_is_fast_and_correct():
    import mlx_fastpath as fp
    mlx_limit, max_footprint, min_free = caps(MODEL)
    guard = fp.MemoryGuard(min_free_gb=min_free, max_footprint_gb=max_footprint).__enter__()  # never swap
    engine = fp.load(MODEL, memory_limit_gb=mlx_limit)
    assert fp.apply()
    try:
        tok = engine.tokenizer
        filler = (Path(__file__).resolve().parent.parent / "research/kmic68/research/261118-transferable-ideas/report.md").read_text()
        body = tok.encode(filler * 4)[:4000]
        for i, (frac, code) in enumerate(zip((0.1, 0.5, 0.9), CODES)):
            at = int(len(body) * frac)
            body[at:at] = tok.encode(f"\n\nThe secret code number {i + 1} is {code}.\n\n")
        text = tok.decode(body) + "\n\nList the three secret codes in the document exactly, then summarize it in 3 bullets."
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False))
        st = fp.generate(engine, ids, max_tokens=160)
    finally:
        fp.remove()
    import mlx.core as mx
    print(f"\n[slow] {Path(MODEL).name}: {st.generated_tokens} tokens at {st.decode_tok_s:.2f} tok/s, accept "
          f"{st.acceptance_rate:.2f}, MLX peak {mx.get_peak_memory() / 2**30:.2f} GB (cap {mlx_limit}), codes "
          f"{sum(c in st.text for c in CODES)}/3, {guard.report()}")
    assert guard.report()["swap_growth_gb"] <= 0.0625, "swap grew"
    assert all(c in st.text for c in CODES), st.text[:400]
    gen = tok.encode(st.text)
    grams = [tuple(gen[i:i + 4]) for i in range(len(gen) - 3)]
    assert len(grams) > 0 and len(set(grams)) / len(grams) > 0.9, "looping output"
    floor = SPEED_FLOOR_TOK_S.get(Path(MODEL).name)
    if floor is not None:
        assert st.decode_tok_s >= floor, st.decode_tok_s
