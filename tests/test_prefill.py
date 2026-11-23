"""Last-position-only prefill gives the same next-token logits as running the LM head on every position."""
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models import qwen3
from mlx_lm.models.cache import make_prompt_cache

from mlx_fastpath.prefill import prefill_chunked


@pytest.fixture
def tiny():
    mx.random.seed(0)
    args = qwen3.ModelArgs(model_type="qwen3", hidden_size=128, num_hidden_layers=2, intermediate_size=256,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=32, rms_norm_eps=1e-6,
                           vocab_size=500, max_position_embeddings=4096, rope_theta=10000.0, tie_word_embeddings=False)
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    return model


def _stock(model, ids, step):
    cache = make_prompt_cache(model)
    for i in range(0, len(ids), step):
        logits = model.lm_head(model.model(mx.array([ids[i:i + step]]), cache=cache))
    return logits[0, -1]


@pytest.mark.parametrize("n,step", [(700, 256), (700, 1024), (513, 512)])
def test_last_position_prefill_matches_full_logits(tiny, n, step):
    ids = (mx.arange(n) * 37 % 500).tolist()
    eng = SimpleNamespace(_inner=tiny.model, _lm_head=tiny.lm_head)
    hidden, logits = prefill_chunked(eng, ids, make_prompt_cache(tiny), step=step)
    ref = _stock(tiny, ids, step)
    assert logits.shape[1] == 1
    # one-row vs whole-chunk matmul take different kernels (fp32 runs at TF32-like precision on M5): ~1e-3
    assert mx.allclose(logits[0, -1], ref, rtol=5e-3, atol=5e-3).item()
    assert logits[0, -1].argmax().item() == ref.argmax().item()
