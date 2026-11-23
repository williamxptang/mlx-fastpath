"""0.8B, sky prompt: which single fastpath swap turns generate_ngram's first token from stock's '1' into the model's 'The'."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import mlx_fastpath as fp
from mlx_fastpath.guard import MemoryGuard
from optiq.runtime import ngram_spec
M = MODELS + "/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed"
OFF = dict(attention_dispatch=False, draft_vocab=0, fused_capture=False, prefill_step=None)
CFG = {"stock": None, "all": {}, "attention only": {**OFF, "attention_dispatch": True},
       "draft head only": {**OFF, "draft_vocab": 32768}, "fused capture only": {**OFF, "fused_capture": True},
       "prefill only": {**OFF, "prefill_step": 1024}}
with MemoryGuard(min_free_gb=6, max_footprint_gb=4.5):
    eng = fp.load(M, memory_limit_gb=2.5, n2=False)
    tok = eng.tokenizer
    ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": "Explain in five sentences why the sky is blue."}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
    for name, kw in CFG.items():
        fp.remove()
        done = fp.apply(**kw) if kw is not None else {}
        for s in (ngram_spec._COST_MODELS, ngram_spec._PASS_COSTS, ngram_spec._ACCEPTANCE_PRIOR, ngram_spec._MTP_DRAFT_COSTS):
            s.clear()
        out = fp.generate_ngram(eng, ids, max_tokens=6)
        fp.remove()
        print(f"{name:20s} {sorted(done)} -> {[tok.decode([x]) for x in out]}")
