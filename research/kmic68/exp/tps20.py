"""Can Qwen3.5-9B-OptiQ-4bit decode at >= 20 tok/s on M5? optiq MTP engine + runtime swaps (no files edited):
  cap   = fused GDN capture backend (optiq's linear_gdn_from_conv) instead of the per-token loop
  vocab = reduced-vocab draft head: drafts score only token ids < K (lossless: full model still verifies)
  fold  = attention dispatch: bf16 KV -> GQA fold when gqa*qL >= 16; 4-bit KV -> fused kernel (qL<=3) / folded qmm
  q4    = 4-bit KV cache (as the model's kv_config.json)
Greedy, temp 0. Capped: MLX limit 8 GB + watchdog kill > 9 GB footprint."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import sys, os, time, json, threading, subprocess, re, glob
sys.path.insert(0, "<staging>/exp")
import mlx.core as mx
mx.set_memory_limit(8 * 2**30); mx.set_cache_limit(1 * 2**30)


def watchdog(limit_gb=9.0):
    pid = os.getpid()
    while True:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True).stdout
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*(GB|MB|KB)", out)
        if m:
            gb = float(m.group(1)) * {"GB": 1, "MB": 1 / 1024, "KB": 1 / 2**20}[m.group(2)]
            watchdog.peak = max(watchdog.peak, gb)
            if gb > limit_gb:
                print(f"WATCHDOG: {gb:.1f} GB > {limit_gb} GB, exiting", flush=True); os._exit(3)
        time.sleep(2)


watchdog.peak = 0.0
watchdog.loaded = False
threading.Thread(target=watchdog, daemon=True).start()

# Load BEFORE importing optiq: optiq's import-time mlx-lm patches lift the load peak from 7.2 to 10.9 GB.
from mlx_lm import load as _load
_m, _t = _load(MODELS + "/Qwen3.5-9B-OptiQ-4bit")
watchdog.loaded = True
from optiq.runtime.engine import OptiqEngine
import optiq.runtime.mtp.gdn_capture as gc
import mlx_lm.models.qwen3_next as qn
import mlx_lm.models.base as base
from mlx_lm.models.cache import QuantizedKVCache
from folded_ops import folded_sdpa, folded_qsdpa
from qkv_attn import qkv_sdpa2

MODEL = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
GEN = 160
eng = OptiqEngine.from_loaded(_m, _t, MODEL)  # no duplicate model copy (OptiqEngine(path) peaks at 10 GB)
watchdog.loaded = True; watchdog.peak = 0.0
print('loaded, load peak was over; steady-state cap 9 GB', 'has_mtp', eng.has_mtp, flush=True)
tok = eng.tokenizer
lm = eng._lm

# ---- swaps ----
_orig_capture = gc.forward_with_gdn_capture
_orig_attn = base.scaled_dot_product_attention
_orig_draft = OptiqEngine._mtp_draft


def capture_fused(*a, **k):
    k.setdefault("capture_backend", "linear_gdn_from_conv")
    return _orig_capture(*a, **k)


def attn_dispatch(queries, keys, values, cache, scale, mask, sinks=None):
    B, Hq, qL, D = queries.shape
    plain_mask = mask is None if qL == 1 else (isinstance(mask, str) and mask == "causal")
    if B == 1 and sinks is None and plain_mask and D == 256 and qL <= 16:
        g = Hq // keys[0].shape[1] if isinstance(keys, (tuple, list)) else Hq // keys.shape[1]
        if isinstance(cache, QuantizedKVCache):
            if qL <= 3:
                return qkv_sdpa2(queries, cache.keys, cache.values, scale, cache.bits, n_valid=cache.offset,
                                 block=512 if cache.bits == 4 else 256, nsg=4)
            return folded_qsdpa(queries, keys, values, scale, cache.bits)
        if g * qL >= 16:
            return folded_sdpa(queries, keys, values, scale)
    return _orig_attn(queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks)


head = lm.lm_head
V = head.weight.shape[0]
SUB = {}


def make_sub(K):
    SUB[K] = (head.weight[:K], head.scales[:K], head.biases[:K]); mx.eval(SUB[K])


def draft_reduced(self, hidden_last, next_token_id, mtp_cache):
    _, post = self._lm._mtp_core(hidden_last, mx.array([[next_token_id]]), mtp_cache=mtp_cache, emit_logits=False)
    w, s, b = SUB[draft_reduced.K]
    sub = mx.quantized_matmul(post, w, s, b, transpose=True, group_size=head.group_size, bits=head.bits)
    full = mx.full((1, 1, V), -1e30, dtype=sub.dtype)
    full[..., :draft_reduced.K] = sub
    return full, post


def configure(cap=False, vocab=0, fold=False):
    gc.forward_with_gdn_capture = capture_fused if cap else _orig_capture
    qn.scaled_dot_product_attention = attn_dispatch if fold else _orig_attn
    if vocab:
        draft_reduced.K = vocab
        if vocab not in SUB: make_sub(vocab)
        OptiqEngine._mtp_draft = draft_reduced
    else:
        OptiqEngine._mtp_draft = _orig_draft


docs = sorted(glob.glob("<staging>/research/261118-transferable-ideas/**/*.md", recursive=True))
corpus = tok.encode("\n\n".join(open(p).read() for p in docs) * 3)
ask = "\n\nSummarize the document above as a numbered list of its key findings, with numbers."


def prompt_ids(n):
    if n == 0:
        body = "Explain, step by step, how a CPU cache works and why cache lines matter for performance."
    else:
        body = tok.decode(corpus[:n]) + ask
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False))


CONFIGS = [  # name, depth, kv_bits, swaps
    ("AR", 0, None, {}),
    ("AR+q4kv+fold", 0, 4, dict(fold=True)),
    ("MTP1", 1, None, {}),
    ("MTP2", 2, None, {}),
    ("MTP1+cap", 1, None, dict(cap=True)),
    ("MTP2+cap", 2, None, dict(cap=True)),
    ("MTP1+cap+vocab32k", 1, None, dict(cap=True, vocab=32768)),
    ("MTP2+cap+vocab32k", 2, None, dict(cap=True, vocab=32768)),
    ("MTP3+cap+vocab32k+fold", 3, None, dict(cap=True, vocab=32768, fold=True)),
    ("MTP2+cap+vocab32k+fold+q4kv", 2, 4, dict(cap=True, vocab=32768, fold=True)),
]
for n in (0, 4000, 8000):
    ids = prompt_ids(n)
    ref = None
    for name, depth, kvb, sw in CONFIGS:
        configure(**sw)
        st = eng.generate("", prompt_ids=ids, max_tokens=GEN, depth=depth, temperature=0.0, kv_bits=kvb)
        gen = tok.encode(st.text) if False else None
        if name == "AR":
            ref = st.text
            ar_ids = tok.encode(st.text)
            cov = {K: round(sum(t < K for t in ar_ids) / len(ar_ids), 3) for K in (16384, 32768, 65536)}
        same = next((i for i, (a, b) in enumerate(zip(st.text, ref)) if a != b), min(len(st.text), len(ref)))
        print(json.dumps({"prompt_tokens": len(ids), "config": name, "tok_s": round(st.decode_tok_s, 2),
                          "accept": round(st.acceptance_rate, 3) if depth else None, "tokens": st.generated_tokens,
                          "same_text_as_AR": st.text == ref, "common_prefix_chars": same, "of_chars": len(ref),
                          **({"vocab_coverage_of_AR_tokens": cov} if name == "AR" else {}),
                          "peak_footprint_gb": round(watchdog.peak, 2)}), flush=True)
    configure()
