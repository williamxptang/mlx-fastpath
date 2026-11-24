"""A/B: does fastpath's attention patch speed up mlx-lm's OWN generate loop (no optiq engine)?

Only `mlx_fastpath.attention` is switched (enable/disable); nothing from optiq is used. Decoding runs through
`mlx_lm.generate.generate_step`, exactly as `mlx_lm.generate --kv-bits ...` would.

Per (model, prompt length, KV mode): the prompt minus its last token is prefilled ONCE the way generate_step does
(2048-token chunks, maybe_quantize_kv_cache after each chunk, quantized_kv_start=5000 as mlx-lm's default) and saved
with save_prompt_cache. Each run loads that cache fresh and decodes GEN greedy tokens with generate_step.
Order per cell: off, on, on, off (ABBA). Reports decode tok/s (median of 2), tokens identical on/off, and whether
the two "off" runs agree (checks the cache restore).

KV modes: bf16 (control: decode is qL=1, the patch routes it to stock, expect level) and q4 (--kv-bits 4: the
patch's fused 4-bit kernel replaces mlx-lm's quantized attention).

    python mlxlm_ab.py TEXT_DIR OUT_DIR MODEL_DIR [MODEL_DIR ...]
"""
import glob
import json
import os
import statistics
import sys
import time

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # this repo's mlx_fastpath, not an installed copy
import mlx.core as mx

mx.set_memory_limit(6 * 2**30)
mx.set_cache_limit(1 * 2**30)

from mlx_lm import load
from mlx_lm.generate import generate_step, maybe_quantize_kv_cache
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

from mlx_fastpath import attention

assert Path(attention.__file__).resolve().parents[1] == Path(__file__).resolve().parents[2], attention.__file__

text_dir, out_dir, models = sys.argv[1], sys.argv[2], sys.argv[3:]
assert models, "give at least one model dir"
os.makedirs(out_dir, exist_ok=True)
CACHE_FILE = os.path.join(out_dir, "ab_cache.safetensors")   # overwritten per cell
GEN = 256
Q_START, GROUP, STEP = 5000, 64, 2048
CELLS = [(4096, None), (16384, None), (16384, 4), (32768, None), (32768, 4)]
if os.environ.get("SMOKE"):
    GEN, CELLS = 16, [(6000, 4)]

files = sorted(glob.glob(os.path.join(text_dir, "**", "*.md"), recursive=True))
assert files, "no .md files"
raw = "\n\n".join(open(p).read() for p in files) * 3


def prompt_ids(tok, n):
    body = tok.decode(tok.encode(raw)[:n])
    ids = tok.apply_chat_template([{"role": "user", "content": body + "\n\nSummarize the text above in one paragraph."}],
                                  add_generation_prompt=True, tokenize=True)
    return ids


def prefill(model, ids, kv_bits):
    cache = make_prompt_cache(model)
    for i in range(0, len(ids), STEP):
        model(mx.array(ids[i:i + STEP])[None], cache=cache)
        maybe_quantize_kv_cache(cache, Q_START, GROUP, kv_bits)
        mx.eval([c.state for c in cache])
    save_prompt_cache(CACHE_FILE, cache)
    kinds = sorted({type(c).__name__ for c in cache})
    del cache
    mx.clear_cache()
    return kinds


def decode(model, last_id, kv_bits, patched):
    (attention.enable if patched else attention.disable)()
    cache = load_prompt_cache(CACHE_FILE)
    mx.reset_peak_memory()
    toks, t0, t_first = [], None, None
    t0 = time.perf_counter()
    for tok, _ in generate_step(mx.array([last_id]), model, max_tokens=GEN, prompt_cache=cache, kv_bits=kv_bits,
                                kv_group_size=GROUP, quantized_kv_start=Q_START, prefill_step_size=STEP):
        toks.append(int(tok))
        if t_first is None:
            t_first = time.perf_counter()
    t_end = time.perf_counter()
    attention.disable()
    kinds = sorted({type(c).__name__ for c in cache})
    del cache
    mx.clear_cache()
    rate = (len(toks) - 1) / (t_end - t_first)          # decode rate after the first token
    return toks, rate, mx.get_peak_memory() / 2**30, kinds


with open(os.path.join(out_dir, "mlxlm_ab.jsonl"), "a") as out:
    for mdir in models:
        model, tok = load(mdir)
        for n, kv_bits in CELLS:
            ids = prompt_ids(tok, n)
            pre_kinds = prefill(model, ids[:-1], kv_bits)
            runs = {False: [], True: []}
            for patched in (False, True, True, False):
                runs[patched].append(decode(model, ids[-1], kv_bits, patched))
            off, on = runs[False], runs[True]
            first_diff = next((i for i, (a, b) in enumerate(zip(off[0][0], on[0][0])) if a != b), None)
            rec = {"model": os.path.basename(mdir.rstrip("/")), "prompt_tokens": len(ids),
                   "kv": f"q{kv_bits}" if kv_bits else "bf16", "cache_after_prefill": pre_kinds,
                   "cache_in_decode": on[0][3],
                   "off_tok_s": [round(r[1], 2) for r in off], "on_tok_s": [round(r[1], 2) for r in on],
                   "speedup": round(statistics.median(r[1] for r in on) / statistics.median(r[1] for r in off), 3),
                   "tokens_same": off[0][0] == on[0][0], "first_diff": first_diff,
                   "off_runs_agree": off[0][0] == off[1][0], "gen": len(on[0][0]),
                   "peak_gb_off": round(max(r[2] for r in off), 2), "peak_gb_on": round(max(r[2] for r in on), 2)}
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(json.dumps(rec), flush=True)
        del model
        mx.clear_cache()
