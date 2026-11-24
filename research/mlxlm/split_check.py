"""Are the 4-bit-KV token splits (mlx-lm alone vs + fastpath attention) near-ties, and which side is closer to the
model without KV quantization?

Per (model, N) with 4-bit KV:
  1. Prefill the same prompt twice, as mlx-lm's generate_step does: once with 4-bit KV (quantized_kv_start=5000)
     -> Q, once with a plain bf16 KV cache -> F (the reference: no KV quantization).
  2. Greedy-decode GEN tokens from Q with mlx-lm alone (A) and with the patch (B) via generate_step; d = first split.
  3. Teacher-force A's tokens up to d (one token per step, as in decoding) three ways: Q + stock attention ("off"),
     Q + fastpath attention ("on"), F + stock attention ("ref"). At every step compare off and on to ref
     (KL(ref||x), top-1 agreement); at the split step report the logit gap between the two contested tokens in each.

    python split_check.py TEXT_DIR OUT_DIR MODEL_DIR [MODEL_DIR ...]
"""
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # this repo's mlx_fastpath
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
QF, FF = os.path.join(out_dir, "q4_cache.safetensors"), os.path.join(out_dir, "bf16_cache.safetensors")
GEN, Q_START, GROUP, STEP = 256, 5000, 64, 2048
LENGTHS = [16384, 32768]
if os.environ.get("SMOKE"):
    GEN, LENGTHS = 160, [16384]

files = sorted(glob.glob(os.path.join(text_dir, "**", "*.md"), recursive=True))
assert files, "no .md files"
raw = "\n\n".join(open(p).read() for p in files) * 3


def prompt_ids(tok, n):
    body = tok.decode(tok.encode(raw)[:n])
    return tok.apply_chat_template([{"role": "user", "content": body + "\n\nSummarize the text above in one paragraph."}],
                                   add_generation_prompt=True, tokenize=True)


def prefill(model, ids, kv_bits, path):
    cache = make_prompt_cache(model)
    for i in range(0, len(ids), STEP):
        model(mx.array(ids[i:i + STEP])[None], cache=cache)
        maybe_quantize_kv_cache(cache, Q_START, GROUP, kv_bits)
        mx.eval([c.state for c in cache])
    save_prompt_cache(path, cache)
    del cache
    mx.clear_cache()


def greedy(model, last_id, patched):
    (attention.enable if patched else attention.disable)()
    cache = load_prompt_cache(QF)
    out = [int(t) for t, _ in generate_step(mx.array([last_id]), model, max_tokens=GEN, prompt_cache=cache, kv_bits=4,
                                            kv_group_size=GROUP, quantized_kv_start=Q_START)]
    attention.disable()
    del cache
    mx.clear_cache()
    return out


def forced(model, path, patched, feed):
    """Logits (fp32, last position) after each fed token; one token per call, as in decoding."""
    (attention.enable if patched else attention.disable)()
    cache = load_prompt_cache(path)
    rows = []
    for t in feed:
        logits = model(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(logits)
        rows.append(logits)
    attention.disable()
    del cache
    mx.clear_cache()
    return rows


def kl(ref, x):
    lp, lq = ref - mx.logsumexp(ref), x - mx.logsumexp(x)
    return (mx.exp(lp) * (lp - lq)).sum().item()


def gap(row, a, b):
    return round(row[a].item() - row[b].item(), 3)


def margin(row):
    t = mx.topk(row, 2)
    return round(abs(t[0].item() - t[1].item()), 3)


with open(os.path.join(out_dir, "split_check.jsonl"), "a") as out:
    for mdir in models:
        model, tok = load(mdir)
        for n in LENGTHS:
            ids = prompt_ids(tok, n)
            prefill(model, ids[:-1], 4, QF)
            prefill(model, ids[:-1], None, FF)
            A, B = greedy(model, ids[-1], False), greedy(model, ids[-1], True)
            d = next((i for i, (a, b) in enumerate(zip(A, B)) if a != b), None)
            rec = {"model": os.path.basename(mdir.rstrip("/")), "prompt_tokens": len(ids), "gen": len(A), "split": d}
            if d is not None:
                feed = [ids[-1]] + A[:d]                    # logits at step k predict A[k]
                ref = forced(model, FF, False, feed)
                off = forced(model, QF, False, feed)
                on = forced(model, QF, True, feed)
                a, b = A[d], B[d]
                rec.update({
                    "check_off_argmax_is_A": off[d].argmax().item() == a,
                    "check_on_argmax_is_B": on[d].argmax().item() == b,
                    "tokens_at_split": [tok.decode([a]), tok.decode([b])],
                    "ref_pick": "A (mlx-lm)" if ref[d][a].item() > ref[d][b].item() else "B (fastpath)",
                    "gap_A_minus_B": {"ref": gap(ref[d], a, b), "off": gap(off[d], a, b), "on": gap(on[d], a, b)},
                    "ref_top2_margin_at_split": margin(ref[d]),
                    "mean_kl_vs_ref": {"off": round(sum(kl(r, x) for r, x in zip(ref, off)) / len(ref), 5),
                                       "on": round(sum(kl(r, x) for r, x in zip(ref, on)) / len(ref), 5)},
                    "top1_agree_with_ref": {"off": sum(r.argmax().item() == x.argmax().item() for r, x in zip(ref, off)),
                                            "on": sum(r.argmax().item() == x.argmax().item() for r, x in zip(ref, on)),
                                            "of": len(ref)},
                    "text_A": tok.decode(A[max(0, d - 10):d + 8]), "text_B": tok.decode(B[max(0, d - 10):d + 8])})
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(json.dumps(rec), flush=True)
        del model
        mx.clear_cache()
