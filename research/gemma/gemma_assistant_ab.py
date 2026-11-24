"""Gemma 4 E2B + its -assistant MTP drafter through optiq's own drafter loop: does it beat plain decoding and
fastpath's pipelined plain decode, and is the output plain greedy?

Configs (greedy, GEN tokens, stop at EOS), per weights x prompt, order C1..Cn then reversed (ABBA), median of 2:
  plain      mlx_lm.generate.generate_step (what `optiq serve` runs without --drafter)
  assist-gN  optiq.runtime.spec.assistant_generate_step(gamma=N)  (what `optiq serve --drafter` runs; serve uses N=2)
  fast-ar    mlx_fastpath.generate_ar (pipelined plain decode)
  adaptive   mlx_fastpath.assistant.assistant_generate_step(gamma="auto"): gamma chosen per pass
Speed = decode tok/s after the first token; ttft = time to the first token (includes adaptive's one-time cost
calibration in its first run). Quality = token ids vs plain; at a split, plain's top-2 logprob margin.

    python gemma_assistant_ab.py PROMPTS_JSON DRAFTER_DIR OUT_DIR MODEL_DIR [MODEL_DIR ...]
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # this repo's mlx_fastpath
import mlx.core as mx

import mlx_fastpath as fp
from mlx_fastpath.guard import MemoryGuard

assert Path(fp.__file__).resolve().parents[1] == Path(__file__).resolve().parents[2], fp.__file__

prompts_json, drafter_dir, out_dir, models = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
assert models, "give at least one model dir"
os.makedirs(out_dir, exist_ok=True)
GEN = 256
CONFIGS = [c for c in os.environ.get("CONFIGS", "plain,assist-g1,assist-g2,assist-g3,fast-ar").split(",") if c]
PROMPTS = json.load(open(prompts_json))["prompts"]          # chat, math, code_edit, needle_4k
if os.environ.get("ONLY"):
    PROMPTS = {k: v for k, v in PROMPTS.items() if k in os.environ["ONLY"].split(",")}
if os.environ.get("SMOKE"):
    GEN, CONFIGS, PROMPTS = 32, ["plain", "assist-g2", "adaptive"], {"chat": PROMPTS["chat"]}
assert PROMPTS and CONFIGS


def run(cfg, eng, drafter, ids, eos):
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from optiq.runtime.spec import SpecStats, assistant_generate_step

    from mlx_fastpath.assistant import assistant_generate_step as adaptive_step

    mx.clear_cache()
    t_start = time.perf_counter()
    if cfg == "fast-ar":
        r = fp.generate_ar(eng, ids, max_tokens=GEN)
        return {"tokens": r["tokens"], "tok_s": r["tok_s"]}
    cache = make_prompt_cache(eng._model)
    stats = SpecStats()
    if cfg == "plain":
        steps = generate_step(mx.array(ids), eng._model, max_tokens=GEN, prompt_cache=cache)
    elif cfg == "adaptive":
        steps = adaptive_step(mx.array(ids), eng._model, drafter=drafter, gamma="auto", max_tokens=GEN,
                              prompt_cache=cache, stats=stats)
    else:
        steps = assistant_generate_step(mx.array(ids), eng._model, drafter=drafter, gamma=int(cfg[-1]),
                                        max_tokens=GEN, prompt_cache=cache, stats=stats)
    toks, margins, t_first = [], [], None
    for tok, logprobs in steps:
        tok = int(tok)
        if t_first is None:
            t_first = time.perf_counter()
        if cfg == "plain":
            top = mx.topk(logprobs.reshape(-1).astype(mx.float32), 2)
            margins.append(round(abs(top[0].item() - top[1].item()), 3))
        if tok in eos:
            break
        toks.append(tok)
    t_end = time.perf_counter()
    out = {"tokens": toks, "tok_s": (len(toks) - 1) / (t_end - t_first) if len(toks) > 1 else 0.0,
           "ttft_s": round(t_first - t_start, 3)}
    if cfg == "plain":
        out["margins"] = margins
    else:
        out["accept"] = round(stats.acceptance_rate, 3)
        out["drafted"], out["accepted"] = stats.n_drafted, stats.n_accepted
        ch = getattr(stats, "chooser", None)
        if ch is not None:
            out["gamma_picks"] = dict(sorted(ch.picks.items()))
            out["costs_ms"] = {k: {n: round(v, 2) for n, v in d.items()} for k, d in ch.costs.items()}
            out["p_by_position"] = [round(ch.p(j), 3) for j in range(1, ch.gamma_max + 1)]
    return out


with MemoryGuard(min_free_gb=8.2, max_footprint_gb=7.2) as guard, \
        open(os.path.join(out_dir, "gemma_ab.jsonl"), "a") as f:
    for mdir in models:
        eng = fp.load(mdir, memory_limit_gb=5.2, n2=False)     # load BEFORE importing optiq (lower load peak)
        from optiq.runtime.spec import GemmaAssistantDrafter
        fp.remove()                                             # no fastpath swaps: generate_ar needs none
        tok = eng.tokenizer
        eos = set(tok.eos_token_ids) if getattr(tok, "eos_token_ids", None) else {tok.eos_token_id}
        drafter = GemmaAssistantDrafter.from_pretrained(drafter_dir)
        for pname, ptext in PROMPTS.items():
            ids = tok.apply_chat_template([{"role": "user", "content": ptext}], add_generation_prompt=True,
                                          tokenize=True)
            res = {c: [] for c in CONFIGS}
            for c in CONFIGS + CONFIGS[::-1]:
                res[c].append(run(c, eng, drafter, ids, eos))
            ref = res["plain"][0]
            for c in CONFIGS:
                a = res[c][0]
                split = next((i for i, (x, y) in enumerate(zip(ref["tokens"], a["tokens"])) if x != y), None)
                if split is None and len(ref["tokens"]) != len(a["tokens"]):
                    split = min(len(ref["tokens"]), len(a["tokens"]))
                rec = {"model": os.path.basename(mdir.rstrip("/")), "prompt": pname, "prompt_tokens": len(ids),
                       "config": c, "tok_s": [round(r["tok_s"], 2) for r in res[c]],
                       "tok_s_median": round(statistics.median(r["tok_s"] for r in res[c]), 2),
                       "x_plain": round(statistics.median(r["tok_s"] for r in res[c])
                                        / statistics.median(r["tok_s"] for r in res["plain"]), 3),
                       "gen": len(a["tokens"]), "same_as_plain": split is None, "split": split,
                       "reps_agree": res[c][0]["tokens"] == res[c][1]["tokens"],
                       "ttft_s": [r.get("ttft_s") for r in res[c]]}
                if split is not None and split < len(ref.get("margins", [])):
                    rec["plain_margin_at_split"] = ref["margins"][split]
                if "accept" in a:
                    rec.update(accept=a["accept"], drafted=a["drafted"], accepted=a["accepted"])
                for k in ("gamma_picks", "costs_ms", "p_by_position"):
                    if k in a:
                        rec[k] = a[k]
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(json.dumps(rec), flush=True)
        del eng, drafter
        mx.clear_cache()
    print(json.dumps({"guard": guard.report(), "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2)}), flush=True)
