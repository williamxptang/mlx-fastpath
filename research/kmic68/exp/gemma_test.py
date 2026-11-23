"""Gemma-4-12B-OptiQ-4bit + its MTP assistant drafter (mlx-vlm 0.6.13): decode tok/s vs plain, at several
contexts and draft block sizes, with an audit (hidden codes, repetition, divergence from plain, saved texts).
Caps (2026-11-19, Mac cleared): MLX limit 12 GB, watchdog kill > 14 GB footprint, abort if swap grows > 0.5 GB.
usage: gemma_test.py smoke | full"""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import sys, os, time, json, threading, subprocess, re, glob
import mlx.core as mx
mx.set_memory_limit(12 * 2**30); mx.set_cache_limit(2 * 2**30)


def swap_used_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(re.search(r"used = ([\d.]+)M", out).group(1)) / 1024


SWAP0 = swap_used_gb()


def watchdog():
    pid = os.getpid()
    while True:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True).stdout
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*(GB|MB|KB)", out)
        if m:
            gb = float(m.group(1)) * {"GB": 1, "MB": 1 / 1024, "KB": 1 / 2**20}[m.group(2)]
            watchdog.peak = max(watchdog.peak, gb)
            if gb > 14.0:
                print(f"WATCHDOG: footprint {gb:.1f} GB > 14 GB, exiting", flush=True); os._exit(3)
        dswap = swap_used_gb() - SWAP0
        watchdog.swap = max(watchdog.swap, dswap)
        if dswap > 0.5:
            print(f"WATCHDOG: swap grew {dswap:.2f} GB, exiting", flush=True); os._exit(4)
        time.sleep(1)


watchdog.peak = 0.0
watchdog.swap = 0.0
threading.Thread(target=watchdog, daemon=True).start()

from mlx_vlm import load, generate
from mlx_vlm.speculative.drafters import load_drafter

T = MODELS + "/gemma-4-12B-it-qat-OptiQ-4bit"
D = MODELS + "/gemma-4-12B-it-qat-assistant-4bit"
model, proc = load(T)
print(json.dumps({"stage": "target loaded", "footprint_peak_gb": round(watchdog.peak, 2), "active_gb": round(mx.get_active_memory() / 2**30, 2)}), flush=True)
draft, kind = load_drafter(D, kind="mtp")
print(json.dumps({"stage": "drafter loaded", "kind": kind, "footprint_peak_gb": round(watchdog.peak, 2), "active_gb": round(mx.get_active_memory() / 2**30, 2)}), flush=True)
tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc

CODES = ["KESTREL-4471", "MAGENTA-0928", "OBSIDIAN-3365"]
TASK = ("\n\nTask 1: The document above contains exactly three secret codes (each written as 'The secret code "
        "number N is ...'). List all three codes exactly.\nTask 2: Then summarize the document in 5 bullet points.")
docs = sorted(glob.glob("<staging>/research/261118-transferable-ideas/**/*.md", recursive=True))
corpus = tok.encode("\n\n".join(open(p).read() for p in docs) * 3, add_special_tokens=False)


def make_prompt(n):
    if n == 0:
        body = "Explain, step by step, how a CPU cache works and why cache lines matter for performance."
    else:
        b = list(corpus[:n])
        for i, (frac, code) in enumerate(zip((0.1, 0.5, 0.9), CODES)):
            at = int(len(b) * frac)
            b[at:at] = tok.encode(f"\n\nThe secret code number {i + 1} is {code}.\n\n", add_special_tokens=False)
        body = tok.decode(b) + TASK
    return tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True)


def repetition(g):
    grams = [tuple(g[i:i + 4]) for i in range(len(g) - 3)]
    longest = 0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            k = 0
            while j + k < len(g) and g[i + k] == g[j + k]: k += 1
            longest = max(longest, k)
    return round(len(set(grams)) / max(1, len(grams)), 3), longest


MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
CTX = (0,) if MODE == "smoke" else (0, 4000, 16000, 32000)
BLOCKS = (None, 3) if MODE == "smoke" else (None, 2, 3, 4, 6)
GEN = 120 if MODE == "smoke" else 200
texts = {}
for n in CTX:
    prompt = make_prompt(n)
    ref = None
    for bs in ("AR",) + BLOCKS:
        kw = dict(max_tokens=GEN, temperature=0.0, prefill_step_size=512, verbose=False)
        if bs != "AR":
            kw.update(draft_model=draft, draft_kind=kind, draft_block_size=bs)
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        r = generate(model, proc, prompt, **kw)
        wall = time.perf_counter() - t0
        prefill_s = r.prompt_tokens / r.prompt_tps if r.prompt_tps else float("nan")
        g = tok.encode(r.text, add_special_tokens=False)
        name = "AR" if bs == "AR" else f"MTP block={bs or 'default'}"
        if bs == "AR": ref = r.text
        split = next((i for i, (a, b) in enumerate(zip(r.text, ref)) if a != b), min(len(r.text), len(ref)))
        d4, rep = repetition(g)
        print(json.dumps({"prompt_tokens": r.prompt_tokens, "config": name, "tok_s": round(r.generation_tps, 2),
                          "wall_tok_s_excl_prefill": round(r.generation_tokens / max(1e-9, wall - prefill_s), 2),
                          "tokens": r.generation_tokens, "codes_found": f"{sum(c in r.text for c in CODES)}/3" if n else None,
                          "distinct_4gram": d4, "longest_repeat": rep, "same_as_AR": r.text == ref,
                          "common_prefix_chars": split, "of": len(ref), "prefill_s": round(prefill_s, 1),
                          "mlx_peak_gb": round(r.peak_memory, 2), "footprint_peak_gb": round(watchdog.peak, 2),
                          "swap_growth_gb": round(watchdog.swap, 2)}), flush=True)
        texts[f"{r.prompt_tokens} | {name}"] = r.text
        json.dump(texts, open(f"gemma_texts_{MODE}.json", "w"), ensure_ascii=False, indent=1)
