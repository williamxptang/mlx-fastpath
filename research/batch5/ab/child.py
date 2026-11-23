"""One A/B configuration per process: argv MODEL_PATH RUNTIME. Prints one JSON row per prompt (+ one for the QA set)
and a final {"guard": ...} row, as research/batch2/run.py's runner expects. Greedy, thinking off, one warm-up first.
RUNTIME: stock | optiq-mtp | fast-ar | fast-mtp | fast-ngram  (fast-mtp on an N2 copy = pass the N2 path).
Env from the runner: GUARD_FREE_GB GUARD_CAP_GB GUARD_SWAP_GB MLX_LIMIT_GB MLX_CACHE_GB."""
import json
import os
import re
import sys
import time
from pathlib import Path

from mlx_fastpath.guard import MemoryGuard

MODEL, RT = sys.argv[1], sys.argv[2]
D = json.load(open(Path(__file__).parent / "prompts_text.json"))
MAX, QA_MAX = 300, 32


def emit(row):
    print(json.dumps(row), flush=True)


def score(name, text):
    t = text.lower()
    if name == "needle_4k":
        return sum(c.lower() in t for c in D["codes"])
    if name == "math":
        boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
        return int(bool(boxed) and set(re.findall(r"-?\d+", boxed[-1])) == set(D["math_answer"]))
    if name == "code_edit":
        code = re.search(r"```(?:python)?\n(.*?)(?:```|$)", text, re.S)
        src = code.group(1) if code else text
        try:
            compile(src, "edit.py", "exec"); ok = 1
        except SyntaxError:
            ok = 0
        return {"compiles": ok, "edit_modules": int("mlx_lm.models.qwen3_5_moe" in src),
                "edit_max_verify": int(bool(re.search(r"MAX_VERIFY\s*=\s*12\b", src)))}
    return None


def loopiness(ids):
    grams = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return round(len(set(grams)) / len(grams), 3) if grams else None


with MemoryGuard(min_free_gb=float(os.environ["GUARD_FREE_GB"]), max_footprint_gb=float(os.environ["GUARD_CAP_GB"]),
                 swap_tolerance_gb=float(os.environ["GUARD_SWAP_GB"])) as g:
    import mlx.core as mx
    mx.set_memory_limit(int(float(os.environ["MLX_LIMIT_GB"]) * 2**30))
    mx.set_cache_limit(int(float(os.environ["MLX_CACHE_GB"]) * 2**30))
    if RT == "stock":
        from mlx_lm import load, stream_generate
        model, tok = load(MODEL)
        eos = set(tok.eos_token_ids) if hasattr(tok, "eos_token_ids") else {tok.eos_token_id}

        def run(ids, n):
            out, last = [], None
            for r in stream_generate(model, tok, ids, max_tokens=n):
                out.append(int(r.token)); last = r
            return out, {"decode_tok_s": round(last.generation_tps, 2), "prefill_tok_s": round(last.prompt_tps, 1)}
    else:
        import mlx_fastpath as fp
        eng = fp.load(MODEL, memory_limit_gb=float(os.environ["MLX_LIMIT_GB"]),
                      cache_limit_gb=float(os.environ["MLX_CACHE_GB"]), n2=False)
        tok = eng.tokenizer
        eos = set(eng._eos_ids())
        if RT == "optiq-mtp":
            fp.remove()
        else:
            assert fp.apply(), "apply() ran stock"

        def run(ids, n):
            if RT in ("optiq-mtp", "fast-mtp"):
                st = (eng.generate("", prompt_ids=list(ids), max_tokens=n, depth=2) if RT == "optiq-mtp"
                      else fp.generate(eng, ids, max_tokens=n))
                return tok.encode(st.text, add_special_tokens=False), {
                    "decode_tok_s": round(st.decode_tok_s, 2), "accept": round(st.acceptance_rate, 3),
                    "prefill_s": round(st.prefill_time_s, 3)}
            if RT == "fast-ar":
                r = fp.generate_ar(eng, ids, max_tokens=n)
                return list(r["tokens"]), {"decode_tok_s": round(r["tok_s"], 2)}
            if RT == "fast-ngram":
                return list(fp.generate_ngram(eng, ids, max_tokens=n)), {}
            raise SystemExit(f"unknown runtime {RT}")

    def ids_of(text):
        r = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True,
                                    enable_thinking=False)
        return list(r["input_ids"] if hasattr(r, "keys") else r)

    def timed(ids, n):
        t0 = time.perf_counter()
        out, extra = run(ids, n)
        wall = time.perf_counter() - t0
        out = [t for t in out if t not in eos]
        return out, wall, extra

    timed(ids_of("Say hello."), 16)                                    # warm-up (compiles, cost calibration)
    for name, text in D["prompts"].items():
        ids = ids_of(text)
        out, wall, extra = timed(ids, MAX)
        txt = tok.decode(out, skip_special_tokens=True)
        emit({"prompt": name, "prompt_tokens": len(ids), "new_tokens": len(out), "wall_s": round(wall, 3),
              "e2e_tok_s": round(len(out) / wall, 2), **extra, "score": score(name, txt), "unique_4gram": loopiness(out),
              "ids": out, "text": txt})
    qa, wall_all, toks_all = [], 0.0, 0
    for q, ans in D["qa"]:
        out, wall, _ = timed(ids_of(q), QA_MAX)
        txt = tok.decode(out, skip_special_tokens=True)
        qa.append({"q": q, "answer": ans, "text": txt, "ok": int(bool(re.search(rf"(?<![\w-]){re.escape(ans)}(?![\w])", txt.lower())))})
        wall_all += wall; toks_all += len(out)
    emit({"prompt": "qa", "score": sum(r["ok"] for r in qa), "of": len(qa), "new_tokens": toks_all,
          "wall_s": round(wall_all, 3), "e2e_tok_s": round(toks_all / wall_all, 2), "items": qa})
    emit({"guard": g.report(), "mlx_peak_gb": round(mx.get_peak_memory() / 2**30, 2)})
