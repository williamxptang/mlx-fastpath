"""B2: command-buffer limits (E1 settings) on Qwen3.5-9B-OptiQ-4bit, AR and setup A, chat + needle_4k, 300 tokens.
One child (research/dflash/baseline_child.py, guarded 14/14) per setting x cfg x run; settings in ABBA order (S then
reversed S), and AR/A order flipped on the second pass. Rows to b2.jsonl; summary + audit printed at the end.
Stops at the first child that fails (e.g. the guard refuses), never retries."""
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DF = HERE.parent / "dflash"
PY = os.path.expanduser("<venv>/bin/python")
SETTINGS = [(None, None), ("10", "10"), ("100", None), (None, "200"), ("100", "200"), ("300", "1000")]
CODES = json.load(open(DF / "prompts.json"))["codes"]
OUT = HERE / "b2.jsonl"


def child(s, cfg, run):
    env = dict(os.environ, PROMPTS=str(DF / "prompts.json"), ONLY="chat,needle_4k", MAX="300")
    env.pop("MLX_MAX_OPS_PER_BUFFER", None); env.pop("MLX_MAX_MB_PER_BUFFER", None)
    if s[0]: env["MLX_MAX_OPS_PER_BUFFER"] = s[0]
    if s[1]: env["MLX_MAX_MB_PER_BUFFER"] = s[1]
    p = subprocess.run(["taskpolicy", "-b", "nice", "-n", "10", "/opt/homebrew/bin/gtimeout", "400", PY,
                        str(DF / "baseline_child.py"), cfg], env=env, cwd=HERE, capture_output=True, text=True)
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode or len(rows) != 2:
        sys.exit(f"child failed: setting {s} cfg {cfg} run {run} exit {p.returncode}\n{p.stderr[-2000:]}")
    with open(OUT, "a") as f:
        for r in rows:
            r |= {"ops": s[0] or "default", "mb": s[1] or "default", "run": run}
            f.write(json.dumps(r) + "\n"); print(json.dumps({k: v for k, v in r.items() if k != "text"}), flush=True)


def grams4(t):
    w = t.split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return round(len(set(g)) / max(1, len(g)), 3)


if __name__ == "__main__":
    if "--summary" not in sys.argv:
        for run, order in ((1, SETTINGS), (2, SETTINGS[::-1])):
            for s in order:
                for cfg in (("AR", "A") if run == 1 else ("A", "AR")):
                    child(s, cfg, run)
    rows = [json.loads(l) for l in open(OUT)]
    assert rows, "no rows"
    by = defaultdict(list)
    for r in rows:
        by[(r["method"], r["prompt"], r["ops"], r["mb"])].append(r)
    for (m, p, o, mb), v in sorted(by.items()):
        base = by[(m, p, "default", "default")]
        bmed = statistics.median(x["tok_s"] for x in base)
        med = statistics.median(x["tok_s"] for x in v)
        ref = base[0]["text"]
        print(json.dumps({"method": m, "prompt": p, "ops": o, "mb": mb, "n": len(v), "tok_s_med": med,
                          "range": [min(x["tok_s"] for x in v), max(x["tok_s"] for x in v)],
                          "vs_default": round(med / bmed, 3),
                          "base_range": [min(x["tok_s"] for x in base), max(x["tok_s"] for x in base)],
                          "texts_identical_to_default": all(x["text"] == ref for x in v),
                          "tokens": sorted({x["tokens"] for x in v}),
                          "wall_over_runtime": [round(x["wall_tok_s"] / x["tok_s"], 3) for x in v],
                          "codes": [sum(c in x["text"] for c in CODES) for x in v] if p.startswith("needle") else None,
                          "distinct_4grams": [grams4(x["text"]) for x in v],
                          "peak_fp_gb": max(x["peak_footprint_gb"] for x in v), "swap_growth_gb": max(x["swap_growth_gb"] for x in v)}))
