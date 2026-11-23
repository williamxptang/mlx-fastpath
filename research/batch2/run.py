"""Batch 2 runner for a busy Mac (user's instruction 2026-11-20: run here, cap each process to the memory free now).

Before every job: measure free memory as MemoryGuard does (free + inactive + speculative), set the job's footprint cap
to free - MARGIN, and skip the job if its expected peak doesn't fit under that cap. The child's MemoryGuard then
stops it (exit 3) if the footprint passes the cap or system swap would pass SWAP_MAX (3 GB total, the user's limit of
2026-11-21; no swap preferred). The cap at free - 1 GB keeps the job itself from causing swap. MLX limit = cap - 0.5 GB, MLX cache 0.25 GB (frees memory early instead of hitting the guard).

Expected peaks: earlier runs' measured peaks (MLX cache 1 GB) minus 0.5 GB for the smaller cache; N2 models minus
their weight savings (gdnhead -0.9 GB, all -1.5 GB; estimates until measured).

Usage: python run.py {kl|dflash|speed|sample|t1|...} [extra]   ->  run.log, rows to the phase's jsonl
"""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from mlx_fastpath.guard import free_gb, swap_used_gb

HERE = Path(__file__).parent
DF = HERE.parent / "dflash"
PY = os.path.expanduser("<venv>/bin/python")
LLM = MODELS + ""
SRC, GDN, ALL, GPTQ, MID = (f"{LLM}/Qwen3.5-9B-OptiQ-4bit{s}" for s in
                             ("", "-N2-gdnhead", "-N2-all", "-N2-all-gptq", "-N2-gdnhead-mlp-gptq"))
SAVE = {SRC: 0.0, GDN: 0.9, ALL: 1.5, GPTQ: 1.5, MID: 1.2}   # GPTQ: N2-all's bytes; MID: estimate
REF = "<staging>/kl_ref/optiq"
MARGIN = 1.0
SWAP_MAX = 3.0   # user, 2026-11-21: system swap may reach 3 GB in total (none preferred)
SHORT = "chat,math,code_edit"
LOG = open(HERE / "run.log", "a")


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def job(name, cwd, argv, need, env=None, out=None, timeout=1500, tags=None):
    return dict(name=name, cwd=cwd, argv=argv, need=need, env=env or {}, out=out, timeout=timeout, tags=tags or {})


def run(j):
    have = free_gb()
    cap = round(have - MARGIN, 2)
    if cap < j["need"]:
        log(f"SKIP {j['name']}: needs ~{j['need']:.1f} GB, cap would be {cap:.1f} GB ({have:.1f} free)")
        return "skipped"
    swap = swap_used_gb()
    tol = round(SWAP_MAX - swap, 2)
    if tol <= 0.0625:
        log(f"SKIP {j['name']}: system swap {swap:.2f} GB, at the {SWAP_MAX:.0f} GB limit")
        return "skipped"
    env = dict(os.environ, GUARD_SWAP_GB=f"{tol:.2f}", GUARD_FREE_GB=f"{j['need'] + MARGIN:.2f}", GUARD_CAP_GB=f"{cap:.2f}",
               MLX_LIMIT_GB=f"{cap - 0.5:.2f}", MLX_CACHE_GB="0.25")
    env.update(j["env"])          # a job may set its own MLX_CACHE_GB (never the guard: no job env names it)
    assert not {"GUARD_FREE_GB", "GUARD_CAP_GB", "GUARD_SWAP_GB"} & set(j["env"]), j["name"]
    log(f"START {j['name']}: free {have:.1f} GB, cap {cap:.1f} GB, expected peak {j['need']:.1f} GB, "
        f"swap {swap:.2f} GB (may grow {tol:.2f})")
    t0 = time.time()
    p = subprocess.run(["taskpolicy", "-b", "nice", "-n", "10", "/opt/homebrew/bin/gtimeout", str(j["timeout"]), PY, *j["argv"]],
                       cwd=j["cwd"], env=env, capture_output=True, text=True)
    if j.get("raw"):                                  # keep the whole output (e.g. pytest's), not only JSON rows
        Path(j["raw"]).write_text(f"$ {' '.join(map(str, j['argv']))}\n--- stdout\n{p.stdout}\n--- stderr\n{p.stderr}")
    rows = []
    for line in p.stdout.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and "prompt" in r:
            rows.append({**r, **j["tags"], "job": j["name"], "cap_gb": cap})
        elif isinstance(r, dict) and "guard" in r:          # a child's final guard report (whole job)
            rows.append({"peak_footprint_gb": r["guard"].get("peak_footprint_gb"),
                         "swap_growth_gb": r["guard"].get("swap_growth_gb"), "_guard_only": True})
    peaks = [r.get("peak_footprint_gb") for r in rows if r.get("peak_footprint_gb")]
    swaps = [r.get("swap_growth_gb") for r in rows if r.get("swap_growth_gb") is not None]
    rows = [r for r in rows if not r.get("_guard_only")]
    if j["out"] and rows:
        with open(HERE / j["out"], "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    err = [l for l in p.stderr.splitlines() if "MemoryGuard" in l or "Error" in l or "Traceback" in l]
    status = {0: "ok", 3: "STOPPED BY GUARD", 124: "TIMEOUT"}.get(p.returncode, f"exit {p.returncode}")
    log(f"END {j['name']}: {status}, {len(rows)} rows, {time.time() - t0:.0f}s, peak {max(peaks) if peaks else '?'} GB, "
        f"swap growth {max(swaps) if swaps else '?'} GB"
        + (f" | {' | '.join(err[-3:])}" if err else ""))
    if p.returncode != 0:
        (HERE / "run_errors.log").open("a").write(f"--- {j['name']} {time.ctime()}\n{p.stderr[-4000:]}\n")
    return status


def phase(name, extra=None):
    J = []
    if name == "kl":            # R2-R4 on the short prompts (source at 4k/16k needs 9-10 GB)
        e = {"ONLY": SHORT}
        J += [job("kl-ref-source", HERE, ["kl_gate.py", "ref", SRC, REF], 8.1, e),
              job("kl-control-source", HERE, ["kl_gate.py", "score", SRC, REF, "control-c64", "64"], 8.1, e),
              job("kl-n2-gdnhead", HERE, ["kl_gate.py", "score", GDN, REF, "n2-gdnhead", "64"], 8.1 - SAVE[GDN], e),
              job("kl-n2-all", HERE, ["kl_gate.py", "score", ALL, REF, "n2-all", "64"], 8.1 - SAVE[ALL], e)]
    elif name == "dflash":      # R1, ABBA, short prompts (4k+ peaked 10-11 GB)
        for i, v in enumerate(("plain", "idset", "idset", "plain")):
            env = {"ONLY": SHORT, **({"VOCAB": "32768", "VOCAB_PROMPT": "1"} if v == "idset" else {})}
            J.append(job(f"dflash-{v}-{i}", DF, ["dflash_child.py", f"D2-w4-{v}", "w4:gs64", "adaptive", "0"], 8.3,
                         env, "r1_dflash.jsonl", tags={"variant": v, "run": i}))
    elif name == "speed":       # R5: models x methods, ABBA
        order = [(m, meth) for m in (SRC, GDN, ALL) for meth in ("AR", "A")]
        for i, (m, meth) in enumerate(order + [(m, meth) for m, meth in reversed(order)]):
            J.append(job(f"speed-{meth}-{Path(m).name}-{i}", DF, ["baseline_child.py", meth], 8.1 - SAVE[m],
                         {"ONLY": SHORT, "MODEL": m}, "r5_speed.jsonl", tags={"run": i // len(order)}))
        for m in (GDN, ALL):    # tasks: needle codes at 4k, the whole code file at 800 tokens
            J.append(job(f"task-needle4k-{Path(m).name}", DF, ["baseline_child.py", "AR"], 9.2 - SAVE[m],
                         {"ONLY": "needle_4k", "MODEL": m}, "r5_tasks.jsonl"))
            J.append(job(f"task-code800-{Path(m).name}", DF, ["baseline_child.py", "AR"], 8.4 - SAVE[m],
                         {"PROMPTS": "prompts_c.json", "ONLY": "code_edit", "MAX": "800", "MODEL": m}, "r5_tasks.jsonl"))
    elif name == "kl_long":     # R2-R4 at 4k and 16k (own reference folder: ref.json is per folder)
        e, ref = {"ONLY": "needle_4k,needle_16k"}, REF + "_long"
        J += [job("kl-ref-source-long", HERE, ["kl_gate.py", "ref", SRC, ref], 9.2, e),
              job("kl-control-source-long", HERE, ["kl_gate.py", "score", SRC, ref, "control-c64-long", "64"], 9.2, e),
              job("kl-n2-gdnhead-long", HERE, ["kl_gate.py", "score", GDN, ref, "n2-gdnhead-long", "64"], 9.2 - SAVE[GDN], e),
              job("kl-n2-all-long", HERE, ["kl_gate.py", "score", ALL, ref, "n2-all-long", "64"], 9.2 - SAVE[ALL], e)]
    elif name == "dflash_long":  # R1 at 4k and 16k (peaked 10-11 GB with a 1 GB MLX cache)
        for i, v in enumerate(("plain", "idset", "idset", "plain")):
            env = {"ONLY": "needle_4k,needle_16k", **({"VOCAB": "32768", "VOCAB_PROMPT": "1"} if v == "idset" else {})}
            J.append(job(f"dflash-long-{v}-{i}", DF, ["dflash_child.py", f"D2-w4-{v}", "w4:gs64", "adaptive", "0"], 10.5,
                         env, "r1_dflash.jsonl", tags={"variant": v, "run": i}))
    elif name == "tie":         # where plain and id-set DFlash texts split: near-tie or not
        J.append(job("tie-check", HERE, ["tie_check.py"], 8.1, out="tie_check.jsonl"))
    elif name == "tie_long":    # batch 3 T3: the same check at 4k/16k (AR at 16k peaked 9.4 GB on the source)
        J.append(job("tie-check-long", HERE, ["tie_check.py"], 9.4, {"ONLY": "chat,math,needle_4k,needle_16k"}, "tie_check.jsonl"))
    elif name == "task16k":     # needle codes at 16k (AR peaked 9.4 GB on the source)
        for m in (GDN, ALL):
            J.append(job(f"task-needle16k-{Path(m).name}", DF, ["baseline_child.py", "AR"], 9.0 - SAVE[m],
                         {"ONLY": "needle_16k", "MODEL": m}, "r5_tasks.jsonl"))
    elif name == "cache":       # does the 0.25 GB MLX cache (set for the cap) slow plain decoding vs 1 GB?
        for i, c in enumerate(("0.25", "1", "1", "0.25")):
            J.append(job(f"cache-{c}-{i}", DF, ["baseline_child.py", "AR"], 8.6, {"ONLY": "chat,math", "MLX_CACHE_GB": c},
                         "r5_cache.jsonl", tags={"cache_gb": float(c), "run": i}))
    elif name == "sample":      # R6: temperature 0.7, seeds 0-2, source and one N2 model (argv extra)
        models = [SRC] + ([f"{LLM}/{extra}"] if extra else [])
        for seed in (0, 1, 2):
            for m in models:
                for meth in ("AR", "A"):
                    J.append(job(f"sample-{meth}-{Path(m).name}-s{seed}", DF, ["baseline_child.py", meth], 8.1 - SAVE.get(m, 0),
                                 {"ONLY": SHORT, "MODEL": m, "TEMP": "0.7", "SEED": str(seed)}, "r6_sample.jsonl",
                                 tags={"seed": seed}))
    elif name == "t1":          # batch 3 T1: needle codes on 3 documents per length (10k-24k), AR, one job per model
        for m, need in ((GDN, 8.4), (ALL, 7.8), (SRC, 9.3)):
            J.append(job(f"t1-needle-{Path(m).name}", DF, ["baseline_child.py", "AR"], need,
                         {"PROMPTS": "../batch2/prompts_var.json", "MODEL": m, **({"ONLY": extra} if extra else {})},
                         "../batch3/t1_needle.jsonl", timeout=2400))
    elif name == "t4":          # batch 3 T4 (R7): crossover repeats on 3 documents per length, ABBA per pair
        V = "../batch2/prompts_var.json"
        near = ",".join(f"needle_{k}_v{v}" for k in ("10k", "12k") for v in (1, 2, 3))
        far = ",".join(f"needle_{k}_v{v}" for k in ("20k", "24k") for v in (1, 2, 3))
        dfl = lambda i: job(f"t4-D2-{i}", DF, ["dflash_child.py", "D2-w4-adaptive", "w4:gs64", "adaptive", "0"], 10.0,
                            {"PROMPTS": V, "ONLY": near}, "../batch3/t4_cross.jsonl", 2400, {"pair": "D2-vs-A", "run": i})
        base = lambda meth, only, pair, need, i: job(f"t4-{meth}-{pair}-{i}", DF, ["baseline_child.py", meth], need,
                                                     {"PROMPTS": V, "ONLY": only}, "../batch3/t4_cross.jsonl", 2400,
                                                     {"pair": pair, "run": i})
        J += [dfl(0), base("A", near, "D2-vs-A", 9.5, 0), base("A", near, "D2-vs-A", 9.5, 1), dfl(1)]
        J += [base(m, far, "A-vs-Aq4", 10.0, i) for i, m in enumerate(("A", "Aq4", "Aq4", "A"))]
    elif name == "t7":          # batch 3 T7: N2-all's layer set with GPTQ instead of rounding; the checks N2-all failed
        need = lambda base: base - SAVE[GPTQ]
        J += [job("t7-kl", HERE, ["kl_gate.py", "score", GPTQ, REF, "n2-all-gptq", "64"], need(8.1), {"ONLY": SHORT}),
              job("t7-kl-long", HERE, ["kl_gate.py", "score", GPTQ, REF + "_long", "n2-all-gptq-long", "64"], need(9.2),
                  {"ONLY": "needle_4k,needle_16k"}),
              job("t7-needle4k", DF, ["baseline_child.py", "AR"], need(9.2), {"ONLY": "needle_4k", "MODEL": GPTQ},
                  "../batch3/t7_tasks.jsonl"),
              job("t7-needle16k", DF, ["baseline_child.py", "AR"], need(9.0), {"ONLY": "needle_16k", "MODEL": GPTQ},
                  "../batch3/t7_tasks.jsonl"),
              job("t7-code800", DF, ["baseline_child.py", "AR"], need(8.4),
                  {"PROMPTS": "prompts_c.json", "ONLY": "code_edit", "MAX": "800", "MODEL": GPTQ}, "../batch3/t7_tasks.jsonl"),
              job("t7-needle-var", DF, ["baseline_child.py", "AR"], 7.9,
                  {"PROMPTS": "../batch2/prompts_var.json", "MODEL": GPTQ}, "../batch3/t1_needle.jsonl", timeout=2400)]
        for i, (m, meth) in enumerate([(GPTQ, "AR"), (ALL, "AR"), (ALL, "A"), (GPTQ, "A")]):   # speed = N2-all's?
            J.append(job(f"t7-speed-{meth}-{Path(m).name}-{i}", DF, ["baseline_child.py", meth], need(8.1),
                         {"ONLY": SHORT, "MODEL": m}, "../batch3/t7_speed.jsonl", tags={"run": i}))
    elif name == "t2":          # batch 3 T2: GDN + head + MLP at 4-bit with GPTQ, full attention kept 8-bit
        need = lambda base: base - SAVE[MID]
        J += [job("t2-kl", HERE, ["kl_gate.py", "score", MID, REF, "n2-mid-gptq", "64"], need(8.1), {"ONLY": SHORT}),
              job("t2-kl-long", HERE, ["kl_gate.py", "score", MID, REF + "_long", "n2-mid-gptq-long", "64"], need(9.2),
                  {"ONLY": "needle_4k,needle_16k"}),
              job("t2-needle4k", DF, ["baseline_child.py", "AR"], need(9.2), {"ONLY": "needle_4k", "MODEL": MID},
                  "../batch3/t2_tasks.jsonl"),
              job("t2-needle16k", DF, ["baseline_child.py", "AR"], need(9.0), {"ONLY": "needle_16k", "MODEL": MID},
                  "../batch3/t2_tasks.jsonl"),
              job("t2-code800", DF, ["baseline_child.py", "AR"], need(8.4),
                  {"PROMPTS": "prompts_c.json", "ONLY": "code_edit", "MAX": "800", "MODEL": MID}, "../batch3/t2_tasks.jsonl"),
              job("t2-needle-var", DF, ["baseline_child.py", "AR"], 8.2,
                  {"PROMPTS": "../batch2/prompts_var.json", "MODEL": MID}, "../batch3/t1_needle.jsonl", timeout=2400)]
        order = [(m, meth) for m in (MID, GDN, SRC) for meth in ("AR", "A")]
        for i, (m, meth) in enumerate(order + order[::-1]):                  # ABBA against N2-gdnhead and the source
            J.append(job(f"t2-speed-{meth}-{Path(m).name}-{i}", DF, ["baseline_child.py", meth], 8.1 - SAVE[m],
                         {"ONLY": SHORT, "MODEL": m}, "../batch3/t2_speed.jsonl", tags={"run": i // len(order)}))
    elif name == "first":       # batch 3: first-token margin on the needle prompts (16k/24k prefill; AR 16k peaked 9.4 GB)
        for m in (SRC, GDN, MID, GPTQ, ALL):
            J.append(job(f"first-{Path(m).name}", HERE.parent / "batch3", ["first_token.py", m], 9.6 - SAVE[m], {},
                         "../batch3/first_token.jsonl", timeout=900))
    elif name == "branch":      # batch 3: where each model leaves the source's needle_16k answer (16k prefill)
        for m in (SRC, GDN, MID, GPTQ, ALL):
            J.append(job(f"branch-{Path(m).name}", HERE.parent / "batch3", ["branch_16k.py", m], 9.6 - SAVE[m], {},
                         "../batch3/branch_16k.jsonl", timeout=900))
    else:
        raise SystemExit(__doc__)
    assert J, "no jobs"
    log(f"PHASE {name}: {len(J)} jobs")
    res = [run(j) for j in J]
    log(f"PHASE {name} done: " + ", ".join(f"{s} x{res.count(s)}" for s in sorted(set(res))))


if __name__ == "__main__":
    phase(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
