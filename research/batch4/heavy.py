"""Batch 4 heavy run: H1-H4 in ONE unattended go (research/batch4/plan.md, "Heavy-batch rules").

- Each job runs through research/batch2/run.py's run(): free memory is measured right before the job, the job's
  footprint cap = free - 1 GB, MLX limit = cap - 0.5, MLX cache 0.25 GB, MemoryGuard stops it on the cap or on swap
  growth past the user's 3 GB total. A job that doesn't fit is skipped, not waited for.
- Nothing stops the chain: a skipped / stopped / failed / timed-out job is logged and retried once after the rest.
  An "only if" whose condition is false is logged as "not run (condition)".
- Caps: a timeout per job, and the whole chain stops starting jobs after CHAIN_MIN minutes.
- Every job runs the package from this repo (PYTHONPATH), not the older installed copy.
Usage:  python heavy.py            (the real run; ~45-60 min; log heavy.log, rows h*.jsonl)
        python heavy.py --dry-run  (stub jobs, seconds; log heavy_dry.log; one stub fails once, then is retried)
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent
REPO = R.parent
sys.path.insert(0, str(R / "batch2"))
import run as runner  # noqa: E402

CHAIN_MIN = 90
SHORT = "chat,math,code_edit,needle_4k"
ENV = {"PYTHONPATH": str(REPO)}
FAILED = {"skipped", "STOPPED BY GUARD", "TIMEOUT"}


def jobs():
    J = [runner.job("H1-sustained", HERE, ["h1_sustained.py"], 7.6, dict(ENV), HERE / "h1_sustained.jsonl", 900,
                    {"task": "H1"})]
    order = [("optiq-ar", R / "dflash", ["baseline_child.py", "AR"], 8.7, {"ONLY": SHORT, "MLX_FASTPATH_N2": "0"}),
             ("ar-n2off", HERE, ["h2_packaged.py", "ar"], 8.7, {"N2": "0"}),
             ("ar-n2on", HERE, ["h2_packaged.py", "ar"], 7.8, {"N2": "1"}),
             ("A-n2off", HERE, ["h2_packaged.py", "A"], 8.7, {"N2": "0"}),
             ("A-n2on", HERE, ["h2_packaged.py", "A"], 7.8, {"N2": "1"}),
             ("A-idset-n2on", HERE, ["h2_packaged.py", "A-idset"], 7.9, {"N2": "1"})]
    for i, (name, cwd, argv, need, env) in enumerate(order + order[::-1]):       # ABBA
        J.append(runner.job(f"H2-{name}-{i}", cwd, argv, need, {**ENV, **env}, HERE / "h2_packaged.jsonl", 900,
                            {"task": "H2", "cfg": name, "run": i // len(order)}))
    J.append({**runner.job("H3-slow-tests", REPO, ["-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", "-m", "slow",
                                                   "-s", "tests/test_model_slow.py"], 10.0,
                           {**ENV, "MLX_FASTPATH_SLOW": "1"}, None, 900, {"task": "H3"}),
              "raw": HERE / "h3_slow_tests.out"})            # pytest's own pass/skip lines: exit 0 alone can't tell
    # H4 (E9 model-level prefill) runs only after M1 passed and the int8 kernel exists (both need the user's OK)
    J.append({**runner.job("H4-e9-prefill", HERE, ["h4_e9_prefill.py"], 9.0, dict(ENV), HERE / "h4_e9.jsonl", 900,
                           {"task": "H4"}), "only_if": lambda: (HERE / "M1_PASSED").exists() and (HERE / "h4_e9_prefill.py").exists()})
    return J


def stubbed(J, fail_once):
    """Dry run: the same chain, every job replaced by stub_job.py (prints 2 rows; `fail_once` fails on its first try)."""
    out = []
    for j in J:
        out.append({**j, "cwd": HERE, "argv": ["stub_job.py", j["name"], "1" if j["name"] == fail_once else "0"],
                    "need": 0.5, "out": HERE / "dry" / "rows.jsonl" if j["out"] else None, "timeout": 60,
                    "raw": HERE / "dry" / f"{j['name']}.out" if j.get("raw") else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    J = jobs()
    if a.dry_run:
        (HERE / "dry").mkdir(exist_ok=True)
        J = stubbed(J, fail_once="H2-ar-n2on-2")
    runner.LOG = open(HERE / ("heavy_dry.log" if a.dry_run else "heavy.log"), "a")
    log = runner.log
    t0 = time.time()
    log(f"HEAVY {'DRY RUN ' if a.dry_run else ''}start: {len(J)} jobs, chain cap {CHAIN_MIN} min")
    status, retry = {}, []
    for j in J:
        if "only_if" in j and not j["only_if"]():
            status[j["name"]] = "not run (condition)"
            log(f"NOT RUN {j['name']}: condition false")
            continue
        if time.time() - t0 > CHAIN_MIN * 60:
            status[j["name"]] = "not run (chain cap)"
            log(f"NOT RUN {j['name']}: chain cap reached")
            continue
        s = runner.run(j)
        status[j["name"]] = s
        if s != "ok":
            retry.append(j)
    for j in retry:                                   # one retry each, after the rest
        if time.time() - t0 > CHAIN_MIN * 60:
            log(f"NOT RETRIED {j['name']}: chain cap reached")
            continue
        log(f"RETRY {j['name']} (first try: {status[j['name']]})")
        status[j["name"]] = runner.run(j) + " (retry)"
    counts = {}
    for s in status.values():
        counts[s] = counts.get(s, 0) + 1
    log(f"HEAVY done in {(time.time() - t0) / 60:.1f} min: " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
