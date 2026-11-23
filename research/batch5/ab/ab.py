"""A/B runner (plan.md): every model x weights x runtime config as its own process through research/batch2/run.py's
run() (free-memory cap, MemoryGuard, skip if it doesn't fit, no swap). Per model, configs run in ABBA order (each twice).
A job that is skipped / stopped / failed is retried once after the rest; the chain stops starting jobs after CHAIN_MIN.
Usage: python ab.py              (everything; log ab.log, rows ab.jsonl)
       python ab.py --only TEXT  (only jobs whose name contains TEXT, once, no ABBA: a smoke test; rows ab_smoke.jsonl)
       python ab.py --list
"""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "research" / "batch2"))
import run as runner  # noqa: E402

LLM = MODELS + ""
CHAIN_MIN = 180
ENV = {"PYTHONPATH": str(REPO), "HF_HUB_OFFLINE": "1"}
Q_ALL = ["stock", "optiq-mtp", "fast-ar", "fast-mtp", "fast-ngram"]
MODELS = {   # model: [(weights label, path, expected peak GB, runtimes)]
    "q08": [("bf16", f"{LLM}/Qwen3.5-0.8B-bf16", 3.0, ["stock", "fast-ar"]),
            ("int4", f"{LLM}/Qwen3.5-0.8B-OptiQ-4bit/uniform_4bit", 2.0, Q_ALL),
            ("optiq", f"{LLM}/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed", 2.3, Q_ALL),
            ("optiq-n2", f"{LLM}/Qwen3.5-0.8B-OptiQ-4bit/optiq_mixed-N2-gdnhead", 2.2, ["fast-mtp"])],
    "q2b": [("bf16", f"{LLM}/Qwen3.5-2B-bf16", 5.8, ["stock", "fast-ar"]),
            ("int4", f"{LLM}/Qwen3.5-2B-4bit", 3.0, Q_ALL),
            ("optiq", f"{LLM}/Qwen3.5-2B-OptiQ-4bit", 3.3, Q_ALL)],
    "gemma": [("bf16", f"{LLM}/gemma-4-E2B-it-bf16", 10.5, ["stock", "fast-ar"]),
              ("int4", f"{LLM}/gemma-4-e2b-it-4bit", 4.0, ["stock", "fast-ar"]),
              ("optiq", f"{LLM}/gemma-4-e2b-it-OptiQ-4bit", 5.6, ["stock", "fast-ar"])],
}


def jobs(abba=True, out=HERE / "ab.jsonl"):
    J = []
    for model, rows in MODELS.items():
        order = [(w, p, need, rt) for w, p, need, rts in rows for rt in rts]
        seq = order + order[::-1] if abba else order
        for i, (w, p, need, rt) in enumerate(seq):
            rep = i // len(order)
            J.append(runner.job(f"{model}-{w}-{rt}-{rep}", HERE, ["child.py", p, rt], need, dict(ENV), out, 900,
                                {"model": model, "weights": w, "runtime": rt, "rep": rep}))
    return J


REFDIR = Path("<tmp>")   # bf16 reference log-probs (0.5 GB each), not kept


def kl_jobs():
    """Per model: the bf16 reference, then each quantized weights scored against it (kl.py), once each."""
    REFDIR.mkdir(parents=True, exist_ok=True)
    J = []
    for model, rows in MODELS.items():
        ref = str(REFDIR / f"{model}.npy")
        (_, bf16, need, _), *quant = rows
        J.append(runner.job(f"kl-{model}-bf16-ref", HERE, ["kl.py", "ref", bf16, ref], need + 1.5, dict(ENV), None, 900))
        for w, p, need_q, _ in quant:
            J.append(runner.job(f"kl-{model}-{w}", HERE, ["kl.py", "score", p, ref, w], need_q + 2.5, dict(ENV),
                                HERE / "kl.jsonl", 900, {"model": model, "weights": w}))
    return J


def n2_2b_jobs():
    """The 2B N2 rows, added after n2's missing-sidecar fix (batch5 results): in one ABBA session, 2B OptiQ stock /
    fast-ar / fast-mtp and N2 fast-ar / fast-mtp, each twice (rows ab_n2_2b.jsonl), then KL of N2 vs bf16 against the
    2B bf16 reference the A/B already wrote (kl_n2_2b.jsonl)."""
    opt, n2 = f"{LLM}/Qwen3.5-2B-OptiQ-4bit", f"{LLM}/Qwen3.5-2B-OptiQ-4bit-N2-gdnhead"
    order = [("optiq", opt, 3.3, "stock"), ("optiq", opt, 3.3, "fast-ar"), ("optiq", opt, 3.3, "fast-mtp"),
             ("optiq-n2", n2, 3.2, "fast-ar"), ("optiq-n2", n2, 3.2, "fast-mtp")]
    J = []
    for i, (w, p, need, rt) in enumerate(order + order[::-1]):
        rep = i // len(order)
        J.append(runner.job(f"q2b-{w}-{rt}-{rep}", HERE, ["child.py", p, rt], need, dict(ENV), HERE / "ab_n2_2b.jsonl",
                            900, {"model": "q2b", "weights": w, "runtime": rt, "rep": rep}))
    ref = REFDIR / "q2b.npy"
    assert ref.exists(), f"{ref} missing: run `ab.py --kl` first"
    J.append(runner.job("kl-q2b-optiq-n2", HERE, ["kl.py", "score", n2, str(ref), "optiq-n2"], 5.7, dict(ENV),
                        HERE / "kl_n2_2b.jsonl", 900, {"model": "q2b", "weights": "optiq-n2"}))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--kl", action="store_true")
    ap.add_argument("--n2-2b", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.n2_2b:
        J = n2_2b_jobs()
    elif a.kl:
        J = kl_jobs()
    elif a.only:
        J = [j for j in jobs(abba=False, out=HERE / "ab_smoke.jsonl") if a.only in j["name"]]
    else:
        J = jobs()
    if a.list:
        for j in J:
            print(j["name"], j["need"], j["argv"][1])
        return 0
    runner.LOG = open(HERE / ("ab_n2_2b.log" if a.n2_2b else "kl.log" if a.kl else "ab_smoke.log" if a.only else "ab.log"), "a")
    log, t0, status, retry = runner.log, time.time(), {}, []
    log(f"AB start: {len(J)} jobs, chain cap {CHAIN_MIN} min")
    for j in J:
        if time.time() - t0 > CHAIN_MIN * 60:
            status[j["name"]] = "not run (chain cap)"; log(f"NOT RUN {j['name']}: chain cap"); continue
        status[j["name"]] = s = runner.run(j)
        if s != "ok":
            retry.append(j)
    for j in retry:
        if time.time() - t0 > CHAIN_MIN * 60:
            log(f"NOT RETRIED {j['name']}: chain cap"); continue
        log(f"RETRY {j['name']} (first try: {status[j['name']]})")
        status[j["name"]] = runner.run(j) + " (retry)"
    counts = {}
    for s in status.values():
        counts[s] = counts.get(s, 0) + 1
    log(f"AB done in {(time.time() - t0) / 60:.1f} min: " + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
