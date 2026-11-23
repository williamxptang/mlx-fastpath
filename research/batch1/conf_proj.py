"""B3 offline projection from conf.jsonl (no model). Confidence trimming of setup A's depth-2 chain:
keep d1 only if p1 >= t (else the cycle is a plain width-1 step), keep d2 only if d1 was kept and p2 >= t.
Per position: width = 1 + kept drafts; tokens = 1 + a1 (if d1 kept) + a1*a2 (if d2 kept);
time = verify_ms[width] + draft_step_ms * (draft steps run: step 1 always, since p1 is known only after it; step 2
only if d1 was kept). t = 0 is setup A as shipped (always width 3).
Projected tok/s = sum tokens / sum time (each position treated as a cycle start, as in research/branch/sim.py).
Writes conf_proj.jsonl."""
import json
from pathlib import Path

HERE = Path(__file__).parent
T = (0.0, 0.1, 0.2, 0.3, 0.5)
rows = [json.loads(l) for l in open(HERE / "conf.jsonl")]
logs = {r["prompt"]: r for r in rows if "p1" in r}
cost = {r["prompt"]: r for r in rows if "verify_ms" in r}
assert logs and set(logs) <= set(cost), (list(logs), list(cost))
out = open(HERE / "conf_proj.jsonl", "w")
for name, r in logs.items():
    v = {int(k): ms for k, ms in cost[name]["verify_ms"].items()}
    ds = cost[name]["draft_step_ms"]
    res = {}
    for t in T:
        tok = ms = 0.0
        for p1, p2, a1, a2 in zip(r["p1"], r["p2"], r["a1"], r["a2"]):
            k1 = p1 >= t
            k2 = k1 and p2 >= t
            tok += 1 + (a1 if k1 else 0) + (a1 and a2 if k2 else 0)
            ms += v[1 + k1 + k2] + ds * (2 if k1 else 1)
        res[t] = {"tok_per_cycle": round(tok / r["positions"], 3), "ms_per_cycle": round(ms / r["positions"], 2),
                  "tok_s": round(1e3 * tok / ms, 2)}
    best = max(T, key=lambda t: res[t]["tok_s"])
    row = {"prompt": name, "context": r["context"], "verify_ms": v, "draft_step_ms": ds,
           "by_t": {str(t): res[t] for t in T}, "best_t": best,
           "gain_vs_t0": round(res[best]["tok_s"] / res[0.0]["tok_s"] - 1, 4)}
    print(json.dumps(row)); out.write(json.dumps(row) + "\n")
