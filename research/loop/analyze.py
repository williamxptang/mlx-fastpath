"""Paired comparison of loop configs against the plain model on the same chunks.

delta = NLL(config) - NLL(plain) per chunk (negative = better). 95% CI by bootstrap over chunks (fixed seed).
WIN: CI upper bound < 0 on BOTH sets. MEANINGFUL: also mean gain >= half of the plain gap to the yardstick model
(same chunks, same tokenizer).

    python analyze.py RESULTS_JSONL [YARDSTICK_JSONL] [YARDSTICK_MODEL_NAME]
    python analyze.py --pair RESULTS_JSONL CONFIG_A CONFIG_B      (paired A - B on the same chunks, same bootstrap)
"""
import json
import random
import sys

SETS = ("TEST-A", "TEST-B")


def ci(xs, n=4000, seed=0):
    rnd = random.Random(seed)
    k = len(xs)
    means = sorted(sum(xs[rnd.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


if sys.argv[1] == "--pair":
    rows = {json.loads(l)["config"]: json.loads(l) for l in open(sys.argv[2])}
    a, b = rows[sys.argv[3]], rows[sys.argv[4]]
    for s in SETS:
        d = [x - y for x, y in zip(a[s], b[s])]
        assert len(d) == len(a[s]) > 0
        lo, hi = ci(d)
        print(f"{s} {sys.argv[3]} - {sys.argv[4]}: {sum(d) / len(d):+.4f} [{lo:+.4f},{hi:+.4f}]")
    print("train loss last50:", a.get("train_loss_last50"), b.get("train_loss_last50"))
    sys.exit(0)

rows = [json.loads(l) for l in open(sys.argv[1])]
yard = [json.loads(l) for l in open(sys.argv[2])] if len(sys.argv) > 2 else []
yname = sys.argv[3] if len(sys.argv) > 3 else None
by_model = {}
for r in rows:
    by_model.setdefault(r["model"], []).append(r)
for model, rs in by_model.items():
    plain = next(r for r in rs if r["config"] == "plain")
    gap = {}
    for y in yard:
        if y["config"] == "plain" and (yname is None or yname in y["model"]) and y["model"] != model:
            for s in SETS:
                if len(y[s]) == len(plain[s]):
                    gap[s] = sum(p - q for p, q in zip(plain[s], y[s])) / len(y[s])
            gap["_model"] = y["model"]
    print(f"\n== {model}: plain " + "  ".join(f"{s} {plain[s + '_mean']:.4f}" for s in SETS)
          + (f"   | plain gap to {gap['_model']}: " + "  ".join(f"{s} {gap[s]:+.4f}" for s in SETS if s in gap)
             if gap else ""))
    out = []
    for r in rs:
        if r["config"] == "plain":
            continue
        res = {}
        for s in SETS:
            d = [a - b for a, b in zip(r[s], plain[s])]
            assert len(d) == len(plain[s]) > 0
            lo, hi = ci(d)
            res[s] = (sum(d) / len(d), lo, hi)
        win = all(res[s][2] < 0 for s in SETS)
        meaningful = win and gap and all(-res[s][0] >= 0.5 * gap[s] for s in SETS if s in gap)
        out.append((res["TEST-A"][0] + res["TEST-B"][0], r["config"], res, win, meaningful))
    for _, name, res, win, mean in sorted(out):
        print(f"{name:22s} " + "  ".join(f"{s} {res[s][0]:+.4f} [{res[s][1]:+.4f},{res[s][2]:+.4f}]" for s in SETS)
              + ("  WIN" if win else "") + ("  MEANINGFUL" if mean else ""))
