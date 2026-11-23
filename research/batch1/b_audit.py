"""Audit of b2.jsonl / b4.jsonl (argv[1]); no model. Per row: hidden codes on needle prompts, distinct 4-grams and
longest repeated word run, wall tok/s / runtime tok/s, token count, and whether the text equals the earlier run of the
same method (research/dflash/baseline.jsonl for AR / A, dflash.jsonl D2-w4-adaptive for DFlash) or where it splits."""
import json
import sys
from pathlib import Path

DF = Path(__file__).parent.parent / "dflash"
CODES = json.load(open(DF / "prompts.json"))["codes"]
REF = {}
for f, src in (("baseline.jsonl", None), ("dflash.jsonl", "D2-w4-adaptive")):
    for l in open(DF / f):
        r = json.loads(l)
        if src is None or r["method"] == src:
            REF[("dflash" if src else r["method"], r["prompt"])] = r["text"]


def longest_repeat(w):
    """Longest n such that some n-word run occurs twice (non-overlapping start positions differ)."""
    best, n = 0, 1
    while n <= len(w) // 2:
        seen = set()
        if not any((t := tuple(w[i:i + n])) in seen or seen.add(t) for i in range(len(w) - n + 1)):
            break
        best, n = n, n + 1
    return best


rows = [json.loads(l) for l in open(sys.argv[1]) if '"text"' in l]
assert rows, "no rows"
for r in rows:
    t, w = r["text"], r["text"].split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    key = (r["method"] if r["method"] in ("AR", "A") else "dflash", r["prompt"])
    ref = REF.get(key)
    split = None if ref is None or t == ref else next((i for i, (a, b) in enumerate(zip(t, ref)) if a != b), min(len(t), len(ref)))
    print(json.dumps({"method": r["method"], "phase": r.get("phase"), "ops": r.get("ops"), "mb": r.get("mb"), "run": r.get("run"),
                      "prompt": r["prompt"], "tokens": r["tokens"], "full": r["tokens"] == 300,
                      "wall_over_rt": round(r["wall_tok_s"] / r["tok_s"], 3),
                      "codes": sum(c in t for c in CODES) if r["prompt"].startswith("needle") else None,
                      "distinct_4grams": round(len(set(g)) / max(1, len(g)), 3), "longest_repeat_words": longest_repeat(w),
                      "same_as_earlier": None if ref is None else t == ref, "split_char": split}))
