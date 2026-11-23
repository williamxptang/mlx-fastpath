"""Audit and summary of batch 2's rows (no model). Prints one JSON line per group; missing files are skipped.
- kl.jsonl: per label and prompt, KL / flips (the control-c64 rows are the floor).
- r1_dflash.jsonl: DFlash plain vs id-set head: tok/s median [range] per prompt, ratio, texts identical across variants
  and to results.md §13 (D2-w4-adaptive).
- r5_speed.jsonl: per method and prompt, each model's tok/s median [range] and ratio to the source; text vs the source.
- r5_tasks.jsonl: needle codes; code_edit at 800 tokens parses, has both edits, equals the source's AR file (§14.5c).
- r6_sample.jsonl: tok/s and MTP acceptance at temperature 0.7 vs greedy (r5 rows), codes per seed.
- ../batch3/t1_needle.jsonl: needle codes (0-3) per model, document and length; prompt seen in full; checks.
Common checks on every generation row: full token count, wall/runtime tok/s, distinct 4-grams."""
import ast
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
DF = HERE.parent / "dflash"
CODES = json.load(open(DF / "prompts.json"))["codes"]


def rows(name):
    p = HERE / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def med(v):
    return {"median": round(st.median(v), 2), "range": [round(min(v), 2), round(max(v), 2)], "n": len(v)}


def common(r, want_tokens):
    w = r["text"].split()
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return {"full": r["tokens"] == want_tokens, "wall_over_rt": round(r["wall_tok_s"] / r["tok_s"], 3),
            "distinct_4grams": round(len(set(g)) / max(1, len(g)), 3)}


def split_at(a, b):
    return None if a == b else next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def short(m):
    return Path(m).name.replace("Qwen3.5-9B-OptiQ-4bit", "source").replace("source-", "")


def kl():
    for r in rows("kl.jsonl"):
        print(json.dumps({"kl": r["label"], "prompt": r["prompt"], **{k: r[k] for k in
              ("positions", "kl_mean", "kl_p99", "kl_max", "flip_rate", "sure_flip_rate")}}))


def r1():
    R = rows("r1_dflash.jsonl")
    if not R:
        return
    d2 = {json.loads(l)["prompt"]: json.loads(l)["text"] for l in open(DF / "dflash.jsonl") if '"D2-w4-adaptive"' in l}
    by = defaultdict(list)
    for r in R:
        by[(r["prompt"], r["variant"])].append(r)
    for p in sorted({k[0] for k in by}):
        plain, idset = by[(p, "plain")], by[(p, "idset")]
        if not (plain and idset):     # e.g. the long run the guard stopped (needle_4k, plain only)
            print(json.dumps({"r1": p, "skipped": f"plain {len(plain)} / idset {len(idset)} rows"}))
            continue
        texts = {r["text"] for r in plain + idset}
        out = {"r1": p, "plain": med([r["tok_s"] for r in plain]), "idset": med([r["tok_s"] for r in idset]),
               "accept_plain": st.median(r["accept"] for r in plain), "accept_idset": st.median(r["accept"] for r in idset),
               "draft_rows": idset[0].get("draft_rows"), "texts_identical": len(texts) == 1,
               "same_as_s13": texts == {d2.get(p)}, "checks": [common(r, 300) for r in plain + idset]}
        out["ratio"] = round(out["idset"]["median"] / out["plain"]["median"], 3)
        print(json.dumps(out))


def r5():
    R = rows("r5_speed.jsonl")
    by = defaultdict(list)
    for r in R:
        by[(r["method"], r["prompt"], short(r["model"]))].append(r)
    for meth, p in sorted({k[:2] for k in by}):
        src = by.get((meth, p, "source"), [])
        out = {"r5": meth, "prompt": p}
        for m in ("source", "N2-gdnhead", "N2-all"):
            v = by.get((meth, p, m), [])
            if not v:
                continue
            out[m] = med([r["tok_s"] for r in v])
            if meth == "A":
                out[m]["accept"] = round(st.median(r["accept"] for r in v), 3)
            if src and m != "source":
                out[m]["ratio"] = round(out[m]["median"] / out["source"]["median"], 3)
                out[m]["text_split_vs_source"] = split_at(v[0]["text"], src[0]["text"])
            out[m]["checks_ok"] = all(c["full"] and 0.97 <= c["wall_over_rt"] <= 1.03 and c["distinct_4grams"] >= 0.9
                                      for c in (common(r, 300) for r in v))
        print(json.dumps(out))
    want = next((json.loads(l)["text"] for l in open(DF / "long_out.jsonl")
                 if json.loads(l)["method"] == "AR" and json.loads(l)["prompt"] == "code_edit"), None)
    for r in rows("r5_tasks.jsonl"):
        out = {"task": r["prompt"], "model": short(r["model"]), "tokens": r["tokens"], "tok_s": r["tok_s"]}
        if r["prompt"].startswith("needle"):
            out["codes"] = sum(c in r["text"] for c in CODES)
        else:
            code = r["text"].split("```python\n", 1)[-1].split("```", 1)[0]
            try:
                ast.parse(code); out["parses"] = True
            except SyntaxError:
                out["parses"] = False
            out["both_edits"] = "mlx_lm.models.qwen3_5_moe" in code and "MAX_VERIFY = 12" in code
            out["equals_source_file"] = want is not None and code.strip() == want.split("```python\n", 1)[-1].split("```", 1)[0].strip()
        print(json.dumps(out))


def r6():
    R = rows("r6_sample.jsonl")
    if not R:
        return
    greedy = defaultdict(list)
    for r in rows("r5_speed.jsonl"):
        greedy[(r["method"], short(r["model"]))].append(r)
    by = defaultdict(list)
    for r in R:
        by[(r["method"], short(r["model"]))].append(r)
    for (meth, m), v in sorted(by.items()):
        out = {"r6": meth, "model": m, "temp0.7": med([r["tok_s"] for r in v]),
               "greedy": med([r["tok_s"] for r in greedy[(meth, m)]]) if greedy[(meth, m)] else None,
               "per_prompt": {p: round(st.median(r["tok_s"] for r in v if r["prompt"] == p), 2) for p in sorted({r["prompt"] for r in v})}}
        if meth == "A":
            out["accept_temp0.7"] = round(st.median(r["accept"] for r in v), 3)
            out["accept_greedy"] = round(st.median(r["accept"] for r in greedy[(meth, m)]), 3) if greedy[(meth, m)] else None
        out["checks_ok"] = all(c["full"] and 0.97 <= c["wall_over_rt"] <= 1.03 and c["distinct_4grams"] >= 0.8
                               for c in (common(r, 300) for r in v))
        print(json.dumps(out))


def t1():
    R = rows("../batch3/t1_needle.jsonl")
    if not R:
        return
    var = json.load(open(HERE / "prompts_var.json"))
    codes = var["codes"]
    for r in R:
        print(json.dumps({"t1": short(r["model"]), "prompt": r["prompt"], "codes": sum(c in r["text"] for c in codes),
                          "prompt_tokens": r["prompt_tokens"], "tokens": r["tokens"], "tok_s": r["tok_s"],
                          "peak_gb": r.get("peak_footprint_gb"), "swap_growth_gb": r.get("swap_growth_gb"),
                          **common(r, 300)}))


if __name__ == "__main__":
    kl(); r1(); r5(); r6(); t1()
