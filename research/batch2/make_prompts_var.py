"""Needle prompts in 3 variants per length, for repeats at the crossovers (results.md §14.5a: DFlash vs A ~11k;
§14.6: bf16 vs 4-bit KV 20-24k). Same recipe and codes as research/dflash/make_prompts.py, so the existing audits
work. v1 is the original prompt (corpus from 0, codes at 10/50/90%); v2 and v3 start the document a third and two
thirds into the docs and move the codes. Tokenizer only, no model. Writes prompts_var.json."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DF = HERE.parent / "dflash"
sys.argv = ["make_prompts.py"]
ns = {}
exec((DF / "make_prompts.py").read_text().split("src = (ROOT")[0], ns)   # tokenizer, CODES, chat, corpus, needle()
tok, corpus, chat, CODES = ns["tok"], ns["corpus"], ns["chat"], ns["CODES"]
SIZES = [10000, 12000, 20000, 24000]
# the corpus is the docs 3 times over, so offsets stay inside the first copy (1/9, 2/9 = a third, two thirds of it)
VARIANTS = {"v1": (0.0, (0.1, 0.5, 0.9)), "v2": (1 / 9, (0.2, 0.6, 0.8)), "v3": (2 / 9, (0.15, 0.45, 0.85))}


def needle_var(n, start_frac, fracs):
    s = int(len(corpus) * start_frac)
    assert s + n <= len(corpus), (s, n, len(corpus))
    body = list(corpus[s:s + n])
    for i, (frac, code) in enumerate(zip(fracs, CODES)):
        at = int(len(body) * frac); body[at:at] = tok.encode(f"\n\nThe secret code number {i + 1} is {code}.\n\n")
    return chat(tok.decode(body) + "\n\nTask 1: List the three secret codes in the document exactly. "
                                   "Task 2: Summarize the document in 5 bullet points.")


P = {f"needle_{n // 1000}k_{v}": needle_var(n, s, f) for n in SIZES for v, (s, f) in VARIANTS.items()}
P = {k: (list(v["input_ids"]) if hasattr(v, "keys") else list(v)) for k, v in P.items()}
orig = json.load(open(DF / "prompts_long.json"))["prompts"]
for n in SIZES:   # v1 must be the prompt the earlier runs used
    assert P[f"needle_{n // 1000}k_v1"] == orig[f"needle_{n // 1000}k"], n
for n in SIZES:   # the variants are different documents, not the same text shifted
    v = [set(P[f"needle_{n // 1000}k_{x}"]) for x in VARIANTS]
    assert all(len(v[i] & v[j]) / len(v[i] | v[j]) < 0.9 for i in range(3) for j in range(i + 1, 3)), n
json.dump({"prompts": P, "codes": CODES}, open(HERE / "prompts_var.json", "w"))
print({k: len(v) for k, v in P.items()}, "corpus", len(corpus))
