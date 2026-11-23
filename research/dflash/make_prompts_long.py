"""Extra needle prompts (same recipe as make_prompts.py) for the crossover runs: 6k-12k (DFlash vs A) and
20k-32k (bf16 MTP vs 4-bit KV path, 32k verify attention). Writes prompts_long.json."""
import json
import sys
from pathlib import Path

ARGS, sys.argv = sys.argv[1:], ["make_prompts.py"]
here = Path(__file__).parent
src = (here / "make_prompts.py").read_text().split("src = (ROOT")[0]  # tokenizer, CODES, chat, corpus, needle()
ns = {}
exec(src, ns)
SIZES = [int(s) for s in ARGS] or [6000, 8000, 10000, 12000, 20000, 24000, 28000, 32000]
assert len(ns["corpus"]) >= max(SIZES), (len(ns["corpus"]), max(SIZES))
P = {f"needle_{n // 1000}k": ns["needle"](n) for n in SIZES}
P = {k: (list(v["input_ids"]) if hasattr(v, "keys") else list(v)) for k, v in P.items()}
json.dump({"prompts": P, "codes": ns["CODES"]}, open(here / "prompts_long.json", "w"))
print({k: len(v) for k, v in P.items()}, "corpus", len(ns["corpus"]))
