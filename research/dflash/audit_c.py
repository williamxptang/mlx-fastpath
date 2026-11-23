"""Item 5c task checks on long_out.jsonl: code_edit = the returned file parses, has both edits, and equals the expected
file; copy_table = the main "Config (tok/s)" table of results.md §9 is reproduced exactly. Plus common text with AR. No model."""
import ast, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
src = (ROOT / "mlx_fastpath/attention.py").read_text()
want = src.replace('MODULES = ("mlx_lm.models.qwen3_next",)', 'MODULES = ("mlx_lm.models.qwen3_next", "mlx_lm.models.qwen3_5_moe")').replace("MAX_VERIFY = 16", "MAX_VERIFY = 12")
md = (ROOT / "research/kmic68/research/261118-transferable-ideas/results.md").read_text()
sec9 = md[md.index("## 9."):md.index("## 10.")]
lines = sec9.splitlines(); i0 = next(i for i, l in enumerate(lines) if l.startswith("| Config (tok/s)"))  # the table every method copied (§9 also has a smaller "Part | GB" table first)
table = "\n".join(lines[i0:next(i for i in range(i0, len(lines)) if not lines[i].startswith("|"))])
rows = [json.loads(l) for l in open("long_out.jsonl")]
ar = {r["prompt"]: r["text"] for r in rows if r["method"] == "AR"}
for r in rows:
    t, out = r["text"], {"method": r["method"], "prompt": r["prompt"], "tokens": r["tokens"]}
    if r["prompt"] == "code_edit":
        code = t.split("```python\n", 1)[-1].split("```", 1)[0]
        try:
            ast.parse(code); out["parses"] = True
        except SyntaxError:
            out["parses"] = False
        out["both_edits"] = "mlx_lm.models.qwen3_5_moe" in code and "MAX_VERIFY = 12" in code
        out["equals_expected_file"] = code.strip() == want.strip()
    else:
        out["table_exact"] = table in t
    out["same_as_AR"] = t == ar[r["prompt"]]
    out["common_prefix_with_AR"] = next((i for i, (a, b) in enumerate(zip(t, ar[r["prompt"]])) if a != b), min(len(t), len(ar[r["prompt"]])))
    out["chars"] = len(t)
    print(json.dumps(out))
