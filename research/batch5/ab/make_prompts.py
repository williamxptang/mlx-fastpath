"""Builds prompts_text.json: the A/B prompts as TEXT (each model tokenizes them with its own chat template).
chat / math / code_edit / needle_4k as in research/dflash/make_prompts.py (needle body cut to ~4000 Qwen tokens),
plus a graded QA set: 12 questions with one exact answer each."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
tok = AutoTokenizer.from_pretrained(MODELS + "/Qwen3.5-0.8B-bf16")
CODES = ["KESTREL-4471", "MAGENTA-0928", "OBSIDIAN-3365"]
docs = sorted((ROOT / "research/kmic68/research").rglob("*.md"))
corpus = tok.encode("\n\n".join(p.read_text() for p in docs) * 3)
body = list(corpus[:4000])
for i, (frac, code) in enumerate(zip((0.1, 0.5, 0.9), CODES)):
    at = int(len(body) * frac)
    body[at:at] = tok.encode(f"\n\nThe secret code number {i + 1} is {code}.\n\n")
needle = (tok.decode(body) + "\n\nTask 1: List the three secret codes in the document exactly. "
          "Task 2: Summarize the document in 5 bullet points.")
src = (ROOT / "mlx_fastpath/attention.py").read_text()
PROMPTS = {
    "chat": "Explain, step by step, how a CPU cache works and why cache lines matter for performance.",
    "math": ("The function $f$ satisfies the functional equation \\[ f(x) + f(y) = f(x + y) - xy - 1 \\] for all real "
             "numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such "
             "integers, separated by commas. Please reason step by step, and put your final answer within \\boxed{}."),
    "code_edit": ("Here is a Python file:\n\n```python\n" + src + "\n```\n\nReturn the complete file, unchanged except: "
                  "add \"mlx_lm.models.qwen3_5_moe\" to the MODULES tuple and set MAX_VERIFY = 12. Output only the code."),
    "needle_4k": needle,
}
QA = [("What is 17 * 23? Answer with the number only.", "391"),
      ("What is the capital of Australia? Answer with one word.", "canberra"),
      ("How many days are in a leap year? Answer with the number only.", "366"),
      ("What is 144 divided by 12? Answer with the number only.", "12"),
      ("Which planet is known as the Red Planet? Answer with one word.", "mars"),
      ("What is the chemical symbol for gold? Answer with the symbol only.", "au"),
      ("How many meters are in 3 kilometers? Answer with the number only.", "3000"),
      ("What is the square root of 81? Answer with the number only.", "9"),
      ("Who wrote Romeo and Juliet? Answer with the author's last name only.", "shakespeare"),
      ("What is 2 to the power of 10? Answer with the number only.", "1024"),
      ("How many sides does a hexagon have? Answer with the number only.", "6"),
      ("What is 1000 minus 357? Answer with the number only.", "643")]
json.dump({"prompts": PROMPTS, "codes": CODES, "math_answer": ["1", "-2"], "qa": QA},
          open(Path(__file__).parent / "prompts_text.json", "w"), indent=1)
print({k: len(tok.encode(v)) for k, v in PROMPTS.items()}, len(QA), "QA")
