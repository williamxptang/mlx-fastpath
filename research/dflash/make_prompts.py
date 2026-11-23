"""Builds the shared prompt set (token ids) so every method decodes exactly the same input."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
from pathlib import Path
from transformers import AutoTokenizer
ROOT = Path(__file__).resolve().parents[2]
tok = AutoTokenizer.from_pretrained(MODELS + "/Qwen3.5-9B-OptiQ-4bit")
CODES = ["KESTREL-4471", "MAGENTA-0928", "OBSIDIAN-3365"]
chat = lambda t: tok.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False)
docs = sorted((ROOT / "research/kmic68/research").rglob("*.md"))
corpus = tok.encode("\n\n".join(p.read_text() for p in docs) * 3)


def needle(n):
    body = list(corpus[:n])
    for i, (frac, code) in enumerate(zip((0.1, 0.5, 0.9), CODES)):
        at = int(len(body) * frac); body[at:at] = tok.encode(f"\n\nThe secret code number {i + 1} is {code}.\n\n")
    return chat(tok.decode(body) + "\n\nTask 1: List the three secret codes in the document exactly. "
                                   "Task 2: Summarize the document in 5 bullet points.")


src = (ROOT / "mlx_fastpath/attention.py").read_text()
P = {
    "chat": chat("Explain, step by step, how a CPU cache works and why cache lines matter for performance."),
    "math": chat("The function $f$ satisfies the functional equation \\[ f(x) + f(y) = f(x + y) - xy - 1 \\] for all real "
                 "numbers $x$ and $y$. If $f(1) = 1$, then find all integers $n$ such that $f(n) = n$. Enter all such "
                 "integers, separated by commas. Please reason step by step, and put your final answer within \\boxed{}."),
    "code_edit": chat("Here is a Python file:\n\n```python\n" + src + "\n```\n\nReturn the complete file, unchanged except: "
                      "add \"mlx_lm.models.qwen3_5_moe\" to the MODULES tuple and set MAX_VERIFY = 12. Output only the code."),
    "needle_4k": needle(4000),
    "needle_16k": needle(16000),
}
P = {k: (list(v["input_ids"]) if hasattr(v, "keys") else list(v)) for k, v in P.items()}
json.dump({"prompts": P, "codes": CODES}, open("prompts.json", "w"))
print({k: len(v) for k, v in P.items()})
