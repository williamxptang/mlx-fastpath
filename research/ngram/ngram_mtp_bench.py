"""#3 n-gram (prompt lookup) + MTP drafting on Qwen3.5-9B-OptiQ-4bit, vs plain decoding and our MTP setup A.
Methods (greedy, same prompt, max 400 tokens):
  AR  : optiq engine depth 0
  A   : optiq engine MTP depth 2 + mlx_fastpath.apply() (cheap draft head, fused capture, fold, fast prefill)
  N   : optiq ngram_generate_step with its stock MtpDrafter (full-vocab head, depth 1)
  N+  : same, MtpDrafter swapped for a 32k-vocab head at depth 2, plus mlx_fastpath.apply()
Audit: decode tok/s (wall, after first token), text vs AR (common prefix), distinct 4-grams, code task compiles.
Guard: refuses < 11 GB free; kills > 10 GB footprint or any swap growth."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import ast, json, sys, time
from pathlib import Path
import mlx.core as mx
import mlx_fastpath as fp
from mlx_fastpath.draft import sub_head, sub_logits

ROOT = Path(__file__).resolve().parents[2]
guard = fp.MemoryGuard(min_free_gb=11, max_footprint_gb=10).__enter__()
eng = fp.load(MODELS + "/Qwen3.5-9B-OptiQ-4bit", memory_limit_gb=8)
from optiq.runtime import ngram_spec as ng
tok = eng.tokenizer
MAX = 400


class FastMtpDrafter(ng.MtpDrafter):
    """MtpDrafter with the 32k-vocab head (draft ids < K only; the target still verifies every token)."""
    def __init__(self, model, depth=2, vocab=32768):
        super().__init__(model, depth=depth)
        head = self.model.lm_head if not getattr(self.model.args, "tie_word_embeddings", False) else self.model.model.embed_tokens
        self.sub = sub_head(head, vocab); mx.eval(self.sub[1:])

    def draft(self, hidden, token, position=None):
        out, h, nxt = [], hidden.reshape(1, 1, -1), token
        for i in range(self.depth):
            _, h = self.model._mtp_core(h, mx.array([[nxt]], mx.uint32), mtp_cache=self.model.make_mtp_cache(),
                                        emit_logits=False, position_offset=None if position is None else position + i)
            nxt = int(mx.argmax(sub_logits(self.sub, h)[0, -1]).item())
            out.append(nxt); h = h[:, -1:, :]
        return out


def chat(text):
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False))


src = (ROOT / "mlx_fastpath/attention.py").read_text()
results_md = (ROOT / "research/kmic68/research/261118-transferable-ideas/results.md").read_text()
sec9 = results_md[results_md.index("## 9."):results_md.index("## 10.")]
TASKS = {
    "code_edit": chat("Here is a Python file:\n\n```python\n" + src + "\n```\n\nReturn the complete file, unchanged "
                      "except: add \"mlx_lm.models.qwen3_5_moe\" to the MODULES tuple and set MAX_VERIFY = 12. "
                      "Output only the code in one ```python block."),
    "copy_table": chat(sec9 + "\n\nReproduce the first table of the section above exactly as markdown, then add one "
                       "sentence on which configuration is fastest."),
    "free_chat": chat("Explain, step by step, how a CPU cache works and why cache lines matter for performance."),
}


def run_engine(ids, depth):
    t0 = time.perf_counter()
    st = eng.generate("", prompt_ids=ids, max_tokens=MAX, depth=depth, temperature=0.0)
    return st.text, st.generated_tokens, st.decode_tok_s, st.prefill_time_s


def run_ngram(ids, drafter):
    stats = ng.NgramStats()
    gen = ng.ngram_generate_step(mx.array(ids), eng._model, context_tokens=ids, max_tokens=MAX,
                                 mtp_drafter=drafter, stats=stats, persist_costs=False)
    out, t_first, t0 = [], None, time.perf_counter()
    for t, _ in gen:
        if t_first is None: t_first = time.perf_counter()
        t = int(t)
        if t in eng._eos_ids(): break
        out.append(t)
        if len(out) >= MAX: break
    el = time.perf_counter() - t_first
    return tok.decode(out), len(out), (len(out) - 1) / el, t_first - t0


def audit(text, ref):
    g = tok.encode(text)
    grams = [tuple(g[i:i + 4]) for i in range(len(g) - 3)]
    prefix = next((i for i, (a, b) in enumerate(zip(text, ref)) if a != b), min(len(text), len(ref)))
    return {"distinct_4gram": round(len(set(grams)) / max(1, len(grams)), 3), "same_as_AR": text == ref,
            "common_prefix_chars": prefix, "chars": len(text)}


def compiles(text):
    if "```python" not in text: return False
    code = text.split("```python", 1)[1].split("```", 1)[0]
    try:
        ast.parse(code); return "qwen3_5_moe" in code and "MAX_VERIFY = 12" in code
    except SyntaxError:
        return False


texts = {}
for task, ids in TASKS.items():
    ref = None
    for method in ("AR", "A", "N", "N+"):
        fp.remove()
        if method == "AR":
            text, n, tps, pf = run_engine(ids, 0)
        elif method == "A":
            fp.apply(); text, n, tps, pf = run_engine(ids, 2)
        elif method == "N":
            text, n, tps, pf = run_ngram(ids, ng.MtpDrafter(eng._model, depth=1))
        else:
            fp.apply(); text, n, tps, pf = run_ngram(ids, FastMtpDrafter(eng._model, depth=2))
        fp.remove()
        if method == "AR": ref = text
        row = {"task": task, "prompt_tokens": len(ids), "method": method, "tokens": n, "tok_s": round(tps, 2),
               **audit(text, ref), **({"code_ok": compiles(text)} if task == "code_edit" else {}), **guard.report()}
        print(json.dumps(row), flush=True)
        texts[f"{task} | {method}"] = text
        json.dump(texts, open(ROOT / "research/ngram/ngram_mtp_texts.json", "w"), ensure_ascii=False, indent=1)
guard.__exit__(None, None, None)
