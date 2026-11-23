"""(B) End-to-end greedy MTP decode with optiq's engine on Qwen3.5-9B-OptiQ-4bit: depth 0..3, fold off/on,
short and long prompt. Phase timers wrapped at runtime around draft / verify / commit / rollback."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import sys, time, json, glob, collections
sys.path.insert(0, "<staging>/exp")
import mlx.core as mx
import patch_fold
from optiq.runtime.engine import OptiqEngine
import optiq.runtime.mtp.gdn_capture as gc
import optiq.runtime.mtp.cache_state as cs

MODEL = MODELS + "/Qwen3.5-9B-OptiQ-4bit"
MAX_TOK = int(sys.argv[1]) if len(sys.argv) > 1 else 200
phase = collections.defaultdict(float)
count = collections.Counter()


def timed(name, fn, ev=None):
    def w(*a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        if ev: mx.eval(ev(r))
        phase[name] += time.perf_counter() - t; count[name] += 1
        return r
    return w


gc.forward_with_gdn_capture = timed("verify", gc.forward_with_gdn_capture, lambda r: [r[0], r[1]])
gc.commit_captured_prefix = timed("commit", gc.commit_captured_prefix)
cs.rollback_after_verify = timed("rollback", cs.rollback_after_verify)
cs.snapshot_untrimmable_cache = timed("snapshot", cs.snapshot_untrimmable_cache)
OptiqEngine._mtp_draft = timed("draft", OptiqEngine._mtp_draft, lambda r: r[0])

eng = OptiqEngine(MODEL)
print("has_mtp", eng.has_mtp, flush=True)
docs = sorted(glob.glob("<staging>/research/261118-transferable-ideas/**/*.md", recursive=True))
long_text = "\n\n".join(open(p).read() for p in docs)
tok = eng.tokenizer
ids = tok.encode(long_text)
long_text = tok.decode(ids[:16000])
prompts = {
    "short": "Explain, step by step, how a CPU cache works and why cache lines matter for performance.",
    "long16k": long_text + "\n\nSummarize the document above in a numbered list of its key findings.",
}
for pname, prompt in prompts.items():
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ref_text = None
    for depth in (0, 1, 2, 3):
        for fold in (False, True):
            if depth == 0 and fold: continue
            patch_fold.enable(fold)
            phase.clear(); count.clear()
            st = eng.generate(text, max_tokens=MAX_TOK, temperature=0.0, depth=depth)
            out = st.text
            if depth == 0: ref_text = out
            same = out == ref_text
            prefix = next((i for i, (a, b) in enumerate(zip(out, ref_text)) if a != b), min(len(out), len(ref_text)))
            print(json.dumps({"prompt": pname, "depth": depth, "fold": fold, "decode_tok_s": round(st.decode_tok_s, 2),
                              "accept": round(st.acceptance_rate, 3) if depth else None,
                              "tokens": st.generated_tokens, "same_text_as_AR": same, "common_prefix_chars": prefix,
                              "phase_s": {k: round(v, 3) for k, v in phase.items()}, "phase_n": dict(count),
                              "decode_s": round(st.decode_time_s, 3)}), flush=True)
    patch_fold.enable(False)
print("attn calls", patch_fold.stats)
