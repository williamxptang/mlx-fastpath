"""16k / 32k test of the fast MTP setup on Qwen3.5-9B-OptiQ-4bit, with an audit for 'fast but rubbish':
  - task with ground truth: 3 secret codes hidden at 10/50/90% depth; answer must list all three
  - teacher-forced re-score by the plain model (argmax agreement, mean NLL)
  - repetition: distinct 4-gram ratio, longest repeated token span
  - wall-clock tok/s measured independently of optiq's own stats; token count checked
Capped: MLX limit 8 GB + 9 GB watchdog; prefill in 256-token chunks. Outputs saved for reading."""
import sys, json, time
sys.argv = [sys.argv[0]]
exec(open("<staging>/exp/tps20.py").read().split("CONFIGS = [")[0])
from mlx_lm.models.cache import make_prompt_cache

model = eng._model
_pf = OptiqEngine._prefill_chunked
OptiqEngine._prefill_chunked = lambda self, ids, cache, step=None: _pf(self, ids, cache, step or 256)
GEN = 200
CODES = ["KESTREL-4471", "MAGENTA-0928", "OBSIDIAN-3365"]
TASK = ("\n\nTask 1: The document above contains exactly three secret codes (each written as 'The secret code "
        "number N is ...'). List all three codes exactly.\nTask 2: Then summarize the document in 5 bullet points.")


def long_prompt(n):
    body = list(corpus[:n])
    for i, (frac, code) in enumerate(zip((0.1, 0.5, 0.9), CODES)):
        at = int(len(body) * frac)
        body[at:at] = tok.encode(f"\n\nThe secret code number {i + 1} is {code}.\n\n")
    text = tok.decode(body) + TASK
    return tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False))


def rescore(prompt, gen):
    cache = make_prompt_cache(model)
    seq = prompt + gen
    for i in range(0, len(prompt) - 1, 256):
        model(mx.array(seq[i:min(i + 256, len(prompt) - 1)])[None], cache=cache); mx.eval([c.state for c in cache])
    logits = model(mx.array(seq[len(prompt) - 1:-1])[None], cache=cache)[0].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = mx.array(gen)
    nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).mean().item()
    agree = (logits.argmax(-1) == tgt).astype(mx.float32).mean().item()
    del cache
    return round(agree, 4), round(nll, 4)


def repetition(g):
    grams = [tuple(g[i:i + 4]) for i in range(len(g) - 3)]
    distinct = len(set(grams)) / max(1, len(grams))
    longest = 0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            k = 0
            while j + k < len(g) and g[i + k] == g[j + k]: k += 1
            longest = max(longest, k)
    return round(distinct, 3), longest


CONFIGS = [  # name, depth, kv_bits, swaps, rescore?
    ("AR", 0, None, {}, False),
    ("MTP1+cap+vocab32k+fold+q4kv", 1, 4, dict(cap=True, vocab=32768, fold=True), True),
    ("MTP2+cap+vocab32k+fold+q4kv", 2, 4, dict(cap=True, vocab=32768, fold=True), False),
]
texts = {}
for n in (16000, 32000):
    ids = long_prompt(n)
    for name, depth, kvb, sw, rs in CONFIGS:
        configure(**sw)
        t0 = time.perf_counter()
        st = eng.generate("", prompt_ids=ids, max_tokens=GEN, depth=depth, temperature=0.0, kv_bits=kvb)
        wall = time.perf_counter() - t0
        configure()
        gen = tok.encode(st.text)
        found = [c for c in CODES if c in st.text]
        distinct, longest = repetition(gen)
        row = {"prompt_tokens": len(ids), "config": name, "tok_s": round(st.decode_tok_s, 2),
               "wall_tok_s_excl_prefill": round(st.generated_tokens / (wall - st.prefill_time_s), 2),
               "tokens": st.generated_tokens, "accept": round(st.acceptance_rate, 3) if depth else None,
               "codes_found": f"{len(found)}/3", "distinct_4gram": distinct, "longest_repeat_tokens": longest,
               "prefill_s": round(st.prefill_time_s, 1)}
        if rs:
            row["rescore_agree"], row["rescore_nll"] = rescore(ids, gen)
        row["peak_footprint_gb"] = round(watchdog.peak, 2)
        print(json.dumps(row), flush=True)
        texts[f"{len(ids)} | {name}"] = st.text
        json.dump(texts, open("tps_long2_texts.json", "w"), ensure_ascii=False, indent=1)
