"""Is the fast MTP output as good as plain greedy? Generate AR and MTP2+cap+vocab32k outputs, then re-score each
with ONE teacher-forced plain forward: argmax agreement with the produced token, and mean NLL per token.
Control: the AR output re-scored the same way (batched numerics vs one-at-a-time). Capped like tps20.py."""
import sys, json
sys.argv = [sys.argv[0]]
exec(open("<staging>/exp/tps20.py").read().split("CONFIGS = [")[0])
from mlx_lm.models.cache import make_prompt_cache

model = eng._model


def rescore(prompt, gen):
    cache = make_prompt_cache(model)
    seq = prompt + gen
    for i in range(0, len(prompt) - 1, 512):
        model(mx.array(seq[i:min(i + 512, len(prompt) - 1)])[None], cache=cache); mx.eval([c.state for c in cache])
    logits = model(mx.array(seq[len(prompt) - 1:-1])[None], cache=cache)[0].astype(mx.float32)  # predicts gen
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = mx.array(gen)
    nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).mean().item()
    agree = (logits.argmax(-1) == tgt).astype(mx.float32).mean().item()
    margin = mx.sort(logits, axis=-1)[:, -2:]
    gap = (margin[:, 1] - margin[:, 0])
    return round(agree, 4), round(nll, 4), round(mx.median(gap).item() if hasattr(mx, "median") else 0, 3)


for n in (0, 4000, 8000):
    ids = prompt_ids(n)
    out = {}
    for name, depth, sw in (("AR", 0, {}), ("MTP2+cap+vocab32k", 2, dict(cap=True, vocab=32768))):
        configure(**sw)
        st = eng.generate("", prompt_ids=ids, max_tokens=GEN, depth=depth, temperature=0.0)
        configure()
        gen = tok.encode(st.text)
        out[name] = (gen, st.decode_tok_s)
    for name, (gen, tps) in out.items():
        agree, nll, _ = rescore(ids, gen)
        print(json.dumps({"prompt_tokens": len(ids), "output": name, "tok_s": round(tps, 2), "tokens": len(gen),
                          "teacher_forced_argmax_agreement": agree, "mean_nll": nll,
                          "peak_footprint_gb": round(watchdog.peak, 2)}), flush=True)
    a, b = out["AR"][0], out["MTP2+cap+vocab32k"][0]
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    print(json.dumps({"prompt_tokens": len(ids), "first_differing_token": first,
                      "ar_text_at_split": tok.decode(a[max(0, first - 8):first + 6]),
                      "mtp_text_at_split": tok.decode(b[max(0, first - 8):first + 6])}, ensure_ascii=False), flush=True)
