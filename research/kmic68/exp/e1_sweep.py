import subprocess, json, os, sys, io, random, statistics
import pyarrow.parquet as pq, soundfile as sf
B = "<asr>/test/bench/fleurs/parquet-data/en_us/validation-00000-of-00001.parquet"
t = pq.read_table(B).to_pylist(); random.Random(0).shuffle(t)
for i, r in enumerate(t[:8]):
    a, sr = sf.read(io.BytesIO(r["audio"]["bytes"])); sf.write(f"e1_clips/en-{i}.wav", a, sr)
SETTINGS = [(None, None), ("10", "10"), ("100", None), (None, "200"), ("100", "200"), ("300", "1000")]
PY = os.path.expanduser("<venv>/bin/python")
res = {s: [] for s in SETTINGS}
order = SETTINGS + SETTINGS[::-1] + SETTINGS  # ABBA-style, 3 runs each
for s in order:
    env = dict(os.environ)
    if s[0]: env["MLX_MAX_OPS_PER_BUFFER"] = s[0]
    if s[1]: env["MLX_MAX_MB_PER_BUFFER"] = s[1]
    out = subprocess.run(["taskpolicy", "-b", "nice", "-n", "10", "/opt/homebrew/bin/gtimeout", "300", PY, "e1_child.py"],
                         env=env, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    d = json.loads(out); res[s].append(d); print(out, flush=True)
base = statistics.median(x["secs"] for x in res[(None, None)])
for s, v in res.items():
    med = statistics.median(x["secs"] for x in v)
    print(json.dumps({"ops": s[0] or "default", "mb": s[1] or "default", "median_s": round(med, 3),
                      "speedup_vs_default": round(base / med, 3), "texts_identical": len({x["text_sha"] for x in v} | {res[(None, None)][0]["text_sha"]}) == 1}))
