"""H1 (E10, the fork's power-cap / sustained-load idea): setup A as shipped (fp.load with its defaults, so the
N2-gdnhead copy when it exists; fp.apply(); MTP depth 2) on the `chat` prompt, 300-token generations back to back
for DURATION seconds in one process (model loaded once). One row per generation (time, tok/s, acceptance), and
`macmon pipe -i 1000` alongside (GPU power, clock, busy share, temperatures) into h1_macmon.jsonl.
Bar (research/batch4/plan.md): report the first- vs last-minute median tok/s; flag if the last is < 90% of the first.
Env: DURATION (s, default 600), GUARD_FREE_GB / GUARD_CAP_GB / MLX_LIMIT_GB / MLX_CACHE_GB (set by the runner)."""
import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import json
import os
import subprocess
import time
from pathlib import Path

import mlx_fastpath as fp

HERE = Path(__file__).parent
MODEL = os.environ.get("MODEL", MODELS + "/Qwen3.5-9B-OptiQ-4bit")
DURATION = float(os.environ.get("DURATION", 600))
IDS = json.load(open(HERE.parent / "dflash" / "prompts.json"))["prompts"]["chat"]
MACMON = "/opt/homebrew/bin/macmon"

with fp.MemoryGuard(min_free_gb=float(os.environ.get("GUARD_FREE_GB", 14)),
                    max_footprint_gb=float(os.environ.get("GUARD_CAP_GB", 14))) as g:
    eng = fp.load(MODEL, memory_limit_gb=float(os.environ.get("MLX_LIMIT_GB", 10)),
                  cache_limit_gb=float(os.environ.get("MLX_CACHE_GB", 0.25)))
    applied = fp.apply()
    from mlx_fastpath.n2 import resolve_model
    used = str(resolve_model(MODEL))                     # the folder fp.load picked (N2 copy when present)
    mon = None
    if Path(MACMON).exists():
        mon = subprocess.Popen([MACMON, "pipe", "-i", "1000"], stdout=open(HERE / "h1_macmon.jsonl", "a"),
                               stderr=subprocess.DEVNULL)
    t0 = time.time()
    i = 0
    try:
        while time.time() - t0 < DURATION:
            st = fp.generate(eng, IDS, max_tokens=300)
            print(json.dumps({"prompt": "chat", "run": i, "t": round(time.time() - t0, 1), "unix": round(time.time(), 1),
                              "tokens": st.generated_tokens, "tok_s": round(st.decode_tok_s, 2),
                              "accept": round(st.acceptance_rate, 3), "model": used,
                              "applied": sorted(applied), "peak_footprint_gb": g.report()["peak_footprint_gb"]}),
                  flush=True)
            i += 1
    finally:
        if mon:
            mon.terminate()
    print(json.dumps({"guard": g.report()}))
