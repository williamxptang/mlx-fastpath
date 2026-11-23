import os as _os
MODELS = _os.path.expanduser(_os.environ.get("MLX_MODELS", "~/models"))
import os, time, threading, subprocess, re, sys
import mlx.core as mx
mx.set_memory_limit(8 * 2**30); mx.set_cache_limit(1 * 2**30)
samples = []; stage = ["start"]
def watch():
    pid = os.getpid()
    while True:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True).stdout
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*(GB|MB|KB)", out)
        if m:
            gb = float(m.group(1)) * {"GB": 1, "MB": 1/1024, "KB": 1/2**20}[m.group(2)]
            samples.append((stage[0], gb))
            if gb > 12: print("ABORT >12GB at", stage[0], flush=True); os._exit(3)
        time.sleep(0.5)
threading.Thread(target=watch, daemon=True).start()
from mlx_lm import load
stage[0] = "mlx_lm.load"; m, t = load(MODELS + "/Qwen3.5-9B-OptiQ-4bit")
print("after load: active", round(mx.get_active_memory()/2**30,2), "peak", round(mx.get_peak_memory()/2**30,2), "cache", round(mx.get_cache_memory()/2**30,2), flush=True)
stage[0] = "from_loaded"
from optiq.runtime.engine import OptiqEngine
e = OptiqEngine.from_loaded(m, t, MODELS + "/Qwen3.5-9B-OptiQ-4bit")
print("after attach: active", round(mx.get_active_memory()/2**30,2), "peak", round(mx.get_peak_memory()/2**30,2), "cache", round(mx.get_cache_memory()/2**30,2), flush=True)
stage[0] = "idle"; time.sleep(3)
import collections
mx_ = collections.defaultdict(float)
for s, g in samples: mx_[s] = max(mx_[s], g)
print({k: round(v, 2) for k, v in mx_.items()})
