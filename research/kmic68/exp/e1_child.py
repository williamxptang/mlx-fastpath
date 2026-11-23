"""One timed ASR run under the env vars set by the parent. Memory capped at 2 GB."""
import sys, time, json, glob, hashlib, os
import mlx.core as mx
mx.set_memory_limit(2 * 2**30); mx.set_cache_limit(512 * 2**20)
from mlx_audio.stt.utils import load_model
m = load_model("<asr>/Qwen3-ASR-0.6B/Qwen3-ASR-0.6B-mlx-int8")
clips = sorted(glob.glob("<staging>/exp/e1_clips/*.wav"))
assert clips
m.generate(clips[0], language="English")  # warmup
texts, t0 = [], time.perf_counter()
for _ in range(2):
    for c in clips:
        texts.append(m.generate(c, language="English").text)
el = time.perf_counter() - t0
print(json.dumps({"ops": os.environ.get("MLX_MAX_OPS_PER_BUFFER", "default"), "mb": os.environ.get("MLX_MAX_MB_PER_BUFFER", "default"),
                  "secs": round(el, 3), "text_sha": hashlib.sha1("|".join(texts).encode()).hexdigest()[:10],
                  "peak_gb": round(mx.get_peak_memory() / 2**30, 2)}))
