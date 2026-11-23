"""Shared setup for the opt-in real-model tests: where the models are, and memory caps sized from each model.

Caps come from the weights on disk (language model shards + the MTP sidecar; a vision sidecar is never loaded):
MLX limit = weights + 1.2 GB (KV cache, activations, LM head output), footprint cap = MLX limit + 2 GB (Python,
Metal driver), free memory needed to start = footprint cap + 1 GB. For Qwen3.5-9B-OptiQ-4bit (6.8 GB) that gives the
8 / 10 / 11 GB its slow test always used.
"""
import os
from pathlib import Path

LLM = Path(os.environ.get("MLX_FASTPATH_MODELS", "~/models")).expanduser()
GB = 2**30


def slow_enabled():
    return os.environ.get("MLX_FASTPATH_SLOW") == "1"


def weights_gb(path):
    path = Path(path)
    files = list(path.glob("model*.safetensors")) + list(path.glob("optiq/mtp.safetensors"))
    return sum(f.stat().st_size for f in files) / GB


def caps(path):
    """(mlx_limit_gb, max_footprint_gb, min_free_gb) for a model folder."""
    mlx_limit = round(weights_gb(path) + 1.2, 1)
    return mlx_limit, mlx_limit + 2.0, mlx_limit + 3.0
