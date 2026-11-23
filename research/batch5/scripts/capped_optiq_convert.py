"""Run `optiq convert ARGS` under an MLX memory limit and mlx_fastpath's MemoryGuard (stops on footprint cap or swap growth)."""
import sys
import mlx.core as mx
from mlx_fastpath.guard import MemoryGuard
limit_gb, cap_gb, min_free = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
mx.set_memory_limit(int(limit_gb * 2**30))
with MemoryGuard(min_free_gb=min_free, max_footprint_gb=cap_gb) as g:
    from optiq.cli import cli
    try:
        cli(args=["convert", *sys.argv[4:]], standalone_mode=False)
    finally:
        print("[guard]", g.report(), "MLX peak GB", round(mx.get_peak_memory() / 2**30, 2), flush=True)
