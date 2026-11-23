"""Memory guard for model runs on a shared Mac: never swap.

    with MemoryGuard(min_free_gb=12, max_footprint_gb=10):
        ... load and run ...

Refuses to start without enough free memory (free + inactive + speculative pages), then watches the
process: kills it (exit 3) if its footprint passes the cap or if system swap grows at all
(tolerance 64 MB for other processes' noise).
"""
import os
import re
import subprocess
import sys
import threading
import time

GB = 2**30


def free_gb():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = sum(int(m.group(2)) for m in re.finditer(r"Pages (free|inactive|speculative):\s+(\d+)", out))
    return pages * page / GB


def swap_used_gb():
    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)([MG])", out)
    return float(m.group(1)) / (1024 if m.group(2) == "M" else 1)


def footprint_gb(pid=None):
    out = subprocess.run(["footprint", str(pid or os.getpid())], capture_output=True, text=True).stdout
    m = re.search(r"phys_footprint:\s*([\d.]+)\s*(GB|MB|KB)", out)
    return float(m.group(1)) * {"GB": 1, "MB": 1 / 1024, "KB": 1 / 2**20}[m.group(2)] if m else 0.0


class MemoryGuard:
    def __init__(self, min_free_gb=12.0, max_footprint_gb=10.0, swap_tolerance_gb=0.0625, interval=1.0):
        self.min_free, self.max_fp, self.tol, self.interval = min_free_gb, max_footprint_gb, swap_tolerance_gb, interval
        self.peak_footprint = 0.0
        self.swap_start = None
        self._stop = threading.Event()

    def _kill(self, why):
        print(f"[MemoryGuard] {why}; stopping the run", file=sys.stderr, flush=True)
        os._exit(3)

    def _watch(self):
        while not self._stop.wait(self.interval):
            fp = footprint_gb()
            self.peak_footprint = max(self.peak_footprint, fp)
            if fp > self.max_fp:
                self._kill(f"footprint {fp:.1f} GB > cap {self.max_fp} GB")
            grew = swap_used_gb() - self.swap_start
            if grew > self.tol:
                self._kill(f"swap grew {grew * 1024:.0f} MB")

    def __enter__(self):
        have = free_gb()
        if have < self.min_free:
            raise RuntimeError(f"[MemoryGuard] only {have:.1f} GB free, need {self.min_free} GB: not starting")
        self.swap_start = swap_used_gb()
        threading.Thread(target=self._watch, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        return False

    def report(self):
        return {"peak_footprint_gb": round(self.peak_footprint, 2),
                "swap_growth_gb": round(swap_used_gb() - self.swap_start, 3)}
