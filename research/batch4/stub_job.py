"""Stand-in job for heavy.py --dry-run: prints two result rows and a guard report like the real jobs.
argv: NAME FAIL_ONCE (1 = fail on the first call, succeed on the next; the marker lives in dry/)."""
import json
import sys
import time
from pathlib import Path

name, fail_once = sys.argv[1], sys.argv[2] == "1"
marker = Path(__file__).resolve().parent / "dry" / f"{name}.tried"
if fail_once and not marker.exists():
    marker.write_text("x")
    print("Traceback: stub failure on purpose", file=sys.stderr)
    sys.exit(1)
for p in ("chat", "math"):
    time.sleep(0.2)
    print(json.dumps({"prompt": p, "stub": name, "tok_s": 1.0, "peak_footprint_gb": 0.1}), flush=True)
print(json.dumps({"guard": {"peak_footprint_gb": 0.1, "swap_growth_gb": 0.0}}))
