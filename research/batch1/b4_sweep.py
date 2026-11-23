"""B4: DFlash knobs, w4:gs64 drafter, research/dflash/dflash_child.py (guarded 14/14), one child per config x run,
DFLASH_VERIFY_QMM unset (= §13's D2). Phases (argv[1]):
  window   argv[2], argv[3] = two draft windows (adaptive), ABBA x2, on chat, code_edit, needle_4k, needle_16k.
           dflash-mlx uses max(requested, the drafter's sliding_window = 4096), so anything below 4096 is 4096.
  tree     adaptive (window 1024) vs ddtree, ABBA x2, all 5 prompts, short prompts first. If the guard stops a child,
           its finished rows are kept, the peak is recorded, and the phase stops (the guard is never lowered).
  profile  argv[2] = verify mode, argv[3] = window: one PROFILE=1 run (dflash-mlx's own per-cycle trace), all 5 prompts.
Rows to b4.jsonl."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DF = HERE.parent / "dflash"
PY = os.path.expanduser("<venv>/bin/python")
OUT = HERE / "b4.jsonl"
ALL = "chat,math,code_edit,needle_4k,needle_16k"


def child(phase, run, vmode, window, only, profile=False):
    label = f"w4_{vmode}_win{window}" + ("_prof" if profile else "")
    env = dict(os.environ, PROMPTS=str(DF / "prompts.json"), ONLY=only, MAX="300", DRAFT_WINDOW=str(window))
    env.pop("DFLASH_VERIFY_QMM", None)
    if profile: env["PROFILE"] = "1"
    p = subprocess.run(["taskpolicy", "-b", "nice", "-n", "10", "/opt/homebrew/bin/gtimeout", "900", PY,
                        str(DF / "dflash_child.py"), label, "w4:gs64", vmode, "0"], env=env, cwd=HERE,
                       capture_output=True, text=True)
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    with open(OUT, "a") as f:
        for r in rows:
            r |= {"phase": phase, "run": run, "verify_mode": vmode}
            f.write(json.dumps(r) + "\n")
            print(json.dumps({k: v for k, v in r.items() if k != "text"}), flush=True)
        if p.returncode or len(rows) != len(only.split(",")):
            err = {"phase": phase, "run": run, "label": label, "failed": True, "exit": p.returncode,
                   "rows_done": len(rows), "stderr_tail": p.stderr[-1500:]}
            f.write(json.dumps(err) + "\n"); print(json.dumps(err), flush=True)
            return False
    return True


if __name__ == "__main__":
    phase = sys.argv[1]
    if phase == "window":
        wa, wb = int(sys.argv[2]), int(sys.argv[3])
        for run, w in ((1, wa), (1, wb), (2, wb), (2, wa)):
            if not child(phase, run, "adaptive", w, "chat,code_edit,needle_4k,needle_16k"):
                sys.exit(1)
    elif phase == "tree":
        for run, m in ((1, "adaptive"), (1, "ddtree"), (2, "ddtree"), (2, "adaptive")):
            if not child(phase, run, m, 1024, ALL):
                sys.exit(1)
    elif phase == "profile":
        child(phase, 1, sys.argv[2], int(sys.argv[3]), ALL, profile=True)
