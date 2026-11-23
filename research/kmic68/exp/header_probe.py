import os, time, threading, subprocess, re
samples = []
def watch():
    pid = os.getpid()
    while True:
        out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True).stdout
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*(GB|MB|KB)", out)
        if m:
            gb = float(m.group(1)) * {"GB": 1, "MB": 1/1024, "KB": 1/2**20}[m.group(2)]
            samples.append((round(time.time() - T0, 1), round(gb, 2)))
            if gb > 12: print("ABORT", samples[-6:], flush=True); os._exit(3)
        time.sleep(0.5)
T0 = time.time()
threading.Thread(target=watch, daemon=True).start()
src = open("tps20.py").read().split("CONFIGS = [")[0]
marks = {}
for i, part in enumerate(src.split("\n\n\n")):
    exec(part, globals())
    marks[i] = (round(time.time() - T0, 1), part.strip().splitlines()[0][:60])
time.sleep(2)
print("marks", marks)
print("max", max(g for _, g in samples), "timeline", samples[::4])
