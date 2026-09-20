"""Complete the PyGOD re-benchmark for the cells the crashed run missed, each in
an isolated subprocess (so GAAN's CUDA assert can't kill the batch). Merges into
rebench_pygod.json. A worker that crashes/errors -> that cell recorded as error
(shown as "/" in the paper).
"""
import os, sys, json, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = {"Enron": "enron", "Reddit": "reddit", "Books": "books", "Disney": "disney",
        "Amazon": "inj_amazon", "Cora": "inj_cora", "Weibo": "weibo"}
METHODS = ["DOMINANT", "AnomalyDAE", "DONE", "CoLA", "GAAN", "OCGNN"]

res = json.load(open(os.path.join(HERE, "rebench_pygod.json"))) if os.path.exists(
    os.path.join(HERE, "rebench_pygod.json")) else {}

for label, key in KEYS.items():
    res.setdefault(label, {})
    for m in METHODS:
        cur = res[label].get(m)
        if isinstance(cur, dict) and "mean" in cur:
            continue  # already have it
        try:
            out = subprocess.run([sys.executable, os.path.join(HERE, "rebench_pygod_worker.py"), key, m],
                                 capture_output=True, text=True, timeout=1800)
            line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
            if line:
                res[label][m] = json.loads(line[0][len("RESULT "):])
                print(f"[fix] {label:7s} {m:10s} {res[label][m]['mean']}+-{res[label][m]['std']}", flush=True)
            else:
                err = (out.stderr.strip().splitlines() or ["crashed"])[-1][:60]
                res[label][m] = {"error": err}
                print(f"[fix] {label:7s} {m:10s} ERR {err}", flush=True)
        except Exception as e:
            res[label][m] = {"error": str(e)[:60]}
            print(f"[fix] {label:7s} {m:10s} ERR {str(e)[:60]}", flush=True)
        json.dump(res, open(os.path.join(HERE, "rebench_pygod.json"), "w"), indent=2)
print("REBENCH_PYGOD_COMPLETE_DONE", flush=True)
