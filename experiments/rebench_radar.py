"""Radar (Li et al., IJCAI 2017) as an additional baseline on all benchmarks,
PyGOD implementation with library defaults, 5 seeds, AUROC + AP, via the same
per-cell isolated worker as the other PyGOD baselines. Writes rebench_radar.json."""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "rebench_radar.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
KEYS = [("Enron","enron"),("Reddit","reddit"),("Books","books"),("Disney","disney"),("Weibo","weibo"),
        ("Cora","inj_cora"),("Amazon","inj_amazon"),("BlogCatalog","blogcatalog"),("ACM","acm"),("Flickr","cola_flickr")] \
       + [(k, k) for k in sys.argv[1:]]
for label, key in KEYS:
    if label in res: continue
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, "rebench_pygod_worker.py"), key, "Radar"],
                       capture_output=True, text=True, timeout=7200, cwd=ROOT)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    res[label] = json.loads(line[7:]) if line else {"error": (p.stderr.strip().splitlines() or ["?"])[-1][:120]}
    print(f"[radar] {label:12s} {res[label]} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
print("RADAR_DONE", flush=True)
