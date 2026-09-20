"""Fill in average precision (AP) for every Table II cell that predates AP
logging. Each phase reruns the EXACT shipped protocol of its AUROC artifact and
records AUROC + AP; the rerun AUROC is compared against the stored artifact as
a protocol check (drift is printed, never silently accepted).

  A. Six PyGOD baselines x 7 standard datasets: rebench_pygod_worker.py
     (seeds 0-4, epoch 200, per-cell subprocess isolation).
  B. CoLA x 3 social graphs: PyGOD defaults, seeds 1-5 (rebench_social.py
     protocol).
  C. SL-GAD x 7 standard (128 test rounds) + 3 social (256), seeds 1-5, via the
     validated port.

Already have AP elsewhere: ARCADE (coverage_bench.json), the other five PyGOD
baselines on the social graphs (rebench_social_suite.json), AutoGAD social
(autogad_faithful.json). AutoGAD's standard-7 AP was lost when the social pass
overwrote its artifact; rerunning its 60-evaluation search is a separate
decision.

Output: experiments/rebench_ap.json (incremental; keys "<Dataset>|<Method>").
"""
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np

OUT = os.path.join(HERE, "rebench_ap.json")
results = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(results, open(OUT, "w"), indent=2)


def note_drift(key, new_mean, ref_mean):
    if ref_mean is None:
        return ""
    d = abs(new_mean - ref_mean)
    return "" if d <= 0.15 else f"  DRIFT vs stored {ref_mean} (|d|={d:.1f})"


# ---------------------------------------------------------------- A: PyGOD
PG_REF = json.load(open(os.path.join(HERE, "rebench_pygod.json")))
STD = [("Enron", "enron"), ("Reddit", "reddit"), ("Books", "books"),
       ("Disney", "disney"), ("Weibo", "weibo"), ("Amazon", "inj_amazon"),
       ("Cora", "inj_cora")]
METHODS = ["DOMINANT", "AnomalyDAE", "DONE", "CoLA", "GAAN", "OCGNN"]

for label, key in STD:
    for m in METHODS:
        rk = f"{label}|{m}"
        if rk in results:
            continue
        ref = PG_REF.get(label, {}).get(m)
        ref_mean = None if (ref is None or "error" in ref) else ref["mean"]
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "rebench_pygod_worker.py"), key, m],
            capture_output=True, text=True, timeout=7200, cwd=ROOT)
        line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
        if proc.returncode != 0 or line is None:
            results[rk] = {"error": (proc.stderr.strip().splitlines() or ["?"])[-1][:120]}
            print(f"[A] {rk:22s} FAILED ({time.time()-t0:.0f}s)", flush=True)
        else:
            r = json.loads(line[len("RESULT "):])
            results[rk] = r
            print(f"[A] {rk:22s} auroc={r['mean']}+-{r['std']} ap={r['ap_mean']}+-{r['ap_std']} "
                  f"({time.time()-t0:.0f}s){note_drift(rk, r['mean'], ref_mean)}", flush=True)
        save()

# ---------------------------------------------------------------- B: CoLA social
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from arcade.data.loader import GraphStore

SOCIAL = [("BlogCatalog", "blogcatalog"), ("ACM", "acm"), ("Flickr", "cola_flickr")]
SOC_REF = json.load(open(os.path.join(HERE, "rebench_social.json")))

for label, key in SOCIAL:
    rk = f"{label}|CoLA"
    if rk in results:
        continue
    from pygod.detector import CoLA
    s = GraphStore(); s.load_dataset(key)
    y = (np.asarray(s.labels) > 0).astype(int)
    aucs, aps = [], []
    t0 = time.time()
    for seed in [1, 2, 3, 4, 5]:
        torch.manual_seed(seed); np.random.seed(seed)
        det = CoLA(gpu=0 if torch.cuda.is_available() else -1)
        det.fit(s.data)
        sc = det.decision_score_.numpy()
        aucs.append(roc_auc_score(y, sc) * 100)
        aps.append(average_precision_score(y, sc) * 100)
        torch.cuda.empty_cache()
    r = {"mean": round(float(np.mean(aucs)), 1), "std": round(float(np.std(aucs)), 1),
         "ap_mean": round(float(np.mean(aps)), 1), "ap_std": round(float(np.std(aps)), 1), "n": 5}
    results[rk] = r
    ref_mean = SOC_REF.get(f"CoLA:{key}", {}).get("mean")
    print(f"[B] {rk:22s} auroc={r['mean']}+-{r['std']} ap={r['ap_mean']}+-{r['ap_std']} "
          f"({time.time()-t0:.0f}s){note_drift(rk, r['mean'], ref_mean)}", flush=True)
    save()

# ---------------------------------------------------------------- C: SL-GAD
from types import SimpleNamespace

import sl_gad_run as slg

SL_REF = json.load(open(os.path.join(HERE, "rebench_slgad.json")))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def slgad_cell(label, key, rounds, ref_mean):
    rk = f"{label}|SL-GAD"
    if rk in results:
        return
    adj, feat, ano = slg.load_ours(key)
    aucs, aps = [], []
    t0 = time.time()
    for sd in [1, 2, 3, 4, 5]:
        args = SimpleNamespace(lr=1e-3, weight_decay=0.0, embedding_dim=64, num_epoch=100,
                               patience=400, batch_size=300, subgraph_size=4, readout='avg',
                               auc_test_rounds=rounds, negsamp_ratio=1, alpha=1.0, beta=0.6,
                               seed=sd, return_scores=False)
        auc, ap = slg.run(adj, feat, ano, args, device)
        aucs.append(auc); aps.append(ap)
    r = {"mean": round(float(np.mean(aucs)), 1), "std": round(float(np.std(aucs)), 1),
         "ap_mean": round(float(np.mean(aps)), 1), "ap_std": round(float(np.std(aps)), 1),
         "rounds": rounds, "n": 5}
    results[rk] = r
    print(f"[C] {rk:22s} auroc={r['mean']}+-{r['std']} ap={r['ap_mean']}+-{r['ap_std']} "
          f"({time.time()-t0:.0f}s){note_drift(rk, r['mean'], ref_mean)}", flush=True)
    save()

for label, key in STD:
    slgad_cell(label, key, 128, SL_REF.get(label, {}).get("mean"))
for label, key in SOCIAL:
    ref = SOC_REF.get(f"SL-GAD:{key}", {}).get("mean")
    slgad_cell(label, key, 256, ref)

print("AP_DONE", flush=True)
