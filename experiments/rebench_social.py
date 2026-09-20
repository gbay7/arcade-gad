"""Self-run SL-GAD and CoLA on the three dense social benchmarks
(BlogCatalog, ACM, Flickr), 5 seeds each — same methodology as the standard
benchmarks, replacing the published-number comparison with a symmetric one and
measuring seed variance.

SL-GAD: the validated DGL-free port (sl_gad_run.run), reference protocol
(lr 1e-3, 100 epochs, 256 test rounds). CoLA: PyGOD implementation with
library defaults. Datasets load through the ARCADE GraphStore.
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import sl_gad_run as F

DS = ["blogcatalog", "acm", "cola_flickr"]
SEEDS = [1, 2, 3, 4, 5]
OUT = os.path.join(HERE, "rebench_social.json")
results = json.load(open(OUT)) if os.path.exists(OUT) else {}


def run_slgad(key, seed):
    adj, feat, ano = F.load_ours(key)
    class A: pass
    a = A()
    for k, v in dict(lr=1e-3, weight_decay=0.0, embedding_dim=64, num_epoch=100,
                     patience=400, batch_size=300, subgraph_size=4, readout='avg',
                     auc_test_rounds=256, negsamp_ratio=1, alpha=1.0, beta=0.6,
                     seed=seed).items():
        setattr(a, k, v)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    auc, ap = F.run(adj, feat, ano, a, dev)
    return auc


def run_cola(key, seed):
    from arcade.data.loader import GraphStore
    from pygod.detector import CoLA
    s = GraphStore(); s.load_dataset(key)
    d = s.data
    torch.manual_seed(seed); np.random.seed(seed)
    det = CoLA(gpu=0 if torch.cuda.is_available() else -1)
    det.fit(d)
    y = (np.asarray(s.labels) > 0).astype(int)
    return roc_auc_score(y, det.decision_score_.numpy()) * 100


for method, fn in [("SL-GAD", run_slgad), ("CoLA", run_cola)]:
    for ds in DS:
        key = f"{method}:{ds}"
        done = results.get(key, {}).get("per_seed", {})
        for seed in SEEDS:
            if str(seed) in done:
                continue
            t = time.time()
            try:
                auc = fn(ds, seed)
                done[str(seed)] = round(float(auc), 2)
                print(f"[soc] {method:7s} {ds:12s} seed {seed}  AUROC={auc:5.1f}  ({time.time()-t:.0f}s)", flush=True)
            except Exception as e:
                done[str(seed)] = None
                print(f"[soc] {method:7s} {ds:12s} seed {seed}  ERROR {str(e)[:80]}", flush=True)
            vals = [v for v in done.values() if v is not None]
            results[key] = {"per_seed": done,
                            "mean": round(float(np.mean(vals)), 1) if vals else None,
                            "std": round(float(np.std(vals)), 1) if vals else None}
            json.dump(results, open(OUT, "w"), indent=2)

print("\n=== self-run social baselines (mean±std, 5 seeds) ===")
for k, v in results.items():
    print(f"{k:22s} {v['mean']} ± {v['std']}")
print("REBENCH_SOCIAL_DONE", flush=True)
