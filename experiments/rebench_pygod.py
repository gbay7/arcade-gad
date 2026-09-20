"""Self-run baseline re-benchmark with variance: 6 PyGOD detectors (DOMINANT,
AnomalyDAE, DONE, CoLA, GAAN, OCGNN) on all 7 datasets, 5 seeds each, reporting
mean+-std AUROC. SL-GAD (the 7th method) is run separately by its own code.
GAAN is mini-batched on large graphs; a method that fails on a graph is recorded
as an error and shown as "/" in the paper.
Output: experiments/rebench_pygod.json
"""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([GlobalStorage, NodeStorage, EdgeStorage, Data])
from arcade.data.loader import GraphStore
import pygod.detector as PD

GPU = 0 if torch.cuda.is_available() else -1
KEYS = {"Enron": "enron", "Reddit": "reddit", "Books": "books", "Disney": "disney",
        "Amazon": "inj_amazon", "Cora": "inj_cora", "Weibo": "weibo"}
METHODS = ["DOMINANT", "AnomalyDAE", "DONE", "CoLA", "GAAN", "OCGNN"]
SEEDS = [0, 1, 2, 3, 4]
HERE = os.path.dirname(os.path.abspath(__file__))


def build(name, n):
    cls = getattr(PD, name)
    kw = dict(gpu=GPU, epoch=200)
    if name == "GAAN":
        kw.update(batch_size=(2048 if n > 2000 else 0), num_neigh=10)
    try:
        return cls(**kw)
    except TypeError:
        kw.pop("epoch", None)
        return cls(**kw)


results = {}
for label, key in KEYS.items():
    store = GraphStore(); store.load_dataset(key)
    y = (np.asarray(store.labels) > 0).astype(int)
    n = store.data.num_nodes
    results[label] = {}
    for m in METHODS:
        aucs = []
        err = None
        for sd in SEEDS:
            torch.manual_seed(sd); np.random.seed(sd)
            try:
                det = build(m, n); det.fit(store.data)
                s = det.decision_score_
                s = s.cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
                if np.isnan(s).any():
                    raise ValueError("nan scores")
                aucs.append(roc_auc_score(y, s) * 100)
            except Exception as e:
                err = str(e)[:60]
            if GPU >= 0:
                torch.cuda.empty_cache()
        if aucs:
            results[label][m] = {"mean": round(float(np.mean(aucs)), 1),
                                 "std": round(float(np.std(aucs)), 1), "n": len(aucs)}
            print(f"[reb] {label:7s} {m:10s} {np.mean(aucs):5.1f}+-{np.std(aucs):.1f} ({len(aucs)}/5)", flush=True)
        else:
            results[label][m] = {"error": err}
            print(f"[reb] {label:7s} {m:10s} ERR {err}", flush=True)
        json.dump(results, open(os.path.join(HERE, "rebench_pygod.json"), "w"), indent=2)
print("REBENCH_PYGOD_DONE", flush=True)
