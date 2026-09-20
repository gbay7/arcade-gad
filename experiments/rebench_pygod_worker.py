"""Worker: run ONE (dataset, method) over 5 seeds and print a JSON line.
Isolated in its own process so a CUDA device-side assert (e.g. GAAN on Amazon)
kills only this worker, not the whole re-benchmark.
Usage: python rebench_pygod_worker.py <dataset_key> <Method>
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([GlobalStorage, NodeStorage, EdgeStorage, Data])
from arcade.data.loader import GraphStore
import pygod.detector as PD

key, method = sys.argv[1], sys.argv[2]
GPU = 0 if torch.cuda.is_available() else -1
store = GraphStore(); store.load_dataset(key)
y = (np.asarray(store.labels) > 0).astype(int)
n = store.data.num_nodes


def build():
    cls = getattr(PD, method)
    kw = dict(gpu=GPU, epoch=200)
    if method == "GAAN":
        kw.update(batch_size=(2048 if n > 2000 else 0), num_neigh=10)
    elif n * store.data.x.shape[1] > 2e7:
        # high-dimensional social graphs: full-batch decoding exceeds GPU
        # memory for several detectors; batch fitting where needed
        kw.update(batch_size=2048)
    # drop keyword arguments a detector does not accept (e.g. Radar takes
    # neither epoch nor batch_size), keeping library defaults for the rest
    for _ in range(3):
        try:
            return cls(**kw)
        except TypeError as e:
            bad = str(e).split("'")[1] if "'" in str(e) else None
            if bad not in kw:
                raise
            kw.pop(bad)
    return cls(**kw)


from sklearn.metrics import average_precision_score

def fit_with_fallback(sd):
    """GPU full config -> smaller batch -> CPU: the same detector, degraded
    only in fitting granularity, so a memory limit yields a number instead of
    a missing cell."""
    torch.manual_seed(sd); np.random.seed(sd)
    for attempt in ("default", "batch512", "cpu"):
        try:
            det = build()
            if attempt == "batch512" and hasattr(det, "batch_size"):
                det.batch_size = 512
            if attempt == "cpu":
                det.device = "cpu"; det.gpu = -1
            det.fit(store.data)
            return det
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
    raise torch.cuda.OutOfMemoryError("all fallbacks exhausted")


aucs, aps = [], []
for sd in [0, 1, 2, 3, 4]:
    det = fit_with_fallback(sd)
    s = det.decision_score_
    s = s.cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
    if np.isnan(s).any():
        raise ValueError("nan")
    aucs.append(roc_auc_score(y, s) * 100)
    aps.append(average_precision_score(y, s) * 100)
    if GPU >= 0:
        torch.cuda.empty_cache()
print("RESULT " + json.dumps({"mean": round(float(np.mean(aucs)), 1),
                              "std": round(float(np.std(aucs)), 1),
                              "ap_mean": round(float(np.mean(aps)), 1),
                              "ap_std": round(float(np.std(aps)), 1), "n": len(aucs)}), flush=True)
