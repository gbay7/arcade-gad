"""Does a cheap probe predict the detector it previews?

The evidence pack shows the analyst a 5-epoch probe of a deep view and presents
it next to the fast paradigms' landmarkers, as if the two were the same kind of
evidence. A 5-epoch preview of a 100-epoch detector is a different object: the
one-class centre is still moving that early, and a score distribution cannot show
sign in any case. This measures the thing that decides whether a probe is
evidence at all:

  - rank correlation between the probe and the SAME paradigm's full run, from
    the shipped seed-1 cache (label-free quantity, the one the analyst is
    implicitly trusting);
  - the probe's own AUROC against the full run's AUROC (labels, to measure);
  - a second budget (20 epochs) so the probe can be asked whether its ranking is
    still moving, which is the convergence signal a single snapshot cannot give.

    python experiments/probe_fidelity.py [graphs...]
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import roc_auc_score
from arcade.data.loader import GraphStore
from arcade.paradigms.detectors import run_detector
from arcade.paradigms.registry import BUILTIN_PARADIGMS as PARADIGMS
from experiments.llm_ablation.cache import DatasetReplayer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "probe_fidelity.json")
VEC = os.path.join(HERE, "probe_scores.npz")      # probe vectors, for later statistics
# subgraph_contrast is the ONLY probe the shipped pack carries, so its fidelity
# is the one the paper's current evidence rests on. It is probed the way the
# runner probes it (few epochs AND few test rounds), not merely with fewer epochs.
DEEP = ["reconstruction", "contrastive", "adversarial", "one_class", "subgraph_contrast"]
SUBGRAPH_ROUNDS = {5: 4, 20: 8}
BUDGETS = [5, 20]
DEFAULT = ["disney", "books", "enron", "reddit", "inj_cora", "weibo", "inj_amazon", "tolokers"]

def ranks(v):
    o = np.argsort(np.argsort(np.asarray(v, dtype=float)))
    return o / max(len(o) - 1, 1)

def spearman(a, b):
    a, b = ranks(a), ranks(b)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

def tail_separation(v):
    """The gate's own shape statistic, which is what the analyst actually reads
    off a landmarker block. Label-free."""
    v = np.asarray(v, dtype=float)
    if len(v) < 20:
        return 0.0
    p90 = np.percentile(v, 90)
    tail, body = v[v > p90], v[v <= p90]
    if len(tail) < 2 or len(body) < 2:
        return 0.0
    return float((tail.mean() - body.mean()) / (body.std() + 1e-9))

def main():
    graphs = sys.argv[1:] or DEFAULT
    vecs = dict(np.load(VEC)) if os.path.exists(VEC) else {}
    byname = {p.name: p for p in PARADIGMS}
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for ds in graphs:
        rep = DatasetReplayer(ds)
        y = (np.asarray(rep.labels) > 0).astype(int)
        store = GraphStore(); store.load_dataset(ds)
        res.setdefault(ds, {})
        for name in DEEP:
            if name not in rep.scores or name in res[ds]:
                continue
            full = np.asarray(rep.scores[name], dtype=float)
            row = {"full_auroc": round(roc_auc_score(y, full) * 100, 1)}
            probes = {}
            for ep in BUDGETS:
                t0 = time.time()
                kw = {"epochs": ep}
                if name == "subgraph_contrast":
                    kw["test_rounds"] = SUBGRAPH_ROUNDS[ep]
                try:
                    p = run_detector(byname[name].detector_name, store, seed=1, **kw)
                except TypeError:
                    p = run_detector(byname[name].detector_name, store, seed=1)
                p = np.asarray(p, dtype=float)
                probes[ep] = p
                row[f"probe{ep}_auroc"] = round(roc_auc_score(y, p) * 100, 1)
                row[f"probe{ep}_to_full_spearman"] = round(spearman(p, full), 3)
                row[f"probe{ep}_tail_sep"] = round(tail_separation(p), 3)
                row[f"probe{ep}_seconds"] = round(time.time() - t0, 1)
                vecs[f"{ds}__{name}__{ep}"] = p.astype(np.float32)
            row["budget5_to_budget20_spearman"] = round(spearman(probes[5], probes[20]), 3)
            row["full_tail_sep"] = round(tail_separation(full), 3)
            res[ds][name] = row
            print(f"[probe] {ds:12s} {name:16s} full {row['full_auroc']:5.1f} | "
                  f"p5 {row['probe5_auroc']:5.1f} rho {row['probe5_to_full_spearman']:+.2f} | "
                  f"p20 {row['probe20_auroc']:5.1f} rho {row['probe20_to_full_spearman']:+.2f} | "
                  f"5-vs-20 rho {row['budget5_to_budget20_spearman']:+.2f}", flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
            np.savez_compressed(VEC, **vecs)
        del store, rep
    print("PROBE_FIDELITY_DONE", flush=True)

main()
