"""Diagnosis prompted by the held-out Tolokers graph: is the conformity formula
(negative truncated-SVD residual) the weak link, and would a negative structured
linear residual (Radar-style) or a negative isolation-forest score be a better
general conformity formula? Labels are used ONLY to evaluate each variant."""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, '.')
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from arcade.data.loader import GraphStore
from arcade.paradigms.detectors import run_detector
from experiments.llm_ablation.cache import DatasetReplayer
print(f"{'dataset':12s} {'conformity':>10s} {'neg.linres':>10s} {'neg.IF':>8s} {'dup':>5s} {'dim':>5s}", flush=True)
for ds in ["enron", "reddit", "books", "disney", "weibo", "inj_cora", "inj_amazon", "tolokers"]:
    s = GraphStore(); s.load_dataset(ds); y = (np.asarray(s.labels) > 0).astype(int); p = s.profile()
    rep = DatasetReplayer(ds)
    def auc(name):
        sc = rep.scores[name] if name in rep.scores else np.asarray(run_detector(name + "_detector", s, seed=1), float)
        return roc_auc_score(y, sc) * 100
    conf = auc("conformity"); lin = 100 - auc("linear_residual")
    X = s.data.x.numpy(); Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    negif = 100 - roc_auc_score(y, -IsolationForest(n_estimators=100, random_state=1).fit(Xs).score_samples(Xs)) * 100
    print(f"{ds:12s} {conf:10.1f} {lin:10.1f} {negif:8.1f} {p.feature_dup_rate:5.2f} {p.feature_dim:5d}", flush=True)
print("VARIANTS_DONE", flush=True)
