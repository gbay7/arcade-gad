"""Label-free diagnostics (tail separation, UED tail quality) of three conformity
formulas slotted into the conformity seat, on the two graphs where the trigger
fires. Labels only evaluate. CPU-only so it can run beside GPU jobs."""
import sys, numpy as np, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, '.')
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from arcade.data.loader import GraphStore
from arcade.paradigms.detectors import run_detector, _normalize
from arcade.engine.scoring import ScoringEngine
from arcade.paradigms.registry import BUILTIN_PARADIGMS as P
from experiments.llm_ablation.cache import DatasetReplayer
for ds in ["enron", "tolokers"]:
    s = GraphStore(); s.load_dataset(ds); y = (np.asarray(s.labels) > 0).astype(int)
    rep = DatasetReplayer(ds)
    X = s.data.x.numpy(); Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    cands = {"conformity_svd": rep.scores["conformity"],
             "neg_linear_residual": _normalize(-np.asarray(run_detector("linear_residual_detector", s, seed=1), float)),
             "neg_isolation_forest": _normalize(IsolationForest(n_estimators=100, random_state=1).fit(Xs).score_samples(Xs))}
    eng = ScoringEngine()
    for p in P: eng.register_paradigm(p)
    for n in rep.available:
        if n != "conformity": eng.set_scores(n, rep.scores[n])
    print(f"== {ds}", flush=True)
    for name, sc in cands.items():
        eng.set_scores("conformity", sc)
        ts = ScoringEngine.tail_separation(sc); ued = eng.compute_ued_all().get("conformity", float('nan'))
        print(f"  {name:22s} AUROC={roc_auc_score(y, sc)*100:5.1f}  tail_sep={ts:.3f}  UED={ued:.3f}", flush=True)
print("DIAG_DONE", flush=True)
