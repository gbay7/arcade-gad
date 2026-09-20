"""Replay every stored analyst decision on amazon_fraud through the frozen gate and
report, per decision: the gate mode, the executed AUROC, each selected view's own
AUROC, and the best single selected view. Labels only measure. CPU-only."""
import sys, json, numpy as np, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, '.')
from sklearn.metrics import roc_auc_score, average_precision_score
from arcade.data.loader import GraphStore
from arcade.engine.scoring import ScoringEngine
from arcade.paradigms.registry import BUILTIN_PARADIGMS as P
from experiments.llm_ablation.cache import DatasetReplayer

DS = "amazon_fraud"
s = GraphStore(); s.load_dataset(DS); y = (np.asarray(s.labels) > 0).astype(int)
rep = DatasetReplayer(DS)
single = {n: (roc_auc_score(y, rep.scores[n]) * 100, average_precision_score(y, rep.scores[n]) * 100)
          for n in rep.available}
out = {"dataset": DS, "n_nodes": int(len(y)), "anomaly_rate": round(float(y.mean() * 100), 1),
       "single_view": {k: [round(v[0], 1), round(v[1], 1)] for k, v in sorted(single.items(), key=lambda kv: -kv[1][0])},
       "decisions": []}
print("== single views (AUROC / AP)")
for n, (a, p) in sorted(single.items(), key=lambda kv: -kv[1][0]):
    print(f"  {n:22s} {a:5.1f}  {p:4.1f}")

def gate(sel):
    eng = ScoringEngine()
    for p in P: eng.register_paradigm(p)
    for n in sel: eng.set_scores(n, rep.scores[n])
    res = eng.fuse(list(sel)) if hasattr(eng, "fuse") else None
    return eng, res

rows = []
for f, pack in [("experiments/llm_ablation/results_heldout.jsonl", "study"),
                ("experiments/llm_ablation/results_heldout_domain.jsonl", "domain")]:
    for l in open(f):
        d = json.loads(l)
        if d.get("dataset") != DS: continue
        sel = [x for x in d["selected"] if x in rep.scores]
        best = max(sel, key=lambda n: single[n][0]) if sel else None
        rows.append({"pack": pack, "model": d["model"], "repeat": d["repeat"], "selected": d["selected"],
                     "executed_auroc": d.get("auroc"),
                     "per_view": {n: round(single[n][0], 1) for n in sel},
                     "best_selected": best, "best_selected_auroc": round(single[best][0], 1) if best else None})
out["decisions"] = rows
print("\n== decisions")
for r in rows:
    print(f"  [{r['pack']:6s} {r['model']:13s} r{r['repeat']}] exec={r['executed_auroc']:5.1f}  "
          f"best_sel={r['best_selected']}({r['best_selected_auroc']})  views={r['per_view']}")
json.dump(out, open("experiments/amazon_decision_replay.json", "w"), indent=2)
print("\nAMAZON_REPLAY_DONE", flush=True)
