"""Non-LLM automated selection policy: rank the applicable
paradigms by the label-free UED tail-quality statistic the gate itself uses
(Idan 2024) and select the top-k, replayed through the genuine gate on the same
caches as every other arm. Also the tail-separation top-1/top-3 rows for
reference. Writes ued_policy.json."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from arcade.engine.scoring import ScoringEngine
from arcade.paradigms.registry import BUILTIN_PARADIGMS as PARADIGMS
from experiments.llm_ablation.cache import DatasetReplayer

DS = ["enron","books","disney","weibo","inj_amazon","acm","blogcatalog","cola_flickr"]
out = {}
for ds in DS:
    rep = DatasetReplayer(ds)
    eng = ScoringEngine()
    for p in PARADIGMS: eng.register_paradigm(p)
    for n in rep.available: eng.set_scores(n, rep.scores[n])
    ued = eng.compute_ued_all()
    order = sorted(rep.available, key=lambda n: -ued.get(n, -1))
    row = {"ued": {n: round(float(ued.get(n, float('nan'))), 4) for n in rep.available}}
    for k in (1, 2, 3):
        sel = order[:k]
        row[f"ued_top{k}"] = {"selected": sel, "auroc": rep.measure(rep.replay(sel).fused)["auroc"]}
    out[ds] = row
    print(ds, {k: row[f"ued_top{k}"]["auroc"] for k in (1,2,3)}, "top3:", order[:3], flush=True)
for k in (1, 2, 3):
    out[f"ued_top{k}_avg"] = round(float(np.mean([out[ds][f"ued_top{k}"]["auroc"] for ds in DS])), 1)
json.dump(out, open(f"{HERE}/ued_policy.json", "w"), indent=2)
print("UED avgs:", {k: out[f"ued_top{k}_avg"] for k in (1,2,3)})
print("UED_DONE", flush=True)
