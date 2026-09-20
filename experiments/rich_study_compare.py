"""Paired check: does the enriched catalog change the study benchmark?

Same model, same graphs, same protocol, same cached scores, same gate; only the
catalog text and the statistics legend differ. Both arms are replayed here so
the two numbers come from one code path. Labels are used once per row.
    python experiments/rich_study_compare.py [model]
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from experiments.llm_ablation.cache import DatasetReplayer

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(HERE, "llm_ablation")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "claude:haiku"

def load(fname, model):
    out = {}
    path = os.path.join(ABL, fname)
    if not os.path.exists(path):
        return out
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "llm" or r.get("model") != model or not r.get("selected"):
            continue
        t = r.get("temperature")
        if not (t in (0, 0.0) or (t is None and model.startswith("claude:"))):
            continue
        out[(r["dataset"], r.get("repeat", 0))] = r["selected"]
    return out

base = load("results.jsonl", MODEL)
for extra in ("results_packv4.jsonl", "results_packv4_fill.jsonl", "results_packv4_fill_opus48.jsonl"):
    base.update(load(extra, MODEL))
rich = load("results_study_rich.jsonl", MODEL)
datasets = sorted({ds for ds, _ in rich})
rows, reps = [], {}
print(f"model {MODEL}: {len(rich)} enriched decisions on {len(datasets)} graphs\n")
print(f"  {'graph':14s} {'shipped':>8s} {'enriched':>9s} {'delta':>7s}   changed selections")
for ds in datasets:
    if ds not in reps:
        reps[ds] = DatasetReplayer(ds)
    rep = reps[ds]
    def score(sel):
        sel = [v for v in sel if v in rep.scores]
        res = rep.replay(sel)
        return rep.measure(res.fused)["auroc"] if res else None
    a = [score(base[k]) for k in sorted(base) if k[0] == ds and base.get(k)]
    b = [score(rich[k]) for k in sorted(rich) if k[0] == ds]
    a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
    if not a or not b:
        continue
    changed = sum(1 for k in sorted(rich) if k[0] == ds and base.get(k) and
                  set(rich[k]) != set(base[k]))
    rows.append({"dataset": ds, "shipped": round(float(np.mean(a)), 1),
                 "enriched": round(float(np.mean(b)), 1), "n_shipped": len(a), "n_enriched": len(b),
                 "selections_changed": changed})
    print(f"  {ds:14s} {np.mean(a):8.1f} {np.mean(b):9.1f} {np.mean(b)-np.mean(a):+7.1f}   "
          f"{changed}/{len([k for k in rich if k[0]==ds])}")
if rows:
    A = float(np.mean([r["shipped"] for r in rows])); B = float(np.mean([r["enriched"] for r in rows]))
    worse = sum(1 for r in rows if r["enriched"] < r["shipped"])
    print(f"  {'MEAN':14s} {A:8.1f} {B:9.1f} {B-A:+7.1f}   enriched worse on {worse}/{len(rows)} graphs")
    json.dump({"model": MODEL, "per_dataset": rows, "mean_shipped": round(A, 1),
               "mean_enriched": round(B, 1), "graphs_worse": worse},
              open(os.path.join(HERE, "rich_study_compare.json"), "w"), indent=2)
print("\nRICH_STUDY_COMPARE_DONE", flush=True)
