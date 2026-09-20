"""Would a 'trust the analyst's lead view alone' rule ship?

Every stored analyst selection is re-executed through the frozen gate twice:
as shipped (the full selection, gate decides dominant/pairmax/AOM), and with
only the first-named view. The analyst names views in its own order of
confidence, so the lead view is a label-free quantity the gate could read.
Benchmark graphs and the two held-out graphs are reported separately.
Labels are used once per row, to measure.
"""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from experiments.llm_ablation.cache import DatasetReplayer

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(HERE, "llm_ablation")
HELDOUT = {"tolokers", "amazon_fraud", "questions"}

files = [os.path.join(ABL, "results.jsonl")] + sorted(
    glob.glob(os.path.join(ABL, "results_packv4_fill*.jsonl"))
    + glob.glob(os.path.join(ABL, "results_heldout*.jsonl")))

latest = {}
for f in files:
    tag = "study" if f.endswith(("results.jsonl",)) or "packv4" in f else (
        "domain" if "domain" in f else ("probeall" if "probeall" in f else "study"))
    for line in open(f):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "llm" or not r.get("selected"):
            continue
        t = r.get("temperature")
        if not (t in (0, 0.0) or (t is None and r["model"].startswith("claude:"))):
            continue
        r["pack"] = tag
        latest[(tag, r["model"], r["dataset"], r.get("repeat", 0))] = r
rows = list(latest.values())

reps, out = {}, []
for r in sorted(rows, key=lambda r: (r["dataset"], r["pack"], r["model"], r.get("repeat", 0))):
    ds = r["dataset"]
    if ds not in reps:
        try:
            reps[ds] = DatasetReplayer(ds)
        except Exception as e:                       # no cached scores for this graph
            print(f"[skip] {ds}: {e}", flush=True); reps[ds] = None
    rep = reps[ds]
    if rep is None:
        continue
    sel = [v for v in r["selected"] if v in rep.scores]
    if not sel:
        continue
    full = rep.replay(sel)
    lead = rep.replay(sel[:1])
    if full is None or lead is None:
        continue
    out.append({"pack": r["pack"], "model": r["model"], "dataset": ds, "repeat": r.get("repeat", 0),
                "selected": sel, "lead": sel[0],
                "shipped_auroc": round(rep.measure(full.fused)["auroc"], 1),
                "shipped_mode": full.fusion_mode,
                "lead_only_auroc": round(rep.measure(lead.fused)["auroc"], 1),
                "held_out": ds in HELDOUT})

def block(title, rs):
    if not rs:
        return None
    print(f"\n=== {title} ({len(rs)} decisions) ===")
    per = {}
    for ds in sorted({o['dataset'] for o in rs}):
        d = [o for o in rs if o['dataset'] == ds]
        a = float(np.mean([o['shipped_auroc'] for o in d]))
        b = float(np.mean([o['lead_only_auroc'] for o in d]))
        per[ds] = [round(a, 1), round(b, 1)]
        print(f"  {ds:14s} n={len(d):3d}  shipped {a:5.1f}  lead-only {b:5.1f}  {b-a:+5.1f}")
    a = float(np.mean([o['shipped_auroc'] for o in rs])); b = float(np.mean([o['lead_only_auroc'] for o in rs]))
    worse = sum(1 for ds, (x, y) in per.items() if y < x)
    print(f"  {'MEAN':14s} n={len(rs):3d}  shipped {a:5.1f}  lead-only {b:5.1f}  {b-a:+5.1f}   "
          f"(lead-only worse on {worse}/{len(per)} graphs)")
    return {"per_dataset": per, "mean_shipped": round(a, 1), "mean_lead_only": round(b, 1),
            "n": len(rs), "graphs_worse": worse, "graphs": len(per)}

summary = {"benchmark": block("benchmark graphs", [o for o in out if not o["held_out"]]),
           "heldout": block("held-out graphs", [o for o in out if o["held_out"]]),
           "heldout_domain": block("held-out graphs, domain-informed pack",
                                   [o for o in out if o["held_out"] and o["pack"] == "domain"])}
json.dump({"summary": summary, "decisions": out}, open(os.path.join(HERE, "lead_view_rule.json"), "w"), indent=2)
print("\nLEAD_RULE_DONE", flush=True)
