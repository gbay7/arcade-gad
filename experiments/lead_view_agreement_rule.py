"""A conservative form of trusting the analyst's lead view.

The plain rule (lead view alone, experiments/lead_view_rule.py) loses the study
benchmark because the gate's composition is worth ~8 AUROC there. This variant
fires only when the analyst's first-named view is ALSO the top-ranked view under
one of the gate's own label-free diagnostics (tail separation, UED tail quality)
among the views the analyst selected: model order and diagnostic order agreeing
is a label-free signal that the analyst found a real lead. Otherwise the shipped
gate runs unchanged.

Reports coverage (how often the rule fires) and the effect, separately for the
benchmark graphs and the held-out graphs, over every stored decision.
Labels are used once per row, to measure.
"""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from experiments.llm_ablation.cache import DatasetReplayer
from arcade.engine.scoring import ScoringEngine
from arcade.paradigms.registry import BUILTIN_PARADIGMS as PARADIGMS

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(HERE, "llm_ablation")
HELDOUT = {"tolokers", "amazon_fraud", "questions"}
ONLY = set(sys.argv[1:])

files = [os.path.join(ABL, "results.jsonl")] + sorted(
    glob.glob(os.path.join(ABL, "results_packv4_fill*.jsonl"))
    + glob.glob(os.path.join(ABL, "results_heldout*.jsonl"))
    + glob.glob(os.path.join(ABL, "results_study_rich.jsonl")))

latest = {}
for f in files:
    tag = ("domain" if "domain" in f else "rich" if "rich" in f else
           "probeall" if "probeall" in f else "study")
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

def diagnostics(rep, sel):
    """tail separation and UED tail quality of each selected view, from the
    same engine calls the gate uses."""
    eng = ScoringEngine()
    for p in PARADIGMS:
        eng.register_paradigm(p)
    for n in sel:
        eng.set_scores(n, rep.scores[n])
    ued = eng.compute_ued_all()
    ts = {n: float(ScoringEngine.tail_separation(rep.scores[n])) for n in sel}
    return ts, {n: float(ued.get(n, float("nan"))) for n in sel}

reps, out = {}, []
for r in sorted(latest.values(), key=lambda r: (r["dataset"], r["pack"], r["model"], r.get("repeat", 0))):
    ds = r["dataset"]
    if ONLY and ds not in ONLY:
        continue
    if ds not in reps:
        try:
            reps[ds] = DatasetReplayer(ds)
        except Exception as e:
            print(f"[skip] {ds}: {e}", flush=True); reps[ds] = None
    rep = reps[ds]
    if rep is None:
        continue
    sel = [v for v in r["selected"] if v in rep.scores]
    if not sel:
        continue
    shipped = rep.replay(sel)
    lead_only = rep.replay(sel[:1])
    if shipped is None or lead_only is None:
        continue
    ts, ued = diagnostics(rep, sel)
    lead = sel[0]
    fires = (lead == max(ts, key=ts.get)) or (lead == max(ued, key=ued.get))
    a = rep.measure(shipped.fused)["auroc"]
    b = rep.measure(lead_only.fused)["auroc"]
    out.append({"pack": r["pack"], "model": r["model"], "dataset": ds, "repeat": r.get("repeat", 0),
                "selected": sel, "lead": lead, "fires": bool(fires),
                "shipped_auroc": round(a, 1), "rule_auroc": round(b if fires else a, 1),
                "lead_only_auroc": round(b, 1), "held_out": ds in HELDOUT})

def block(title, rs):
    if not rs:
        return None
    per = {}
    print(f"\n=== {title} ({len(rs)} decisions, rule fires on "
          f"{sum(r['fires'] for r in rs)}) ===")
    for ds in sorted({o['dataset'] for o in rs}):
        d = [o for o in rs if o['dataset'] == ds]
        a = float(np.mean([o['shipped_auroc'] for o in d]))
        b = float(np.mean([o['rule_auroc'] for o in d]))
        per[ds] = [round(a, 1), round(b, 1), sum(o['fires'] for o in d), len(d)]
        print(f"  {ds:14s} n={len(d):3d} fires={sum(o['fires'] for o in d):3d}  "
              f"shipped {a:5.1f}  with-rule {b:5.1f}  {b-a:+5.1f}")
    a = float(np.mean([o['shipped_auroc'] for o in rs])); b = float(np.mean([o['rule_auroc'] for o in rs]))
    worse = sum(1 for ds, v in per.items() if v[1] < v[0])
    print(f"  {'MEAN':14s} n={len(rs):3d}  shipped {a:5.1f}  with-rule {b:5.1f}  {b-a:+5.1f}  "
          f"(worse on {worse}/{len(per)} graphs)")
    return {"per_dataset": per, "mean_shipped": round(a, 1), "mean_rule": round(b, 1),
            "n": len(rs), "graphs_worse": worse}

summary = {"benchmark": block("benchmark graphs", [o for o in out if not o["held_out"]]),
           "heldout": block("held-out graphs", [o for o in out if o["held_out"]])}
json.dump({"summary": summary, "decisions": out},
          open(os.path.join(HERE, "lead_view_agreement_rule.json"), "w"), indent=2)
print("\nAGREEMENT_RULE_DONE", flush=True)
