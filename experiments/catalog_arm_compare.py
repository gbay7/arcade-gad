"""Compare the evidence-interface arms decision by decision.

Arms differ only in the text of the evidence pack, never in the library, the
preconditions, the gate or the cached scores:
  study        the shipped pack (one-line paradigm descriptions, bare profile)
  probeall     study pack plus a 5-epoch probe of every deep paradigm
  domain       study pack plus a source-cited description of the graph and task
  rich         enriched catalog (what each view computes, its reference frame,
               its inputs, when its formula degenerates) plus a legend defining
               every profile and landmarker field
  rich+domain  both

For each arm it reports the executed AUROC (the selection replayed through the
frozen gate), which views were selected, and the AUROC each selected view
reaches alone. Labels are used only to measure.
    python experiments/catalog_arm_compare.py tolokers amazon_fraud
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from experiments.llm_ablation.cache import DatasetReplayer

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(HERE, "llm_ablation")
ARMS = [("study", "results_heldout.jsonl"), ("probeall", "results_heldout_probeall.jsonl"),
        ("domain", "results_heldout_domain.jsonl"), ("rich", "results_heldout_rich.jsonl"),
        ("rich+domain", "results_heldout_rich_domain.jsonl")]
datasets = sys.argv[1:] or ["tolokers", "amazon_fraud"]
out = {}
for ds in datasets:
    try:
        rep = DatasetReplayer(ds)
    except Exception as e:
        print(f"[skip] {ds}: {e}"); continue
    single = {n: round(rep.measure(rep.scores[n])["auroc"], 1) for n in rep.available}
    print(f"\n######## {ds}   (best single view: "
          f"{max(single, key=single.get)} {max(single.values())})")
    print("  single views:", dict(sorted(single.items(), key=lambda kv: -kv[1])))
    rows = []
    for arm, fname in ARMS:
        path = os.path.join(ABL, fname)
        if not os.path.exists(path):
            continue
        recs = {}
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "llm" or r.get("dataset") != ds or not r.get("selected"):
                continue
            recs[(r["model"], r.get("repeat", 0))] = r
        if not recs:
            continue
        print(f"\n  == arm: {arm}")
        by_model = {}
        for (model, rp), r in sorted(recs.items()):
            sel = [v for v in r["selected"] if v in rep.scores]
            res = rep.replay(sel)
            auroc = round(rep.measure(res.fused)["auroc"], 1) if res else None
            by_model.setdefault(model, []).append(auroc)
            lead = sel[0] if sel else None
            print(f"     {model:14s} r{rp} {auroc:5.1f}  mode={res.fusion_mode if res else '-':8s} "
                  f"lead={lead}({single.get(lead)})  sel={[f'{v}:{single[v]}' for v in sel]}")
            rows.append({"arm": arm, "model": model, "repeat": rp, "auroc": auroc,
                         "selected": sel, "lead": lead,
                         "mode": res.fusion_mode if res else None,
                         "per_view": {v: single[v] for v in sel}})
        for m, a in by_model.items():
            print(f"     {m:14s} MEAN {np.mean([x for x in a if x is not None]):5.1f}")
    out[ds] = {"single_view": single, "decisions": rows}
json.dump(out, open(os.path.join(HERE, "catalog_arm_compare.json"), "w"), indent=2)
print("\nARM_COMPARE_DONE", flush=True)
