"""Render the later result additions as LaTeX/text from the shipped artifacts:
held-out rows for Table II (frozen rules), the Radar column, the 10-benchmark
analyst averages with repeat SD, and the runtime/memory sentence.
Prints what exists; skips what has not landed yet."""
import json, os, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None

METHODS = ["DOMINANT", "AnomalyDAE", "DONE", "CoLA", "GAAN", "OCGNN", "Radar", "SL-GAD"]
def cell(v):
    if not v or "error" in v: return "/"
    return f"{v['mean']:.1f}$\\pm${v['std']:.1f}"

# --- held-out rows
for ds, label in [("tolokers", "Tolokers$^{\\circ}$"), ("questions", "Questions$^{\\circ}$")]:
    h = load(f"heldout_{ds}.json")
    if not h or "ARCADE" not in h: print(f"[{ds}] not ready"); continue
    cells = [cell(h.get(m)) for m in METHODS]
    arc = h["ARCADE"]
    print(f"{label} & " + " & ".join(cells) + f" & / & {arc['mean']:.1f}$\\pm${arc['std']:.1f} \\\\   % frozen commit {h.get('frozen_commit')}")
    best = max(((m, h[m]["mean"]) for m in METHODS if m in h and "error" not in h[m]), key=lambda t: t[1], default=None)
    print(f"   ARCADE {arc['mean']} (AP {arc['ap_mean']}) vs best baseline {best}; mode/trusted per seed: {[ (s['mode'], s['trusted']) for s in arc['seeds']]}")
    print(f"   per-detector (seed 1): {h.get('per_detector_seed1')}")

# --- Radar column for the 10 benchmarks
r = load("rebench_radar.json")
if r:
    order = ["Enron","Reddit","Books","Disney","Weibo","Cora","Amazon","BlogCatalog","ACM","Flickr"]
    print("Radar column:", {k: cell(r.get(k)) for k in order})
    vals = [r[k]["mean"] for k in order if k in r and "error" not in r[k]]
    print(f"Radar average {np.mean(vals):.1f} over {len(vals)}")

# --- analyst study on all 10
rows = []
for f in ["llm_ablation/results_packv4.jsonl", "llm_ablation/results_packv4_fill.jsonl",
          "llm_ablation/results_packv4_fill_opus48.jsonl"]:
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        rows += [json.loads(l) for l in open(p)]
latest = {}
for x in rows:
    if x.get("kind") == "llm":
        # the CLI alias "opus" resolves to claude-opus-5 since September; the study's
        # opus row is claude-opus-4-8, so the explicit re-run replaces the alias fill
        model = "claude:opus" if x["model"] == "claude:claude-opus-4-8" else x["model"]
        if x["model"] == "claude:opus" and x["dataset"] in ("reddit", "inj_cora"):
            continue
        latest[(model, x["dataset"], x["repeat"])] = x
per = collections.defaultdict(lambda: collections.defaultdict(list))
for (m, ds, _), x in latest.items():
    if x.get("executed"): per[m][ds].append(x["auroc"])
allappl = {"enron":83.7,"reddit":59.8,"books":72.1,"disney":86.4,"weibo":94.9,"inj_cora":93.6,"inj_amazon":73.3,"acm":98.9,"blogcatalog":78.9,"cola_flickr":76.5}
print("\nanalyst: model  n_datasets  avg8  avg10  meanSD")
for m in sorted(per, key=lambda m: -np.mean([np.mean(v) for v in per[m].values()])):
    ds8 = [d for d in per[m] if d not in ("reddit", "inj_cora")]
    a8 = np.mean([np.mean(per[m][d]) for d in ds8]); a10 = np.mean([np.mean(per[m][d]) for d in per[m]])
    sd = np.mean([np.std(per[m][d]) for d in per[m]])
    print(f"  {m:40s} {len(per[m]):2d}  {a8:5.1f}  {a10:5.1f}  {sd:4.1f}")
print("run-everything avg10:", round(float(np.mean(list(allappl.values()))), 1))

# --- runtime
rt = load("runtime_bench.json")
if rt:
    for ds, row in rt.items():
        print(f"runtime {ds}: run-everything {row['run_everything']}  analyst total {row['analyst_total_s']}s "
              f"(evidence {row['analyst_evidence']}, selection {row['analyst_selection']} {row['analyst_selected']}, decision {row['analyst_decision_s']}s)  SL-GAD {row['slgad']}")
