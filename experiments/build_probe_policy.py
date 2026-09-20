"""Decide, from label-free evidence, which deep views may show a probe.

A probe is only evidence if a short run ranks nodes the way the full run does.
That is a rank correlation between two score vectors, so it needs no labels; it
is calibrated once, offline, over the study graphs, and the result is a fixed
policy the pack reads at run time.

Admission rule: a paradigm may show a probe at budget B when its probe-to-full
rank correlation has median >= 0.5 AND minimum >= 0.0 across the calibration
graphs. The median asks that the preview usually tracks its detector; the
minimum asks that it never contradicts it, which is the failure that cost
Tolokers (one-class, full 62.7, 5-epoch probe 43.1, correlation -0.03).

    python experiments/build_probe_policy.py
writes experiments/llm_ablation/probe_policy.json
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "probe_fidelity.json")
OUT = os.path.join(HERE, "llm_ablation", "probe_policy.json")
MEDIAN_MIN, WORST_MIN = 0.5, 0.0

def main():
    d = json.load(open(SRC))
    paradigms = sorted({p for g in d.values() for p in g})
    pol = {"_provenance": (
        "Calibrated by experiments/build_probe_policy.py from experiments/probe_fidelity.json: "
        "the rank correlation between each paradigm's short probe and its own full run, over "
        f"{len(d)} study graphs. This statistic uses no labels. A paradigm is admitted at a "
        f"budget when the correlation has median >= {MEDIAN_MIN} and minimum >= {WORST_MIN} "
        "across those graphs. Absence of a probe is a statement about the preview, never about "
        "the paradigm."),
        "calibration_graphs": sorted(d), "median_min": MEDIAN_MIN, "worst_min": WORST_MIN,
        "paradigms": {}}
    for p in paradigms:
        entry = {}
        for ep in (5, 20):
            v = [g[p][f"probe{ep}_to_full_spearman"] for g in d.values()
                 if p in g and f"probe{ep}_to_full_spearman" in g[p]]
            if not v:
                continue
            entry[f"budget{ep}"] = {"median": round(float(np.median(v)), 3),
                                    "worst": round(float(np.min(v)), 3), "n_graphs": len(v),
                                    "admitted": bool(np.median(v) >= MEDIAN_MIN and np.min(v) >= WORST_MIN)}
        admitted = [ep for ep in (20, 5) if entry.get(f"budget{ep}", {}).get("admitted")]
        entry["show_probe"] = bool(admitted)
        entry["budget"] = admitted[0] if admitted else None
        pol["paradigms"][p] = entry
    json.dump(pol, open(OUT, "w"), indent=2)
    print(f"{'paradigm':18s} {'b5 med/worst':>16s} {'b20 med/worst':>16s}   decision")
    for p, e in pol["paradigms"].items():
        b5, b20 = e.get("budget5", {}), e.get("budget20", {})
        print(f"{p:18s} {b5.get('median','-'):>8}/{b5.get('worst','-'):<7} "
              f"{b20.get('median','-'):>8}/{b20.get('worst','-'):<7}   "
              f"{'show at ' + str(e['budget']) + ' epochs' if e['show_probe'] else 'NO PROBE'}")
    print(f"\nwrote {OUT}")

main()
