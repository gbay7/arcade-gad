"""Aggregation arms for the analyst study: does combining the 10 analysts
substitute for a good one?

  - median-of-analysts: per dataset, the median of the 10 models' mean AUROC
    (no winner elected, no replay needed);
  - majority-vote: paradigms selected in >= 50% of all stored pack-v4 decisions
    on that dataset (top-1 most-voted if none reach half), replayed through the
    genuine label-free gate.

Both fall well below run-everything (83.1) and the best analysts, and fail
exactly on the counterintuitive-mechanism benchmarks: the majority misses
those, and consensus inherits the majority's error.

Writes analyst_aggregation.json.
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from experiments.llm_ablation.cache import DatasetReplayer

DS = ["enron", "books", "disney", "weibo", "inj_amazon",
      "acm", "blogcatalog", "cola_flickr"]


def main():
    latest = {}
    for line in open(f"{HERE}/llm_ablation/results_packv4.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "llm":
            latest[(r["model"], r["dataset"], r["repeat"])] = r

    per_model = collections.defaultdict(dict)
    votes = collections.defaultdict(collections.Counter)
    counts = collections.defaultdict(int)
    for (m, ds, _), r in latest.items():
        if ds not in DS or not r.get("executed"):
            continue
        per_model[ds].setdefault(m, []).append(r["auroc"])
        for p in r["selected"]:
            votes[ds][p] += 1
        counts[ds] += 1

    out = {"median_of_analysts": {}, "majority_vote": {}}
    for ds in DS:
        out["median_of_analysts"][ds] = round(float(np.median(
            [np.mean(v) for v in per_model[ds].values()])), 1)
        sel = [p for p, c in votes[ds].items() if c >= counts[ds] / 2]
        if not sel:
            sel = [votes[ds].most_common(1)[0][0]]
        rep = DatasetReplayer(ds)
        out["majority_vote"][ds] = {
            "selected": sorted(sel),
            "auroc": rep.measure(rep.replay(sel).fused)["auroc"],
        }

    out["median_avg"] = round(float(np.mean(list(out["median_of_analysts"].values()))), 1)
    out["vote_avg"] = round(float(np.mean(
        [v["auroc"] for v in out["majority_vote"].values()])), 1)
    json.dump(out, open(f"{HERE}/analyst_aggregation.json", "w"), indent=2)
    print(f"median-of-analysts avg {out['median_avg']}, majority-vote avg {out['vote_avg']}")
    print("AGGREGATION_DONE")


if __name__ == "__main__":
    main()
