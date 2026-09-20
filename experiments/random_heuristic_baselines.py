"""Can heuristic or random selection beat the LLM analyst?

Exhaustively enumerates EVERY possible 1-4 paradigm selection per dataset and
replays it through the genuine label-free gate (same score caches, same replay
path as the LLM arms in the pack-v4 analyst study). From the exact AUROC
distribution of all subsets we get, with no sampling error:

  - E[random]     expected AUROC of a uniformly random 1-4 selection
                  (the random agent under the same decision contract);
  - random3       expected AUROC of a random 3-subset (classic baseline arm);
  - max draw      the luckiest possible selection (upper bound for ANY policy);
  - P(>= qwen)    probability a single random draw matches the best LLM arm.

Label-free heuristics (computable without an LLM):
  - all_applicable   run everything, gate composes (the paper's system mode);
  - closed_form      triggered closed-form paradigms only;
  - tailsep_top1/3   select by tail separation of the cached score distribution
                     (the "obvious" statistic; the evidence ablation already
                     showed it is a false ranking signal).

Hindsight bound (labels used to PICK, disclosed as oracle):
  - best_fixed3      the single 3-subset with the best mean AUROC across all 8
                     datasets when applied everywhere (intersected with each
                     dataset's applicable set). If even this loses to the LLM,
                     no static selection rule can win.

Reference: per-dataset mean AUROC of each pack-v4 LLM arm (3 repeats).
"""
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arcade.engine.scoring import ScoringEngine
from experiments.llm_ablation.cache import DatasetReplayer

DATASETS = ["enron", "books", "disney", "weibo", "inj_amazon",
            "acm", "blogcatalog", "cola_flickr"]
CLOSED_FORM = ["conformity", "linear_residual", "peripheral_mismatch",
               "attribute_inflation", "clique_density"]
PACKV4 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "llm_ablation", "results_packv4.jsonl")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "random_heuristic_baselines.json")


def llm_reference() -> dict:
    """Per-dataset mean AUROC per model from the pack-v4 study (keep-latest
    per (model, dataset, repeat), mean over repeats)."""
    latest = {}
    for line in open(PACKV4):
        r = json.loads(line)
        if r.get("kind") != "llm":
            continue
        latest[(r["model"], r["dataset"], r["repeat"])] = r
    per = {}
    for (model, ds, _), r in latest.items():
        if r.get("executed") and r.get("auroc") is not None:
            per.setdefault(model, {}).setdefault(ds, []).append(r["auroc"])
    return {m: {ds: round(float(np.mean(v)), 1) for ds, v in d.items()}
            for m, d in per.items()}


def main() -> None:
    ref = llm_reference()
    qwen = ref.get("qwen3:8b", {})
    results = {"datasets": {}, "llm_reference": ref}
    subset_table = {}   # dataset -> {frozenset: auroc} for the fixed-policy scan
    avail_by_ds = {}

    for ds in DATASETS:
        t0 = time.time()
        rep = DatasetReplayer(ds)
        avail = rep.available
        avail_by_ds[ds] = set(avail)
        aurocs = {}
        for k in (1, 2, 3, 4):
            for combo in itertools.combinations(avail, k):
                out = rep.replay(list(combo))
                aurocs[frozenset(combo)] = rep.measure(out.fused)["auroc"]
        subset_table[ds] = aurocs

        by_k = {k: [a for s, a in aurocs.items() if len(s) == k] for k in (1, 2, 3, 4)}
        uniform = [float(np.mean(by_k[k])) for k in (1, 2, 3, 4)]
        all_draws = np.array(list(aurocs.values()))
        best_set = max(aurocs, key=aurocs.get)

        tails = {n: ScoringEngine.tail_separation(rep.scores[n]) for n in avail}
        by_tail = sorted(avail, key=lambda n: -tails[n])
        cf = [n for n in avail if n in CLOSED_FORM]

        q = qwen.get(ds)
        # random draw: uniform k in 1..4, then uniform subset of that size
        p_beat = float(np.mean([np.mean([a >= q for a in by_k[k]]) for k in (1, 2, 3, 4)])) if q else None

        results["datasets"][ds] = {
            "n_applicable": len(avail),
            "n_subsets": len(aurocs),
            "random_uniform_1to4": round(float(np.mean(uniform)), 1),
            "random3": round(float(np.mean(by_k[3])), 1),
            "random_by_k": {k: round(float(np.mean(v)), 1) for k, v in by_k.items()},
            "max_draw": {"auroc": aurocs[best_set], "selected": sorted(best_set)},
            "min_draw": round(float(all_draws.min()), 1),
            "all_applicable": rep.measure(rep.replay(avail).fused)["auroc"],
            "closed_form": (rep.measure(rep.replay(cf).fused)["auroc"] if cf else None),
            "closed_form_sel": cf,
            "tailsep_top1": {"auroc": aurocs[frozenset(by_tail[:1])], "selected": by_tail[:1]},
            "tailsep_top3": {"auroc": aurocs[frozenset(by_tail[:3])], "selected": by_tail[:3]},
            "qwen3_8b": q,
            "p_random_draw_ge_qwen": (round(p_beat, 3) if q is not None else None),
        }
        print(f"[{ds}] {len(aurocs)} subsets in {time.time()-t0:.0f}s  "
              f"E[rand]={results['datasets'][ds]['random_uniform_1to4']}  "
              f"max={aurocs[best_set]}  qwen={q}", flush=True)

    # best FIXED 3-subset across datasets (hindsight/oracle, disclosed).
    # A fixed policy names 3 paradigms and, on each graph, runs whichever of
    # them are applicable there; graphs where none apply score chance (50).
    universe = sorted({n for ds in DATASETS for n in avail_by_ds[ds]})
    best_fixed, best_mean = None, -1.0
    for combo in itertools.combinations(universe, 3):
        vals = []
        for ds in DATASETS:
            inter = frozenset(c for c in combo if c in avail_by_ds[ds])
            vals.append(subset_table[ds][inter] if inter else 50.0)
        m = float(np.mean(vals))
        if m > best_mean:
            best_mean, best_fixed = m, combo
    results["best_fixed3_hindsight"] = {
        "selected": list(best_fixed) if best_fixed else None,
        "mean_auroc": round(best_mean, 1) if best_fixed else None,
        "note": "labels used to pick the subset; upper bound for static rules",
    }

    # arm averages over the 8 datasets
    d = results["datasets"]
    results["summary"] = {
        "E_random_1to4_avg": round(float(np.mean([d[x]["random_uniform_1to4"] for x in DATASETS])), 1),
        "E_random3_avg": round(float(np.mean([d[x]["random3"] for x in DATASETS])), 1),
        "max_draw_avg": round(float(np.mean([d[x]["max_draw"]["auroc"] for x in DATASETS])), 1),
        "all_applicable_avg": round(float(np.mean([d[x]["all_applicable"] for x in DATASETS])), 1),
        "closed_form_avg": round(float(np.mean([d[x]["closed_form"] or 50.0 for x in DATASETS])), 1),
        "tailsep_top1_avg": round(float(np.mean([d[x]["tailsep_top1"]["auroc"] for x in DATASETS])), 1),
        "tailsep_top3_avg": round(float(np.mean([d[x]["tailsep_top3"]["auroc"] for x in DATASETS])), 1),
        "qwen3_8b_avg": round(float(np.mean([d[x]["qwen3_8b"] for x in DATASETS])), 1),
    }
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("SUMMARY " + json.dumps(results["summary"]))
    print("BASELINES_DONE", flush=True)


if __name__ == "__main__":
    main()
