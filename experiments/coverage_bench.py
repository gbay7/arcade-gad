"""Coverage benchmark: ARCADE (full library + label-free gate) over all
datasets, 5 seeds. The composition is NOT selected per dataset — every
applicable paradigm runs and the label-free gate (trigger-trust, artifact
pruning, UED weighting, fuse/select) produces the score. Labels are used once
per seed, to measure. Closed-form detectors are deterministic; variance comes
from the trained paradigms.

    ARCADE_CACHE_SEED=s python -m experiments.llm_ablation.cache <datasets>  # build
    python experiments/coverage_bench.py                                     # replay + table
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

DATASETS = ["enron", "reddit", "books", "disney", "weibo", "inj_cora", "inj_amazon",
            "blogcatalog", "acm", "cola_flickr"]
SEEDS = [1, 2, 3, 4, 5]
HERE = os.path.dirname(os.path.abspath(__file__))

results: dict[str, dict] = {}
for ds in DATASETS:
    per_seed = []
    for seed in SEEDS:
        os.environ["ARCADE_CACHE_SEED"] = str(seed)
        # re-import per seed so cache.SEED picks up the env (module-level constant)
        for mod in [m for m in list(sys.modules) if m.endswith("llm_ablation.cache")]:
            del sys.modules[mod]
        from experiments.llm_ablation.cache import DatasetReplayer  # noqa: E402
        try:
            rep = DatasetReplayer(ds)
        except FileNotFoundError:
            print(f"[cov] {ds} seed {seed}: cache missing — skip", flush=True)
            continue
        out = rep.replay(rep.available)          # full library through the gate
        m = rep.measure(out.fused)
        per_seed.append({"seed": seed, "auroc": m["auroc"], "auprc": m["auprc"],
                         "mode": out.fusion_mode, "trusted": out.trusted})
        print(f"[cov] {ds:12s} seed {seed}  AUROC={m['auroc']}  AUPRC={m['auprc']}  "
              f"mode={out.fusion_mode} trusted={out.trusted}", flush=True)
    if per_seed:
        aucs = [r["auroc"] for r in per_seed]
        aps = [r["auprc"] for r in per_seed]
        results[ds] = {"auroc_mean": round(float(np.mean(aucs)), 1),
                       "auroc_std": round(float(np.std(aucs)), 1),
                       "auprc_mean": round(float(np.mean(aps)), 1),
                       "auprc_std": round(float(np.std(aps)), 1),
                       "seeds": per_seed}
        json.dump(results, open(os.path.join(HERE, "coverage_bench.json"), "w"), indent=2)

print("\n=== ARCADE coverage (full library + gate, mean±std over seeds) ===")
for ds, r in results.items():
    print(f"{ds:12s} {r['auroc_mean']:5.1f} ± {r['auroc_std']:3.1f}   AP {r['auprc_mean']:5.1f} ± {r['auprc_std']:3.1f}")
if results:
    print(f"{'AVG':12s} {np.mean([r['auroc_mean'] for r in results.values()]):5.1f}")
print("COVERAGE_BENCH_DONE", flush=True)
