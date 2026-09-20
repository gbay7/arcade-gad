"""Single-seed held-out replay: the frozen gate over every applicable paradigm,
and each paradigm's own AUROC, from one cached seed. Cheap enough for a graph
whose deep views are too slow to cache five times (Questions: the subgraph
contrast path alone takes over an hour on CPU). Labels are used only to measure.
    python experiments/heldout_seed1_replay.py questions
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.llm_ablation.cache import DatasetReplayer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "heldout_seed1.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for ds in sys.argv[1:]:
    rep = DatasetReplayer(ds)
    out = rep.replay(rep.available)
    m = rep.measure(out.fused)
    per = {n: rep.measure(rep.scores[n])["auroc"] for n in rep.available}
    res[ds] = {"seed": 1, "n_nodes": int(len(rep.labels)),
               "anomaly_rate": round(float(rep.labels.mean() * 100), 2),
               "ARCADE_seed1": {"auroc": m["auroc"], "auprc": m["auprc"],
                                "mode": out.fusion_mode, "trusted": out.trusted},
               "per_detector": dict(sorted(per.items(), key=lambda kv: -kv[1]))}
    print(f"[seed1] {ds}: gate {m['auroc']} ({out.fusion_mode}), best single "
          f"{max(per, key=per.get)} {max(per.values())}", flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
print("SEED1_DONE", flush=True)
