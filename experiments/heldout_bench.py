"""Frozen held-out evaluation on a graph never used during development.
Rules (library, preconditions, thresholds, gate) are the shipped ones at the
commit recorded below; only the data loader gained the dataset.

For <ds>: (1) ARCADE run-everything + gate replayed over the 5 seed caches
(build them first with ARCADE_CACHE_SEED=s python -m experiments.llm_ablation.cache <ds>);
(2) PyGOD baselines + Radar via the isolated worker (seeds 0-4, epoch 200);
(3) SL-GAD via the validated port (seeds 1-5, 128 test rounds).
Writes heldout_<ds>.json incrementally.
    python experiments/heldout_bench.py tolokers
"""
import json, os, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np

ds = sys.argv[1]
OUT = os.path.join(HERE, f"heldout_{ds}.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res.setdefault("frozen_commit", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                                text=True, cwd=ROOT).stdout.strip())
def save(): json.dump(res, open(OUT, "w"), indent=2)

# (1) ARCADE coverage replay over seed caches
if "ARCADE" not in res:
    per = []
    for seed in [1, 2, 3, 4, 5]:
        os.environ["ARCADE_CACHE_SEED"] = str(seed)
        for mod in [m for m in list(sys.modules) if m.endswith("llm_ablation.cache")]:
            del sys.modules[mod]
        from experiments.llm_ablation.cache import DatasetReplayer
        try:
            rep = DatasetReplayer(ds)
        except FileNotFoundError:
            print(f"[heldout] {ds} seed {seed}: cache missing", flush=True); continue
        out = rep.replay(rep.available); m = rep.measure(out.fused)
        per.append({"seed": seed, "auroc": m["auroc"], "auprc": m["auprc"], "mode": out.fusion_mode,
                    "trusted": out.trusted, "applicable": rep.available})
        print(f"[heldout] {ds} seed {seed} ARCADE auroc={m['auroc']} ap={m['auprc']} mode={out.fusion_mode}", flush=True)
    if per:
        res["ARCADE"] = {"mean": round(float(np.mean([r["auroc"] for r in per])), 1),
                         "std": round(float(np.std([r["auroc"] for r in per])), 1),
                         "ap_mean": round(float(np.mean([r["auprc"] for r in per])), 1),
                         "ap_std": round(float(np.std([r["auprc"] for r in per])), 1), "seeds": per}
        # per-detector AUROC (seed-1 cache) for the detector-matrix view
        os.environ["ARCADE_CACHE_SEED"] = "1"
        for mod in [m for m in list(sys.modules) if m.endswith("llm_ablation.cache")]: del sys.modules[mod]
        from experiments.llm_ablation.cache import DatasetReplayer
        rep = DatasetReplayer(ds)
        res["per_detector_seed1"] = {n: rep.measure(rep.scores[n])["auroc"] for n in rep.available}
        save()

# (2) PyGOD baselines + Radar
for m in ["DOMINANT", "AnomalyDAE", "DONE", "CoLA", "GAAN", "OCGNN", "Radar"]:
    if m in res: continue
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, "rebench_pygod_worker.py"), ds, m],
                       capture_output=True, text=True, timeout=10800, cwd=ROOT)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    res[m] = json.loads(line[7:]) if line else {"error": (p.stderr.strip().splitlines() or ["?"])[-1][:120]}
    print(f"[heldout] {ds} {m:11s} {res[m]} ({time.time()-t0:.0f}s)", flush=True); save()

# (3) SL-GAD port
if "SL-GAD" not in res:
    import torch
    from types import SimpleNamespace
    import sl_gad_run as slg
    adj, feat, ano = slg.load_ours(ds)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    aucs, aps = [], []; t0 = time.time()
    for sd in [1, 2, 3, 4, 5]:
        a = SimpleNamespace(lr=1e-3, weight_decay=0.0, embedding_dim=64, num_epoch=100, patience=400,
                            batch_size=300, subgraph_size=4, readout='avg', auc_test_rounds=128,
                            negsamp_ratio=1, alpha=1.0, beta=0.6, seed=sd, return_scores=False)
        auc, ap = slg.run(adj, feat, ano, a, dev); aucs.append(auc); aps.append(ap)
    res["SL-GAD"] = {"mean": round(float(np.mean(aucs)), 1), "std": round(float(np.std(aucs)), 1),
                     "ap_mean": round(float(np.mean(aps)), 1), "ap_std": round(float(np.std(aps)), 1),
                     "rounds": 128, "n": 5}
    print(f"[heldout] {ds} SL-GAD {res['SL-GAD']} ({time.time()-t0:.0f}s)", flush=True); save()
print("HELDOUT_DONE", flush=True)
