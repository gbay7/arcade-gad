"""SL-GAD (7th baseline) with variance: 5 seeds on each of the 7 datasets, via
our DGL-free port of the authors' code (sl_gad_run). Reports mean+-std AUROC.
test_rounds reduced to 128 to keep the 35 runs tractable; the multi-round
average is stable well before 256.
Output: experiments/rebench_slgad.json
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, torch
from types import SimpleNamespace
import sl_gad_run as slg

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
KEYS = {"Enron": "enron", "Reddit": "reddit", "Books": "books", "Disney": "disney",
        "Amazon": "inj_amazon", "Cora": "inj_cora", "Weibo": "weibo"}
SEEDS = [1, 2, 3, 4, 5]

results = {}
for label, key in KEYS.items():
    adj, feat, ano = slg.load_ours(key)
    aucs = []
    for sd in SEEDS:
        args = SimpleNamespace(lr=1e-3, weight_decay=0.0, embedding_dim=64, num_epoch=100,
                               patience=400, batch_size=300, subgraph_size=4, readout='avg',
                               auc_test_rounds=128, negsamp_ratio=1, alpha=1.0, beta=0.6,
                               seed=sd, return_scores=False)
        auc, ap = slg.run(adj, feat, ano, args, device)
        aucs.append(auc)
    results[label] = {"mean": round(float(np.mean(aucs)), 1), "std": round(float(np.std(aucs)), 1),
                      "per_seed": [round(a, 2) for a in aucs]}
    print(f"[slgad] {label:7s} {np.mean(aucs):5.1f}+-{np.std(aucs):.1f}", flush=True)
    json.dump(results, open(os.path.join(HERE, "rebench_slgad.json"), "w"), indent=2)
print("REBENCH_SLGAD_DONE", flush=True)
