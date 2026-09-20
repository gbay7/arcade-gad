"""End-to-end cost: wall-clock and peak GPU memory for
  run-everything  : every applicable paradigm through run_detector (+ gate);
  analyst mode    : evidence construction (fast paradigms + cheap deep probe)
                    + the analyst's stored selection (qwen3:8b, pack-v4)
                    + decision latency (median from the stored transcripts);
  SL-GAD          : the strongest baseline, via the validated port (128 rounds).
One seed per dataset (cost, not accuracy). Writes runtime_bench.json.
    python experiments/runtime_bench.py enron inj_amazon acm
"""
import json, os, sys, time, warnings, collections
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, torch
from arcade.data.loader import GraphStore
from arcade.paradigms.registry import BUILTIN_PARADIGMS as PARADIGMS
from arcade.paradigms.detectors import run_detector, subgraph_contrast_detector

FAST = ["local_affinity","density","community","homophily_violation","spectral","attr_struct_mismatch",
        "clique_density","conformity","linear_residual","peripheral_mismatch","attribute_inflation"]
OUT = os.path.join(HERE, "runtime_bench.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def timed(fn):
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time(); fn(); t = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return round(t, 1), round(mem, 2)

def qwen_selection(ds):
    sel = collections.Counter(); lat = []
    for l in open(f"{HERE}/llm_ablation/results_packv4.jsonl"):
        r = json.loads(l)
        if r.get("model") == "qwen3:8b" and r["dataset"] == ds and r.get("executed"):
            sel[tuple(sorted(r["selected"]))] += 1; lat.append(r.get("latency_s", 0))
    return list(sel.most_common(1)[0][0]), float(np.median(lat))

for ds in sys.argv[1:]:
    if ds in res: continue
    store = GraphStore(); store.load_dataset(ds); prof = store.profile()
    byname = {p.name: p for p in PARADIGMS}
    applicable = [p.name for p in PARADIGMS if p.is_applicable(prof)]
    row = {"n": store.data.num_nodes, "applicable": applicable}
    # run-everything
    def run_all():
        for n in applicable: run_detector(byname[n].detector_name, store, seed=1)
    row["run_everything"] = dict(zip(("wall_s", "peak_gb"), timed(run_all)))
    # analyst: evidence (fast + probe) then selection
    def evidence():
        for n in applicable:
            if n in FAST: run_detector(byname[n].detector_name, store, seed=1)
        if "subgraph_contrast" in applicable:
            subgraph_contrast_detector(store, epochs=5, test_rounds=4, seed=1)
    row["analyst_evidence"] = dict(zip(("wall_s", "peak_gb"), timed(evidence)))
    sel, lat = qwen_selection(ds)
    def selected():
        for n in sel:
            if n not in FAST: run_detector(byname[n].detector_name, store, seed=1)
    row["analyst_selected"] = dict(zip(("wall_s", "peak_gb"), timed(selected)))
    row["analyst_selection"] = sel; row["analyst_decision_s"] = round(lat, 1)
    row["analyst_total_s"] = round(row["analyst_evidence"]["wall_s"] + lat + row["analyst_selected"]["wall_s"], 1)
    # SL-GAD
    import sl_gad_run as slg
    from types import SimpleNamespace
    adj, feat, ano = slg.load_ours(ds); dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = SimpleNamespace(lr=1e-3, weight_decay=0.0, embedding_dim=64, num_epoch=100, patience=400, batch_size=300,
                        subgraph_size=4, readout='avg', auc_test_rounds=128, negsamp_ratio=1, alpha=1.0, beta=0.6,
                        seed=1, return_scores=False)
    row["slgad"] = dict(zip(("wall_s", "peak_gb"), timed(lambda: slg.run(adj, feat, ano, a, dev))))
    res[ds] = row; json.dump(res, open(OUT, "w"), indent=2)
    print(ds, json.dumps(row), flush=True)
print("RUNTIME_DONE", flush=True)
