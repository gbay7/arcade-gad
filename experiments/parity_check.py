"""Parity check: our LIBRARY subgraph_contrast_detector vs the FAITHFUL
sl_gad_run, same dataset, same settings (128 rounds, seed 1). If they match
within seed noise, the library port is already faithful and the reported
standalone gap was a stale/low-round measurement; if not, the diff is real and
localizable. Small graph (inj_cora) so both finish fast.
"""
import os, sys, json, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from arcade.data.loader import GraphStore
from arcade.paradigms.detectors import subgraph_contrast_detector

KEY = sys.argv[1] if len(sys.argv) > 1 else "inj_cora"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 128

# ---- our library detector ----
s = GraphStore(); s.load_dataset(KEY)
y = (np.asarray(s.labels) > 0).astype(int)
t = time.time()
lib = subgraph_contrast_detector(s, epochs=100, test_rounds=ROUNDS, seed=1)
lib_auc = roc_auc_score(y, lib) * 100
print(f"[parity] LIBRARY   {KEY} rounds={ROUNDS} AUROC={lib_auc:.2f}  ({time.time()-t:.0f}s)", flush=True)

# ---- faithful SL-GAD ----
import sl_gad_run as F
adj, feat, ano = F.load_ours(KEY)
class A: pass
args = A()
args.lr=1e-3; args.weight_decay=0.0; args.embedding_dim=64; args.num_epoch=100
args.patience=400; args.batch_size=300; args.subgraph_size=4; args.readout='avg'
args.auc_test_rounds=ROUNDS; args.negsamp_ratio=1; args.alpha=1.0; args.beta=0.6; args.seed=1
device = torch.device('cpu')
t = time.time()
f_auc, f_ap = F.run(adj, feat, ano, args, device)
print(f"[parity] FAITHFUL  {KEY} rounds={ROUNDS} AUROC={f_auc:.2f}  ({time.time()-t:.0f}s)", flush=True)

print(f"[parity] GAP (faithful - library) = {f_auc - lib_auc:+.2f}", flush=True)
json.dump({"dataset": KEY, "rounds": ROUNDS, "library": round(lib_auc,2),
           "faithful": round(f_auc,2), "gap": round(f_auc-lib_auc,2)},
          open(os.path.join(HERE, f"parity_{KEY}.json"), "w"), indent=2)
print("PARITY_DONE", flush=True)
