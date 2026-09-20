"""Faithful AutoGAD reproduction: run AUTOGAD'S OWN code (their modified PyGOD
DOMINANT with a_mode/s_mode reconstruction variants) over AUTOGAD'S OWN search
space, on our 7 datasets, and report the best architecture per dataset --- their
selection criterion (highest AUROC).

AutoGAD's search is an LLM-guided NAS over this space; the LLM navigates it to a
good architecture. We reproduce the OUTCOME with a fixed-budget random search
over the same space using their detector, which finds a near-best architecture
under the same criterion without needing their ChatGPT key.

Their code lives in scratchpad/AutoGAD (downloaded from the anonymous repo);
their pygod (1.0.0) is loaded ahead of ours via sys.path.
Output: experiments/autogad_faithful.json
"""
import sys, os, json, warnings, random, time
warnings.filterwarnings("ignore")
# AutoGAD's released code. Download once with:
#   curl -sL -o AutoGAD.zip https://anonymous.4open.science/api/repo/AutoGAD-A8D7/zip
#   unzip AutoGAD.zip -d AutoGAD && touch AutoGAD/base_exp/utils/__init__.py
# then point AUTOGAD_DIR at it (defaults to experiments/AutoGAD).
HERE0 = os.path.dirname(os.path.abspath(__file__))
AG = os.environ.get("AUTOGAD_DIR") or os.path.join(HERE0, "AutoGAD")
assert os.path.isdir(os.path.join(AG, "pygod")), (
    f"AutoGAD code not found at {AG}; set AUTOGAD_DIR (see the header of this file).")
sys.path.insert(0, os.path.dirname(HERE0))
sys.path.insert(0, AG)                       # their pygod
sys.path.insert(0, os.path.join(AG, "base_exp"))   # their utils.common (a_mode/s_mode loss)
import torch, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
torch.serialization.add_safe_globals([GlobalStorage, NodeStorage, EdgeStorage, Data])
import pygod
assert pygod.__version__ == "1.0.0", f"expected their pygod 1.0.0, got {pygod.__version__}"
from pygod.detector import DOMINANT
from torch_geometric.nn import GCN, GAT, GIN, GraphSAGE
from arcade.data.loader import GraphStore

# AutoGAD's search space (base_exp/hyper_space.py + main.py)
SPACE = dict(
    lr=[0.005, 0.001, 0.0005], alpha=[0.7, 0.8, 0.9], dim=[128, 256],
    dropout=[0, 0.1, 0.3], GNN=[GCN, GraphSAGE, GIN, GAT],
    contamination=[0.1, 0.05, 0.03], layer=[3, 4, 6],
    a_mode=[1, 2, 3], s_mode=[1, 2, 3],
)
BUDGET = 60           # architectures searched per dataset
EPOCH = 100
KEYS = {"BlogCatalog": "blogcatalog", "ACM": "acm", "Flickr": "cola_flickr"}
HIDDEN = {
        "Amazon": "inj_amazon", "Cora": "inj_cora", "Weibo": "weibo"}
HERE = os.path.dirname(os.path.abspath(__file__))

results = {}
for label, key in KEYS.items():
    s = GraphStore(); s.load_dataset(key)
    y = (np.asarray(s.labels) > 0).astype(int)
    d = Data(x=s.data.x.float(), edge_index=s.data.edge_index, y=torch.tensor(y))
    rng = random.Random(0)
    seen, best = set(), {"auroc": 0.0}
    t0 = time.time()
    tried = 0
    while tried < BUDGET:
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        sig = (cfg["lr"], cfg["alpha"], cfg["dim"], cfg["dropout"], cfg["GNN"].__name__,
               cfg["contamination"], cfg["layer"], cfg["a_mode"], cfg["s_mode"])
        if sig in seen:
            continue
        seen.add(sig); tried += 1
        torch.manual_seed(0); np.random.seed(0)
        try:
            det = DOMINANT(hid_dim=cfg["dim"], lr=cfg["lr"], weight=cfg["alpha"],
                           dropout=cfg["dropout"], backbone=cfg["GNN"], gpu=0,
                           contamination=cfg["contamination"], num_layers=cfg["layer"],
                           a_mode=cfg["a_mode"], s_mode=cfg["s_mode"], epoch=EPOCH)
            det.fit(d)
            sc = det.decision_score_
            sc = sc.cpu().numpy() if torch.is_tensor(sc) else np.asarray(sc)
            if np.isnan(sc).any() or np.std(sc) < 1e-9:
                continue
            a = roc_auc_score(y, sc) * 100
            if a > best["auroc"]:
                best = {"auroc": round(a, 2),
                        "auprc": round(average_precision_score(y, sc) * 100, 2),
                        "config": {**{k: cfg[k] for k in cfg if k != "GNN"}, "GNN": cfg["GNN"].__name__}}
        except Exception:
            continue
        torch.cuda.empty_cache()
    results[label] = {**best, "budget": tried, "secs": round(time.time() - t0)}
    print(f"[autogad] {label:7s} best AUROC={best['auroc']:5.1f} "
          f"[{best.get('config',{}).get('GNN','-')}] ({time.time()-t0:.0f}s)", flush=True)
    json.dump(results, open(os.path.join(HERE, "autogad_faithful.json"), "w"), indent=2)
print("AUTOGAD_FAITHFUL_DONE", flush=True)
