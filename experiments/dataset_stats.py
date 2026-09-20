"""Dataset statistics table: the exact profile quantities the
preconditions read, plus size and anomaly rate, for every benchmark.
Writes dataset_stats.json."""
import dataclasses, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from arcade.data.loader import GraphStore

KEYS = ["enron", "reddit", "books", "disney", "weibo", "inj_cora", "inj_amazon",
        "blogcatalog", "acm", "cola_flickr"] + sys.argv[1:]
FIELDS = ["num_nodes", "num_edges", "density", "avg_degree", "clustering_coeff",
          "num_communities", "modularity", "homophily", "feature_dim",
          "feature_sparsity", "feature_dup_rate"]
out = json.load(open(f"{HERE}/dataset_stats.json")) if os.path.exists(f"{HERE}/dataset_stats.json") else {}
for key in KEYS:
    if key in out: continue
    s = GraphStore(); s.load_dataset(key); p = dataclasses.asdict(s.profile())
    y = np.asarray(s.labels) > 0
    out[key] = {f: (round(float(p[f]), 4) if isinstance(p[f], float) else p[f]) for f in FIELDS}
    out[key]["anomaly_rate_pct"] = round(float(y.mean()) * 100, 2)
    print(key, out[key], flush=True)
    json.dump(out, open(f"{HERE}/dataset_stats.json", "w"), indent=2)
print("STATS_DONE", flush=True)
