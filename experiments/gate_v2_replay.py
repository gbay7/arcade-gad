"""Gate-v2 before/after: re-execute every stored ablation selection through the
fixed gate (companion anti-signal check, degrade-to-standard, pairmax pairs,
even-median, dominant-select, order-invariant AOM) against the same cached
scores. No LLM is re-queried — the selections are frozen; only the gate changed.
Labels are used once per row, to measure.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from experiments.llm_ablation.cache import DatasetReplayer

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "llm_ablation", "results.jsonl")

rows = [json.loads(l) for l in open(RESULTS) if l.strip()]
# canonical rows only (greedy for ollama, all for claude), deduped keep-latest
latest = {}
for r in rows:
    t = r.get("temperature")
    if r["kind"] == "llm" and not (t in (0, 0.0) or (t is None and r["model"].startswith("claude:"))):
        continue
    latest[(r["model"], r["dataset"], r.get("repeat", 0))] = r
rows = [r for r in latest.values() if r.get("selected")]

by_ds: dict[str, list] = {}
for r in rows:
    by_ds.setdefault(r["dataset"], []).append(r)

out = []
for ds, rs in sorted(by_ds.items()):
    rep = DatasetReplayer(ds)
    for r in rs:
        res = rep.replay(r["selected"])
        if res is None:
            continue
        m = rep.measure(res.fused)
        out.append({"model": r["model"], "dataset": ds, "repeat": r.get("repeat", 0),
                    "selected": r["selected"], "old_auroc": r.get("auroc"),
                    "new_auroc": m["auroc"], "new_mode": res.fusion_mode,
                    "trusted": res.trusted})
    done = [o for o in out if o["dataset"] == ds]
    delta = np.mean([o["new_auroc"] - (o["old_auroc"] or o["new_auroc"]) for o in done])
    print(f"[gate2] {ds:12s} rows={len(done):3d}  mean delta {delta:+.1f}", flush=True)
    json.dump(out, open(os.path.join(HERE, "gate_v2_replay.json"), "w"), indent=2)

# per-model summary
models = sorted({o["model"] for o in out})
print("\n=== per-model mean AUROC: old gate -> new gate ===")
for mname in models:
    ms = [o for o in out if o["model"] == mname]
    old = np.mean([o["old_auroc"] for o in ms if o["old_auroc"] is not None])
    new = np.mean([o["new_auroc"] for o in ms])
    print(f"{mname:16s} {old:5.1f} -> {new:5.1f}", flush=True)
print("GATE2_DONE", flush=True)
