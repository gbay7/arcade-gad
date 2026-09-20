"""Per-dataset paradigm-score cache + genuine composition replay.

Paradigm scores do not depend on which LLM selected them, so the expensive part
of the ablation — running the detectors (deep ones cost minutes) — is computed
ONCE per dataset with a fixed seed and cached. Each model's composition is then
replayed through ARCADE's real ScoringEngine with exactly the server's call
sequence (compute_weights -> set_profile -> compute_fusion_gate -> fuse), so
the label-free gate decisions are the genuine ones.

Division of labour in the ablation:
  - model context/decisions: through the MCP server (protocol.py / runner.py);
  - score computation + replay: in-process through the same arcade functions the
    MCP server itself calls (run_detector, ScoringEngine) — identical code path,
    without re-paying minutes of GPU per model for identical scores.

Labels are stored in the cache but consumed only by `measure`, never by replay.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from arcade.data.loader import GraphStore
from arcade.engine.scoring import ScoringEngine
from arcade.paradigms.registry import BUILTIN_PARADIGMS as PARADIGMS
from arcade.paradigms.detectors import run_detector

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_cache")
# Seed for detector runs. Seed 1 keeps the original file names (the ablation's
# cache); other seeds (ARCADE_CACHE_SEED) get suffixed files for the multi-seed
# coverage protocol.
SEED = int(os.environ.get("ARCADE_CACHE_SEED", 1))


def _cache_path(dataset: str) -> str:
    suffix = "" if SEED == 1 else f"_s{SEED}"
    return os.path.join(CACHE_DIR, f"{dataset}{suffix}.npz")


def _paradigm_by_name() -> dict[str, Any]:
    return {p.name: p for p in PARADIGMS}


def build_cache(dataset: str, force: bool = False) -> str:
    """Run every applicable paradigm once on `dataset`; persist scores + labels."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(dataset)
    if os.path.exists(path) and not force:
        return path

    store = GraphStore()
    store.load_dataset(dataset)
    profile = store.profile()
    byname = _paradigm_by_name()
    applicable = [p.name for p in PARADIGMS if p.is_applicable(profile)]

    arrays: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for name in applicable:
        det = byname[name].detector_name
        t0 = time.time()
        try:
            arrays[f"score__{name}"] = np.asarray(
                run_detector(det, store, seed=SEED), dtype=np.float64)
            timings[name] = round(time.time() - t0, 1)
            print(f"[cache] {dataset:14s} {name:22s} {timings[name]:7.1f}s", flush=True)
        except Exception as e:  # a failing paradigm is recorded, not fatal
            print(f"[cache] {dataset:14s} {name:22s} ERROR {str(e)[:80]}", flush=True)

    arrays["labels"] = (np.asarray(store.labels) > 0).astype(np.int64)
    np.savez_compressed(path, **arrays)
    meta = {"dataset": dataset, "seed": SEED, "applicable": applicable,
            "cached": sorted(k.split("__", 1)[1] for k in arrays if k.startswith("score__")),
            "timings_s": timings}
    with open(path.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return path


@dataclass
class ReplayResult:
    fused: np.ndarray
    fusion_mode: str
    trusted: list[str] | None
    weights: dict[str, float]
    used: list[str]


class DatasetReplayer:
    """Replays any paradigm subset through the genuine ScoringEngine."""

    def __init__(self, dataset: str):
        npz = np.load(_cache_path(dataset))
        self.scores = {k.split("__", 1)[1]: npz[k] for k in npz.files if k.startswith("score__")}
        self.labels = npz["labels"]
        self.available = sorted(self.scores)
        # graph context for the genuine weight computation (cheap to rebuild)
        self._store = GraphStore()
        self._store.load_dataset(dataset)
        self._profile = self._store.profile()
        self._degrees = self._store.degrees()
        self._communities = self._store.communities()

    def replay(self, selected: list[str]) -> ReplayResult | None:
        used = [s for s in selected if s in self.scores]
        if not used:
            return None
        eng = ScoringEngine()
        for p in PARADIGMS:            # same registry the server exposes
            eng.register_paradigm(p)
        for name in used:
            eng.set_scores(name, self.scores[name])
        # exact server sequence (server.py compute_weights tool)
        weights = eng.compute_weights(self._profile, self._degrees, self._communities)
        eng.set_profile(self._profile)
        gate = eng.compute_fusion_gate()
        fused = eng.fuse()
        return ReplayResult(
            fused=np.asarray(fused, dtype=np.float64),
            fusion_mode=gate.get("mode", eng.fusion_mode),
            trusted=gate.get("trusted"),
            weights={k: round(w.w, 4) for k, w in weights.items()},
            used=used,
        )

    def measure(self, fused: np.ndarray) -> dict[str, float]:
        from sklearn.metrics import roc_auc_score, average_precision_score
        return {
            "auroc": round(float(roc_auc_score(self.labels, fused)) * 100, 1),
            "auprc": round(float(average_precision_score(self.labels, fused)) * 100, 1),
        }


if __name__ == "__main__":
    import sys
    for ds in sys.argv[1:] or ["inj_cora"]:
        build_cache(ds)
    print("CACHE_DONE")
