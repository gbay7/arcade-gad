"""Measure tools — run paradigm detectors and compute scores."""

from __future__ import annotations

import numpy as np
from typing import Any

from arcade.models import ToolResult, FailResult
from arcade.engine.workspace import Workspace
from arcade.paradigms.detectors import run_detector


def run_paradigm(ws: Workspace, paradigm: str, **kwargs) -> ToolResult | FailResult:
    registry = ws.scoring._paradigms
    p = registry.get(paradigm)
    if p is None:
        from arcade.paradigms import ParadigmRegistry
        pr = ParadigmRegistry()
        p = pr.get(paradigm)
        if p is None:
            return FailResult(
                error=f"Unknown paradigm: {paradigm}",
                suggestion=f"Available: {list(registry.keys())}",
            )
        ws.scoring.register_paradigm(p)

    try:
        scores = run_detector(p.detector_name, ws.graph, **kwargs)
    except Exception as e:
        return FailResult(error=f"Detector failed: {e}", suggestion="Try with different parameters")

    ws.scoring.set_scores(paradigm, scores)

    p90 = float(np.percentile(scores, 90))
    p95 = float(np.percentile(scores, 95))
    p99 = float(np.percentile(scores, 99))
    top_20 = np.argsort(scores)[-20:][::-1]

    data = {
        "paradigm": paradigm,
        "assumption": p.assumption,
        "num_nodes_scored": len(scores),
        "stats": {
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "p90": round(p90, 4),
            "p95": round(p95, 4),
            "p99": round(p99, 4),
        },
        "top_20_nodes": [
            {"node_id": int(i), "score": round(float(scores[i]), 4)}
            for i in top_20
        ],
        "nodes_above_p95": int((scores > p95).sum()),
        "nodes_above_p99": int((scores > p99).sum()),
    }

    summary = (
        f"Paradigm '{paradigm}' ({p.assumption}): "
        f"mean={data['stats']['mean']:.4f}, std={data['stats']['std']:.4f}. "
        f"P95={p95:.4f} ({data['nodes_above_p95']} nodes above), "
        f"P99={p99:.4f} ({data['nodes_above_p99']} nodes above). "
        f"Top node: {int(top_20[0])} (score {float(scores[top_20[0]]):.4f})."
    )

    return ToolResult(data=data, summary=summary)


def run_all_selected(ws: Workspace, **kwargs) -> ToolResult:
    if ws.profile is None:
        ws.profile = ws.graph.profile()

    from arcade.paradigms import ParadigmRegistry
    pr = ParadigmRegistry()
    applicable = pr.applicable(ws.profile)

    results = {}
    failed = []
    for p in applicable:
        ws.scoring.register_paradigm(p)
        try:
            scores = run_detector(p.detector_name, ws.graph, **kwargs)
            ws.scoring.set_scores(p.name, scores)
            results[p.name] = {
                "mean": round(float(scores.mean()), 4),
                "std": round(float(scores.std()), 4),
                "p95": round(float(np.percentile(scores, 95)), 4),
                "p99": round(float(np.percentile(scores, 99)), 4),
                "top_5": [int(i) for i in np.argsort(scores)[-5:][::-1]],
            }
        except Exception as e:
            failed.append({"paradigm": p.name, "error": str(e)})

    data = {
        "paradigms_run": list(results.keys()),
        "paradigms_failed": failed,
        "results": results,
    }
    summary = (
        f"Ran {len(results)}/{len(applicable)} applicable paradigms. "
        f"Successful: {', '.join(results.keys())}. "
        + (f"Failed: {', '.join(f['paradigm'] for f in failed)}." if failed else "")
    )
    return ToolResult(data=data, summary=summary)


def score_matrix(ws: Workspace) -> ToolResult:
    paradigms = ws.scoring.available_paradigms
    if not paradigms:
        return ToolResult(data={"error": "No scores computed yet"}, summary="No paradigm scores available.")

    n = ws.graph.num_nodes
    top_50 = _top_k_fused(ws, 50)

    data = {
        "paradigms": paradigms,
        "num_nodes": n,
        "top_50": top_50,
    }
    summary = (
        f"Score matrix: {len(paradigms)} paradigms × {n} nodes. "
        f"Top node (fused): {top_50[0]['node_id']} with score {top_50[0]['fused_score']:.4f}."
    )
    return ToolResult(data=data, summary=summary)


def score_percentiles(ws: Workspace, paradigm: str) -> ToolResult:
    scores = ws.scoring.get_scores(paradigm)
    if scores is None:
        return ToolResult(data={"error": f"No scores for {paradigm}"}, summary=f"Paradigm '{paradigm}' not run.")

    from scipy.stats import rankdata
    ranks = rankdata(scores, method="average") / len(scores)

    data = {
        "paradigm": paradigm,
        "percentiles": {
            f"p{p}": round(float(np.percentile(scores, p)), 4)
            for p in [10, 25, 50, 75, 90, 95, 99]
        },
        "top_10_by_percentile": [
            {"node_id": int(i), "score": round(float(scores[i]), 4), "percentile": round(float(ranks[i]), 4)}
            for i in np.argsort(scores)[-10:][::-1]
        ],
    }
    summary = (
        f"Percentiles for '{paradigm}': "
        f"P50={data['percentiles']['p50']}, P90={data['percentiles']['p90']}, "
        f"P95={data['percentiles']['p95']}, P99={data['percentiles']['p99']}."
    )
    return ToolResult(data=data, summary=summary)


def _top_k_fused(ws: Workspace, k: int) -> list[dict]:
    try:
        return ws.scoring.top_k(k)
    except Exception:
        return []
