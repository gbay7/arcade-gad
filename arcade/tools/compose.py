"""Compose tools — composable primitives the agent can combine into new diagnostics."""

from __future__ import annotations

import numpy as np
from typing import Any

from arcade.models import ToolResult
from arcade.engine.workspace import Workspace


def select_nodes(ws: Workspace, condition: str, paradigm: str | None = None, threshold: float = 0.95) -> ToolResult:
    """Select nodes by predicate: 'top_percentile', 'community', 'degree_range'."""
    if condition == "top_percentile" and paradigm:
        scores = ws.scoring.get_scores(paradigm)
        if scores is None:
            return ToolResult(data={"error": f"No scores for {paradigm}"}, summary="Run paradigm first.")
        cutoff = np.percentile(scores, threshold * 100)
        selected = np.where(scores > cutoff)[0].tolist()
    elif condition == "high_degree":
        degs = ws.graph.degrees()
        cutoff = np.percentile(degs, threshold * 100)
        selected = np.where(degs > cutoff)[0].tolist()
    elif condition == "low_degree":
        degs = ws.graph.degrees()
        cutoff = np.percentile(degs, (1 - threshold) * 100)
        selected = np.where(degs < cutoff)[0].tolist()
    else:
        return ToolResult(
            data={"error": f"Unknown condition: {condition}"},
            summary=f"Available conditions: top_percentile, high_degree, low_degree",
        )

    data = {"condition": condition, "num_selected": len(selected), "node_ids": selected[:200]}
    summary = f"Selected {len(selected)} nodes by condition '{condition}'."
    return ToolResult(data=data, summary=summary)


def correlate(ws: Workspace, vec_a: str, vec_b: str) -> ToolResult:
    """Correlate two named vectors (paradigm scores, 'degree', 'community_size')."""
    a = _resolve_vector(ws, vec_a)
    b = _resolve_vector(ws, vec_b)
    if a is None:
        return ToolResult(data={"error": f"Cannot resolve '{vec_a}'"}, summary=f"Unknown vector: {vec_a}")
    if b is None:
        return ToolResult(data={"error": f"Cannot resolve '{vec_b}'"}, summary=f"Unknown vector: {vec_b}")

    rho = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-10 and np.std(b) > 1e-10 else 0.0

    data = {"vec_a": vec_a, "vec_b": vec_b, "pearson_rho": round(rho, 4), "abs_rho": round(abs(rho), 4)}
    summary = f"Correlation between '{vec_a}' and '{vec_b}': ρ={rho:.4f} ({'strong' if abs(rho) > 0.7 else 'moderate' if abs(rho) > 0.4 else 'weak'})."
    return ToolResult(data=data, summary=summary)


def histogram(ws: Workspace, vector: str, bins: int = 30) -> ToolResult:
    """Compute histogram of a named vector."""
    v = _resolve_vector(ws, vector)
    if v is None:
        return ToolResult(data={"error": f"Cannot resolve '{vector}'"}, summary=f"Unknown vector: {vector}")

    counts, edges = np.histogram(v, bins=bins)
    data = {
        "vector": vector,
        "histogram": {"counts": counts.tolist(), "bin_edges": [round(float(e), 4) for e in edges]},
        "stats": {
            "mean": round(float(v.mean()), 4),
            "std": round(float(v.std()), 4),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
        },
    }
    summary = f"Histogram of '{vector}': mean={float(v.mean()):.4f}, std={float(v.std()):.4f}, range [{float(v.min()):.4f}, {float(v.max()):.4f}]."
    return ToolResult(data=data, summary=summary)


def subgraph_stats(ws: Workspace, node_ids: list[int]) -> ToolResult:
    """Compute induced subgraph statistics."""
    edges = ws.graph.subgraph_edges(node_ids)
    max_edges = len(node_ids) * (len(node_ids) - 1) / 2
    density = len(edges) / max(max_edges, 1)

    x = ws.graph.node_features(node_ids)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(x)
    np.fill_diagonal(sims, 0)
    avg_sim = float(sims.sum() / max(sims.size - len(node_ids), 1))

    degs = ws.graph.degrees()
    comms = ws.graph.communities()

    data = {
        "num_nodes": len(node_ids),
        "num_edges": len(edges),
        "density": round(density, 4),
        "avg_pairwise_similarity": round(avg_sim, 4),
        "avg_degree": round(float(degs[node_ids].mean()), 2),
        "communities": {int(k): int(v) for k, v in zip(*np.unique(comms[node_ids], return_counts=True))},
    }
    summary = (
        f"Subgraph of {len(node_ids)} nodes: {len(edges)} edges, density {density:.4f}. "
        f"Avg pairwise similarity: {avg_sim:.3f}. "
        f"{'Near-clique! ' if density > 0.7 else ''}{'Highly similar features. ' if avg_sim > 0.8 else ''}"
    )
    return ToolResult(data=data, summary=summary)


def pairwise_similarity(ws: Workspace, node_ids: list[int]) -> ToolResult:
    """Compute pairwise feature similarity matrix for a node set."""
    x = ws.graph.node_features(node_ids)
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(x)

    data = {
        "node_ids": node_ids[:50],
        "similarity_matrix": sim_matrix[:50, :50].tolist(),
        "mean_similarity": round(float((sim_matrix.sum() - len(node_ids)) / max(sim_matrix.size - len(node_ids), 1)), 4),
    }
    summary = f"Pairwise similarity for {len(node_ids)} nodes: mean off-diagonal = {data['mean_similarity']:.4f}."
    return ToolResult(data=data, summary=summary)


def compare_groups(ws: Workspace, group_a: list[int], group_b: list[int]) -> ToolResult:
    """Compare two node groups across paradigms and features."""
    comparison = {"paradigms": {}, "features": {}}

    for name in ws.scoring.available_paradigms:
        s = ws.scoring.get_scores(name)
        if s is not None:
            comparison["paradigms"][name] = {
                "group_a_mean": round(float(s[group_a].mean()), 4),
                "group_b_mean": round(float(s[group_b].mean()), 4),
                "difference": round(float(s[group_a].mean() - s[group_b].mean()), 4),
            }

    x = ws.graph.node_features()
    norm_a = float(np.linalg.norm(x[group_a], axis=1).mean())
    norm_b = float(np.linalg.norm(x[group_b], axis=1).mean())
    comparison["features"] = {
        "group_a_mean_norm": round(norm_a, 4),
        "group_b_mean_norm": round(norm_b, 4),
    }

    data = {"group_a_size": len(group_a), "group_b_size": len(group_b), "comparison": comparison}
    summary = f"Comparison: group A ({len(group_a)} nodes) vs group B ({len(group_b)} nodes) across {len(comparison['paradigms'])} paradigms."
    return ToolResult(data=data, summary=summary)


def _resolve_vector(ws: Workspace, name: str) -> np.ndarray | None:
    scores = ws.scoring.get_scores(name)
    if scores is not None:
        return scores
    if name == "degree":
        return ws.graph.degrees().astype(float)
    if name == "community_size":
        comms = ws.graph.communities()
        sizes = np.bincount(comms)
        return sizes[comms].astype(float)
    return None
