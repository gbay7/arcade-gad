"""Explore tools — graph profiling, community detection, structural analysis."""

from __future__ import annotations

import numpy as np
from typing import Any

from arcade.models import ToolResult
from arcade.engine.workspace import Workspace


def graph_profile(ws: Workspace) -> ToolResult:
    profile = ws.graph.profile()
    ws.profile = profile

    data = {
        "num_nodes": profile.num_nodes,
        "num_edges": profile.num_edges,
        "density": round(profile.density, 6),
        "avg_degree": round(profile.avg_degree, 2),
        "max_degree": profile.max_degree,
        "degree_std": round(profile.degree_std, 2),
        "clustering_coeff": round(profile.clustering_coeff, 4),
        "num_communities": profile.num_communities,
        "modularity": round(profile.modularity, 4),
        "homophily": round(profile.homophily, 4),
        "feature_dim": profile.feature_dim,
        "feature_sparsity": round(profile.feature_sparsity, 4),
        # Fraction of nodes whose feature row duplicates another node's. The
        # profile always computed this but the tool did not expose it — and
        # without it no analyst, at any model scale, could infer that
        # template-conformity anomalies were plausible (every model missed the
        # mechanism on the one graph where it dominates).
        "feature_dup_rate": round(profile.feature_dup_rate, 4),
        "spectral_gap": round(profile.spectral_gap, 4),
        "is_power_law": profile.is_power_law,
    }

    summary = (
        f"Graph '{ws.graph.name}': {profile.num_nodes} nodes, {profile.num_edges} edges, "
        f"density {profile.density:.4f}, avg degree {profile.avg_degree:.1f}. "
        f"{profile.num_communities} communities (modularity {profile.modularity:.3f}). "
        f"Homophily {profile.homophily:.3f}. Features: {profile.feature_dim}-dim, "
        f"{profile.feature_sparsity:.1%} sparse, {profile.feature_dup_rate:.1%} duplicate rows. "
        f"{'Power-law' if profile.is_power_law else 'Non-power-law'} degree distribution."
    )

    return ToolResult(data=data, summary=summary)


def degree_distribution(ws: Workspace) -> ToolResult:
    degs = ws.graph.degrees()
    data = {
        "min": int(degs.min()),
        "max": int(degs.max()),
        "mean": round(float(degs.mean()), 2),
        "median": round(float(np.median(degs)), 2),
        "std": round(float(degs.std()), 2),
        "p90": round(float(np.percentile(degs, 90)), 2),
        "p95": round(float(np.percentile(degs, 95)), 2),
        "p99": round(float(np.percentile(degs, 99)), 2),
        "histogram": _histogram(degs, bins=20),
    }
    summary = (
        f"Degree distribution: min={data['min']}, max={data['max']}, "
        f"mean={data['mean']}, std={data['std']}. "
        f"P95={data['p95']}, P99={data['p99']}."
    )
    return ToolResult(data=data, summary=summary)


def community_detect(ws: Workspace) -> ToolResult:
    comms = ws.graph.communities()
    sizes = ws.graph.community_sizes()
    n_comms = len(sizes)

    data = {
        "num_communities": n_comms,
        "sizes": sizes,
        "largest": max(sizes.values()),
        "smallest": min(sizes.values()),
        "modularity": round(ws.graph.profile().modularity, 4),
    }
    summary = (
        f"{n_comms} communities detected. "
        f"Sizes range from {data['smallest']} to {data['largest']} nodes. "
        f"Modularity: {data['modularity']}."
    )
    return ToolResult(data=data, summary=summary)


def homophily_score(ws: Workspace) -> ToolResult:
    profile = ws.graph.profile()
    comms = ws.graph.communities()
    ei = ws.graph.edge_index.cpu().numpy()

    per_comm = {}
    for c in range(profile.num_communities):
        mask = comms == c
        node_set = set(np.where(mask)[0].tolist())
        total, same = 0, 0
        for i in range(ei.shape[1]):
            s, t = int(ei[0, i]), int(ei[1, i])
            if s in node_set or t in node_set:
                total += 1
                if s in node_set and t in node_set:
                    same += 1
        per_comm[int(c)] = round(same / max(total, 1), 4)

    data = {
        "global_homophily": round(profile.homophily, 4),
        "per_community": per_comm,
    }
    summary = (
        f"Global homophily: {profile.homophily:.4f}. "
        f"Per-community range: {min(per_comm.values()):.3f} to {max(per_comm.values()):.3f}."
    )
    return ToolResult(data=data, summary=summary)


def feature_stats(ws: Workspace) -> ToolResult:
    x = ws.graph.node_features()
    n, d = x.shape
    per_feature_entropy = []
    for j in range(min(d, 50)):
        col = x[:, j]
        if col.std() < 1e-10:
            per_feature_entropy.append(0.0)
        else:
            hist, _ = np.histogram(col, bins=20)
            p = hist / hist.sum()
            p = p[p > 0]
            per_feature_entropy.append(round(float(-np.sum(p * np.log2(p))), 3))

    data = {
        "num_features": d,
        "sparsity": round(float((x == 0).sum() / max(x.size, 1)), 4),
        "mean_norm": round(float(np.linalg.norm(x, axis=1).mean()), 4),
        "std_norm": round(float(np.linalg.norm(x, axis=1).std()), 4),
        "avg_entropy": round(float(np.mean(per_feature_entropy)), 3),
        "num_constant_features": int(sum(1 for j in range(d) if x[:, j].std() < 1e-10)),
    }
    summary = (
        f"Features: {d}-dim, sparsity {data['sparsity']:.1%}. "
        f"Mean L2 norm: {data['mean_norm']:.3f} (std {data['std_norm']:.3f}). "
        f"Avg feature entropy: {data['avg_entropy']:.3f}. "
        f"{data['num_constant_features']} constant features."
    )
    return ToolResult(data=data, summary=summary)


def spectral_analysis(ws: Workspace) -> ToolResult:
    profile = ws.graph.profile()
    data = {
        "spectral_gap": round(profile.spectral_gap, 6),
        "interpretation": "large" if profile.spectral_gap > 0.5 else "moderate" if profile.spectral_gap > 0.1 else "small",
    }
    summary = (
        f"Spectral gap: {profile.spectral_gap:.6f} "
        f"({'well-separated clusters' if profile.spectral_gap > 0.5 else 'moderate separation' if profile.spectral_gap > 0.1 else 'weak separation'})."
    )
    return ToolResult(data=data, summary=summary)


def _histogram(values: np.ndarray, bins: int = 20) -> dict[str, Any]:
    counts, edges = np.histogram(values, bins=bins)
    return {
        "counts": counts.tolist(),
        "bin_edges": [round(float(e), 4) for e in edges],
    }
