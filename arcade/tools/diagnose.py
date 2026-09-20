"""Diagnose tools — distributions, artifact checks, paradigm correlations."""

from __future__ import annotations

import numpy as np
from typing import Any

from arcade.models import ToolResult, FailResult, EvidenceType
from arcade.engine.workspace import Workspace


def score_distribution(ws: Workspace, paradigm: str, bins: int = 50) -> ToolResult:
    scores = ws.scoring.get_scores(paradigm)
    if scores is None:
        return ToolResult(data={"error": f"No scores for {paradigm}"}, summary=f"Run '{paradigm}' first.")

    counts, edges = np.histogram(scores, bins=bins)

    # Detect modes (simple peak detection)
    modes = []
    for i in range(1, len(counts) - 1):
        if counts[i] > counts[i-1] and counts[i] > counts[i+1]:
            modes.append({
                "bin_center": round(float((edges[i] + edges[i+1]) / 2), 4),
                "count": int(counts[i]),
            })

    # Detect tail spike
    tail_threshold = np.percentile(scores, 90)
    tail_nodes = int((scores > tail_threshold).sum())
    p95 = float(np.percentile(scores, 95))
    tail_counts = counts[-5:]
    has_spike = any(int(c) > int(np.mean(counts)) * 2 for c in tail_counts)

    data = {
        "paradigm": paradigm,
        "histogram": {"counts": counts.tolist(), "bin_edges": [round(float(e), 4) for e in edges]},
        "modes": modes,
        "num_modes": len(modes),
        "is_bimodal": len(modes) >= 2,
        "tail_spike": has_spike,
        "nodes_in_tail": tail_nodes,
        "p95_value": round(p95, 4),
    }
    summary = (
        f"Distribution for '{paradigm}': {len(modes)} mode(s). "
        f"{'Bimodal — two populations detected. ' if len(modes) >= 2 else ''}"
        f"{'Tail spike detected — secondary bump in high scores. ' if has_spike else ''}"
        f"{tail_nodes} nodes above P90 threshold."
    )
    return ToolResult(data=data, summary=summary)


def degree_correlation(ws: Workspace, paradigm: str) -> ToolResult:
    scores = ws.scoring.get_scores(paradigm)
    if scores is None:
        return ToolResult(data={"error": f"No scores for {paradigm}"}, summary=f"Run '{paradigm}' first.")

    degs = ws.graph.degrees()
    rho = float(np.corrcoef(scores, degs)[0, 1]) if np.std(degs) > 1e-10 else 0.0

    is_artifact = abs(rho) > 0.7
    severity = abs(rho)

    if is_artifact:
        ws.evidence.add_artifact(
            paradigm=paradigm,
            artifact_name="degree_confound",
            severity=severity,
            evidence_data={"rho": round(rho, 4)},
        )

    data = {
        "paradigm": paradigm,
        "pearson_rho": round(rho, 4),
        "abs_rho": round(abs(rho), 4),
        "is_degree_artifact": is_artifact,
        "severity": round(severity, 4),
        "interpretation": (
            f"Strong degree confound (ρ={rho:.3f}). Scores mainly reflect degree, not anomaly."
            if is_artifact else
            f"Weak degree correlation (ρ={rho:.3f}). Scores are not driven by degree."
        ),
    }
    summary = (
        f"Degree correlation for '{paradigm}': ρ={rho:.4f}. "
        f"{'ARTIFACT: degree confound detected — down-weight this paradigm.' if is_artifact else 'OK: no degree confound.'}"
    )
    return ToolResult(data=data, summary=summary)


def community_bias(ws: Workspace, paradigm: str) -> ToolResult:
    scores = ws.scoring.get_scores(paradigm)
    if scores is None:
        return ToolResult(data={"error": f"No scores for {paradigm}"}, summary=f"Run '{paradigm}' first.")

    comms = ws.graph.communities()
    top_50 = np.argsort(scores)[-50:]
    top_comms = comms[top_50]
    unique, counts = np.unique(top_comms, return_counts=True)
    max_frac = float(counts.max() / len(top_50))
    dominant_comm = int(unique[counts.argmax()])

    is_artifact = max_frac > 0.6
    if is_artifact:
        ws.evidence.add_artifact(
            paradigm=paradigm,
            artifact_name="community_size",
            severity=max_frac,
            evidence_data={"dominant_community": dominant_comm, "fraction": round(max_frac, 4)},
        )

    per_comm_mean = {}
    for c in range(len(np.unique(comms))):
        mask = comms == c
        if mask.sum() > 0:
            per_comm_mean[int(c)] = round(float(scores[mask].mean()), 4)

    data = {
        "paradigm": paradigm,
        "top50_community_distribution": dict(zip(unique.tolist(), counts.tolist())),
        "max_community_fraction": round(max_frac, 4),
        "dominant_community": dominant_comm,
        "is_community_artifact": is_artifact,
        "per_community_mean_score": per_comm_mean,
    }
    summary = (
        f"Community bias for '{paradigm}': {max_frac:.0%} of top-50 nodes in community {dominant_comm}. "
        f"{'ARTIFACT: community size bias detected.' if is_artifact else 'OK: top nodes spread across communities.'}"
    )
    return ToolResult(data=data, summary=summary)


def paradigm_correlation(ws: Workspace) -> ToolResult:
    corr = ws.scoring.paradigm_correlation_matrix()
    names = corr["paradigms"]
    matrix = np.array(corr["correlation_matrix"])

    high_pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = matrix[i, j]
            if abs(r) > 0.7:
                high_pairs.append({
                    "paradigm_a": names[i],
                    "paradigm_b": names[j],
                    "correlation": round(float(r), 4),
                    "is_redundant": bool(abs(r) > 0.8),
                })

    data = {
        "paradigms": names,
        "correlation_matrix": np.nan_to_num(matrix, nan=0.0).tolist(),
        "high_correlation_pairs": high_pairs,
        "num_redundant": sum(1 for p in high_pairs if p["is_redundant"]),
    }
    summary = (
        f"Paradigm correlation: {len(names)} paradigms. "
        f"{len(high_pairs)} highly correlated pairs (>0.7). "
        f"{data['num_redundant']} redundant pairs (>0.8)."
    )
    return ToolResult(data=data, summary=summary)


def tail_analysis(ws: Workspace, paradigm: str, tail_pct: float = 95) -> ToolResult:
    scores = ws.scoring.get_scores(paradigm)
    if scores is None:
        return ToolResult(data={"error": f"No scores for {paradigm}"}, summary=f"Run '{paradigm}' first.")

    threshold = np.percentile(scores, tail_pct)
    tail_mask = scores > threshold
    tail_nodes = np.where(tail_mask)[0]
    tail_scores = scores[tail_mask]

    from scipy.stats import ks_2samp
    body = scores[~tail_mask]
    if len(tail_scores) > 1 and len(body) > 1:
        ks_stat, ks_p = ks_2samp(tail_scores, body)
    else:
        ks_stat, ks_p = 0.0, 1.0

    gap = float(threshold - np.percentile(scores, tail_pct - 5))

    data = {
        "paradigm": paradigm,
        "tail_percentile": tail_pct,
        "threshold": round(float(threshold), 4),
        "num_tail_nodes": len(tail_nodes),
        "tail_node_ids": tail_nodes.tolist()[:100],
        "tail_mean": round(float(tail_scores.mean()), 4) if len(tail_scores) > 0 else 0,
        "tail_std": round(float(tail_scores.std()), 4) if len(tail_scores) > 0 else 0,
        "gap_from_body": round(gap, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_p, 6),
        "distinct_population": bool(ks_p < 0.01),
    }
    summary = (
        f"Tail analysis for '{paradigm}' (P{tail_pct}): {len(tail_nodes)} nodes above {threshold:.4f}. "
        f"Gap from body: {gap:.4f}. "
        f"{'Distinct population (KS p={:.4f}).'.format(ks_p) if ks_p < 0.01 else 'Not a clearly distinct population.'}"
    )
    return ToolResult(data=data, summary=summary)


def artifact_check(ws: Workspace, paradigm: str, artifact_name: str) -> ToolResult:
    from arcade.knowledge import ArtifactRegistry
    ar = ArtifactRegistry()

    scores = ws.scoring.get_scores(paradigm)
    if scores is None:
        return ToolResult(data={"error": f"No scores for {paradigm}"}, summary=f"Run '{paradigm}' first.")

    result = ar.check_artifact(
        artifact_name,
        scores,
        degrees=ws.graph.degrees(),
        communities=ws.graph.communities(),
        features=ws.graph.node_features(),
    )

    if result.get("detected", False):
        ws.evidence.add_artifact(
            paradigm=paradigm,
            artifact_name=artifact_name,
            severity=result.get("severity", 0.0),
            evidence_data=result,
        )

    data = result
    detected = result.get("detected", False)
    summary = (
        f"Artifact check '{artifact_name}' on '{paradigm}': "
        f"{'DETECTED (severity {:.2f}). {}'.format(result.get('severity', 0), result.get('mitigation', '')) if detected else 'Not detected.'}"
    )
    return ToolResult(data=data, summary=summary)
