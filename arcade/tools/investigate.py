"""Investigate tools — node-level probes, case files, cross-paradigm views."""

from __future__ import annotations

import numpy as np
from typing import Any

from arcade.models import ToolResult, EvidenceType
from arcade.engine.workspace import Workspace


def node_card(ws: Workspace, node_id: int) -> ToolResult:
    x = ws.graph.node_features([node_id])[0]
    degs = ws.graph.degrees()
    nbrs = ws.graph.neighbors(node_id)
    comm = ws.graph.community_of(node_id)

    per_paradigm = {}
    for name in ws.scoring.available_paradigms:
        scores = ws.scoring.get_scores(name)
        if scores is not None:
            rank = float((scores < scores[node_id]).sum() / len(scores))
            per_paradigm[name] = {
                "score": round(float(scores[node_id]), 4),
                "percentile": round(rank * 100, 1),
            }

    evidence = ws.evidence.node_dossier(node_id)

    data = {
        "node_id": node_id,
        "degree": int(degs[node_id]),
        "community": comm,
        "num_neighbors": len(nbrs),
        "feature_norm": round(float(np.linalg.norm(x)), 4),
        "feature_sparsity": round(float((x == 0).sum() / len(x)), 4),
        "paradigm_scores": per_paradigm,
        "evidence": evidence,
    }

    flagged_count = sum(1 for p in per_paradigm.values() if p["percentile"] > 90)
    summary = (
        f"Node {node_id}: degree {int(degs[node_id])}, community {comm}, "
        f"{len(nbrs)} neighbors, feature norm {float(np.linalg.norm(x)):.3f}. "
        f"Flagged by {flagged_count}/{len(per_paradigm)} paradigms (>P90). "
        f"{evidence['convergence_count']} paradigms converge."
    )
    return ToolResult(data=data, summary=summary)


def neighborhood_probe(ws: Workspace, node_id: int) -> ToolResult:
    x = ws.graph.node_features()
    nbrs = ws.graph.neighbors(node_id)
    if not nbrs:
        return ToolResult(
            data={"node_id": node_id, "isolated": True},
            summary=f"Node {node_id} is isolated (no neighbors).",
        )

    node_feat = x[node_id]
    nbr_feats = x[nbrs]
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(node_feat.reshape(1, -1), nbr_feats)[0]

    comms = ws.graph.communities()
    nbr_comms = comms[nbrs]
    same_comm = int((nbr_comms == comms[node_id]).sum())

    nbr_scores = {}
    for name in ws.scoring.available_paradigms:
        s = ws.scoring.get_scores(name)
        if s is not None:
            nbr_scores[name] = {
                "node_score": round(float(s[node_id]), 4),
                "neighbor_mean": round(float(s[nbrs].mean()), 4),
                "neighbor_std": round(float(s[nbrs].std()), 4),
            }

    data = {
        "node_id": node_id,
        "num_neighbors": len(nbrs),
        "neighbor_ids": nbrs[:50],
        "feature_similarity": {
            "mean": round(float(sims.mean()), 4),
            "min": round(float(sims.min()), 4),
            "max": round(float(sims.max()), 4),
        },
        "same_community_neighbors": same_comm,
        "cross_community_neighbors": len(nbrs) - same_comm,
        "neighbor_scores": nbr_scores,
    }
    summary = (
        f"Node {node_id} neighborhood: {len(nbrs)} neighbors. "
        f"Avg feature similarity: {float(sims.mean()):.3f}. "
        f"{same_comm}/{len(nbrs)} in same community. "
        f"{'Low coherence — features differ from neighborhood.' if sims.mean() < 0.5 else 'High coherence.'}"
    )
    return ToolResult(data=data, summary=summary)


def cross_paradigm_view(ws: Workspace, node_ids: list[int]) -> ToolResult:
    paradigms = ws.scoring.available_paradigms
    if not paradigms:
        return ToolResult(data={"error": "No paradigms run"}, summary="Run paradigms first.")

    views = []
    for nid in node_ids[:50]:
        row = {"node_id": nid}
        for p in paradigms:
            s = ws.scoring.get_scores(p)
            if s is not None:
                rank = float((s < s[nid]).sum() / len(s)) * 100
                row[p] = {"score": round(float(s[nid]), 4), "percentile": round(rank, 1)}
        converging = ws.evidence.check_convergence(nid)
        row["convergence_count"] = len(converging)
        row["converging_paradigms"] = converging
        views.append(row)

    views.sort(key=lambda r: r.get("convergence_count", 0), reverse=True)

    data = {"nodes": views, "paradigms": paradigms}
    if views:
        top = views[0]
        summary = (
            f"Cross-paradigm view for {len(views)} nodes across {len(paradigms)} paradigms. "
            f"Most converging: node {top['node_id']} ({top['convergence_count']} paradigms agree)."
        )
    else:
        summary = "No nodes to analyze."
    return ToolResult(data=data, summary=summary)


def case_file(ws: Workspace, node_ids: list[int]) -> ToolResult:
    cases = []
    for nid in node_ids[:20]:
        dossier = ws.evidence.node_dossier(nid)
        degs = ws.graph.degrees()
        nbrs = ws.graph.neighbors(nid)

        per_paradigm = {}
        for name in ws.scoring.available_paradigms:
            s = ws.scoring.get_scores(name)
            if s is not None:
                per_paradigm[name] = round(float(s[nid]), 4)

        cases.append({
            "node_id": nid,
            "degree": int(degs[nid]),
            "community": int(ws.graph.community_of(nid)),
            "num_neighbors": len(nbrs),
            "paradigm_scores": per_paradigm,
            "convergence_count": dossier["convergence_count"],
            "converging_paradigms": dossier["converging_paradigms"],
            "evidence_count": dossier["evidence_count"],
        })

    data = {"cases": cases, "num_nodes": len(cases)}
    summary = (
        f"Case file for {len(cases)} nodes. "
        f"Highest convergence: node {cases[0]['node_id']} ({cases[0]['convergence_count']} paradigms)."
        if cases else "No nodes."
    )
    return ToolResult(data=data, summary=summary)


def fused_scores(ws: Workspace, top_k: int = 50) -> ToolResult:
    if not ws.scoring.weights:
        ws.scoring.compute_weights(ws.graph.profile(), ws.graph.degrees(), ws.graph.communities())

    top = ws.scoring.top_k(top_k)
    summary_data = ws.scoring.score_summary()

    data = {"top_k": top, "summary": summary_data}
    summary = (
        f"Fused scores: mode={summary_data.get('fusion_mode', 'unknown')}. "
        f"Top node: {top[0]['node_id']} ({top[0]['fused_score']:.4f})"
        if top else "No fused scores available."
    )
    return ToolResult(data=data, summary=summary)


def find_convergent_nodes(ws: Workspace, min_paradigms: int = 3, top_k: int = 50) -> ToolResult:
    paradigms = ws.scoring.available_paradigms
    if not paradigms:
        return ToolResult(data={"error": "No paradigms run"}, summary="Run paradigms first.")

    n = ws.graph.num_nodes
    convergence_counts = np.zeros(n, dtype=int)

    for name in paradigms:
        s = ws.scoring.get_scores(name)
        if s is not None:
            p95 = np.percentile(s, 95)
            convergence_counts[s > p95] += 1

    qualified = np.where(convergence_counts >= min_paradigms)[0]
    sorted_idx = qualified[np.argsort(convergence_counts[qualified])[::-1]][:top_k]

    results = []
    for nid in sorted_idx:
        flagging_paradigms = []
        for name in paradigms:
            s = ws.scoring.get_scores(name)
            if s is not None and s[nid] > np.percentile(s, 95):
                flagging_paradigms.append(name)
        results.append({
            "node_id": int(nid),
            "convergence_count": int(convergence_counts[nid]),
            "flagging_paradigms": flagging_paradigms,
        })

        ws.evidence.add_convergence(
            node_ids=[int(nid)],
            paradigms=flagging_paradigms,
            interpretation=f"Node {nid} flagged by {len(flagging_paradigms)} independent paradigms (>P95)",
        )

    data = {
        "min_paradigms": min_paradigms,
        "num_convergent": len(results),
        "total_paradigms": len(paradigms),
        "nodes": results,
    }
    summary = (
        f"Found {len(results)} nodes flagged by ≥{min_paradigms} paradigms (out of {len(paradigms)} total). "
        f"{'Top node: ' + str(results[0]['node_id']) + ' (' + str(results[0]['convergence_count']) + ' paradigms).' if results else 'No convergent nodes found.'}"
    )
    return ToolResult(data=data, summary=summary)
