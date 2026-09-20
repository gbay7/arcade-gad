"""Artifact pattern registry — 5 built-in patterns + agent-discoverable."""

from __future__ import annotations

import numpy as np
from typing import Any

from arcade.models import ArtifactPattern


BUILTIN_ARTIFACTS: list[ArtifactPattern] = [
    ArtifactPattern(
        name="degree_confound",
        artifact_type="confound",
        affected_paradigms=["reconstruction", "spectral", "adversarial", "density"],
        detection_tool="degree_correlation",
        severity_threshold=0.7,
        mitigation="Down-weight paradigm via beta_k when |rho(score, degree)| > 0.7",
        description="High-degree nodes receive high anomaly scores because their neighborhoods are harder to process, not because they are anomalous.",
    ),
    ArtifactPattern(
        name="community_size",
        artifact_type="bias",
        affected_paradigms=["community", "contrastive"],
        detection_tool="community_bias",
        severity_threshold=0.6,
        mitigation="Check if flagged nodes are from one small community; if so, it may be a size effect, not anomaly",
        description="Nodes in small or isolated communities may score high simply because their community is different, not because they are anomalous.",
    ),
    ArtifactPattern(
        name="feature_sparsity",
        artifact_type="bias",
        affected_paradigms=["one_class", "homophily_violation", "attr_struct_mismatch"],
        detection_tool="feature_sparsity_check",
        severity_threshold=0.5,
        mitigation="Check if high-scoring nodes have unusually sparse feature vectors; down-weight if sparsity drives the score",
        description="Nodes with sparse feature vectors score high on attribute-based detectors because zero-heavy vectors are far from dense centroids.",
    ),
    ArtifactPattern(
        name="reconstruction_capacity",
        artifact_type="confound",
        affected_paradigms=["reconstruction", "attr_struct_mismatch"],
        detection_tool="peripheral_check",
        severity_threshold=0.6,
        mitigation="Check if high-scoring nodes are peripheral (low degree, far from core); peripheral position inflates reconstruction error",
        description="Peripheral nodes have high reconstruction error because they are underrepresented in the training distribution, not because they are anomalous.",
    ),
    ArtifactPattern(
        name="detector_redundancy",
        artifact_type="redundancy",
        affected_paradigms=["contrastive", "local_affinity"],
        detection_tool="paradigm_correlation",
        severity_threshold=0.8,
        mitigation="When two paradigms' score vectors correlate > 0.8, they share credit via gamma_k",
        description="Two detectors flagging the same nodes is one signal, not two independent confirmations.",
    ),
]


class ArtifactRegistry:
    """Registry for artifact patterns — built-in + agent-discovered."""

    def __init__(self):
        self._artifacts: dict[str, ArtifactPattern] = {}
        for a in BUILTIN_ARTIFACTS:
            self._artifacts[a.name] = a

    def register(self, artifact: ArtifactPattern) -> None:
        self._artifacts[artifact.name] = artifact

    def get(self, name: str) -> ArtifactPattern | None:
        return self._artifacts.get(name)

    def all(self) -> list[ArtifactPattern]:
        return list(self._artifacts.values())

    def for_paradigm(self, paradigm_name: str) -> list[ArtifactPattern]:
        return [a for a in self._artifacts.values() if paradigm_name in a.affected_paradigms]

    def describe_all(self) -> list[dict[str, Any]]:
        return [
            {
                "name": a.name,
                "type": a.artifact_type,
                "affected_paradigms": a.affected_paradigms,
                "severity_threshold": a.severity_threshold,
                "mitigation": a.mitigation,
                "description": a.description,
                "agent_created": a.agent_created,
            }
            for a in self._artifacts.values()
        ]

    def check_artifact(
        self,
        artifact_name: str,
        scores: np.ndarray,
        degrees: np.ndarray | None = None,
        communities: np.ndarray | None = None,
        features: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run an artifact check and return severity + details."""
        artifact = self._artifacts.get(artifact_name)
        if artifact is None:
            return {"error": f"Unknown artifact: {artifact_name}"}

        if artifact_name == "degree_confound" and degrees is not None:
            rho = float(np.corrcoef(scores, degrees)[0, 1]) if np.std(degrees) > 1e-10 else 0.0
            return {
                "artifact": artifact_name,
                "metric": "pearson_rho_degree",
                "value": round(rho, 4),
                "severity": abs(rho),
                "detected": abs(rho) > artifact.severity_threshold,
                "mitigation": artifact.mitigation,
            }

        if artifact_name == "community_size" and communities is not None:
            top_k = np.argsort(scores)[-50:]
            top_comms = communities[top_k]
            unique, counts = np.unique(top_comms, return_counts=True)
            max_frac = float(counts.max() / len(top_k))
            return {
                "artifact": artifact_name,
                "metric": "max_community_fraction_in_top50",
                "value": round(max_frac, 4),
                "community_distribution": dict(zip(unique.tolist(), counts.tolist())),
                "severity": max_frac,
                "detected": max_frac > artifact.severity_threshold,
                "mitigation": artifact.mitigation,
            }

        if artifact_name == "feature_sparsity" and features is not None:
            top_k = np.argsort(scores)[-50:]
            sparsity_top = float((features[top_k] == 0).sum() / max(features[top_k].size, 1))
            sparsity_all = float((features == 0).sum() / max(features.size, 1))
            diff = sparsity_top - sparsity_all
            return {
                "artifact": artifact_name,
                "metric": "sparsity_diff_top50_vs_all",
                "value": round(diff, 4),
                "top50_sparsity": round(sparsity_top, 4),
                "overall_sparsity": round(sparsity_all, 4),
                "severity": max(0, diff),
                "detected": diff > artifact.severity_threshold * 0.1,
                "mitigation": artifact.mitigation,
            }

        if artifact_name == "reconstruction_capacity" and degrees is not None:
            top_k = np.argsort(scores)[-50:]
            avg_deg_top = float(degrees[top_k].mean())
            avg_deg_all = float(degrees.mean())
            peripheral = avg_deg_top < avg_deg_all * 0.5
            return {
                "artifact": artifact_name,
                "metric": "avg_degree_top50_vs_all",
                "avg_degree_top50": round(avg_deg_top, 2),
                "avg_degree_all": round(avg_deg_all, 2),
                "severity": 0.8 if peripheral else 0.2,
                "detected": peripheral,
                "mitigation": artifact.mitigation,
            }

        if artifact_name == "detector_redundancy":
            return {
                "artifact": artifact_name,
                "metric": "paradigm_correlation",
                "note": "Use paradigm_correlation tool to check pairwise correlations",
                "severity": 0.0,
                "detected": False,
            }

        return {"artifact": artifact_name, "note": "Check not implemented for given inputs", "severity": 0.0}
