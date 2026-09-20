"""Core data model for GUIDE — Tool, Paradigm, Evidence, Artifact."""

from __future__ import annotations

import time
import enum
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ToolCategory(str, enum.Enum):
    EXPLORE = "explore"
    MEASURE = "measure"
    DIAGNOSE = "diagnose"
    INVESTIGATE = "investigate"
    COMPOSE = "compose"


class EvidenceType(str, enum.Enum):
    SIGNAL = "signal"
    ARTIFACT = "artifact"
    CONVERGENCE = "convergence"
    CONFLICT = "conflict"
    EXPLANATION = "explanation"


class AnalystState(str, enum.Enum):
    INIT = "init"
    EXPLORE = "explore"
    MEASURE = "measure"
    DIAGNOSE = "diagnose"
    INVESTIGATE = "investigate"
    REPORT = "report"
    EXTEND = "extend"


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    data: dict[str, Any]
    summary: str
    plots: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "data": self.data,
            "summary": self.summary,
            "plots": self.plots,
            "metadata": self.metadata,
            "provenance": self.provenance,
        }


@dataclass
class FailResult:
    error: str
    partial_data: dict[str, Any] | None = None
    suggestion: str = ""

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"error": self.error}
        if self.partial_data:
            d["partial_data"] = self.partial_data
        if self.suggestion:
            d["suggestion"] = self.suggestion
        return d


# ---------------------------------------------------------------------------
# Weight triplet
# ---------------------------------------------------------------------------

@dataclass
class Weight:
    alpha: float  # relevance (0 or 1)
    beta: float   # reliability [0,1]
    gamma: float  # independence (0,1]
    delta: float = 1.0  # consensus agreement (+1, 0, -1)
    excess_mass: float = 0.0  # I1: EM quality score
    mechanism_flag: int = 1  # I3: 0=mechanism (don't penalize deg), 1=artifact
    fusion_mode: str = "fuse"  # I4: fuse / select / hybrid
    overrides: list[dict[str, Any]] = field(default_factory=list)

    @property
    def w(self) -> float:
        return self.alpha * self.beta * self.gamma * self.delta


# ---------------------------------------------------------------------------
# Paradigm
# ---------------------------------------------------------------------------

@dataclass
class Paradigm:
    name: str
    assumption: str
    detector_name: str
    known_artifacts: list[str] = field(default_factory=list)
    applicability: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    agent_created: bool = False

    def is_applicable(self, profile: GraphProfile) -> bool:
        for prop, condition in self.applicability.items():
            val = profile.properties.get(prop)
            if val is None:
                continue
            op = condition.get("op", ">=")
            threshold = condition.get("threshold", 0)
            if op == ">=" and val < threshold:
                return False
            if op == "<=" and val > threshold:
                return False
            if op == ">" and val <= threshold:
                return False
            if op == "<" and val >= threshold:
                return False
        return True

    def grounding(self, profile: GraphProfile) -> str:
        """Render WHY this paradigm is applicable, with the actual profile values.

        Applicability alone names a candidate; grounding states the evidence
        ("feature_dup_rate = 0.57 satisfies >= 0.4"), which is what an analyst
        needs to connect the precondition to the mechanism. Stating satisfied
        preconditions is evidence completeness, not a recommendation — the
        selection among grounded candidates remains the analyst's.
        """
        if not self.applicability:
            return "no structural precondition (general-purpose paradigm)"
        parts = []
        for prop, condition in self.applicability.items():
            val = profile.properties.get(prop)
            op = condition.get("op", ">=")
            threshold = condition.get("threshold", 0)
            shown = f"{val:.3g}" if isinstance(val, float) else str(val)
            parts.append(f"{prop} = {shown} satisfies {op} {threshold}")
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Artifact pattern
# ---------------------------------------------------------------------------

@dataclass
class ArtifactPattern:
    name: str
    artifact_type: str  # confound | redundancy | bias
    affected_paradigms: list[str]
    detection_tool: str
    severity_threshold: float = 0.7
    mitigation: str = ""
    description: str = ""
    agent_created: bool = False


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    id: str = ""
    source_paradigm: str = ""
    evidence_type: EvidenceType = EvidenceType.SIGNAL
    data: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    related: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            raw = f"{self.source_paradigm}:{self.evidence_type.value}:{self.timestamp}"
            self.id = hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class Finding:
    nodes: list[int]
    confidence: str  # high | medium | low
    anomaly_type: str
    evidence_ids: list[str] = field(default_factory=list)
    artifact_checks_passed: list[str] = field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Graph profile
# ---------------------------------------------------------------------------

@dataclass
class GraphProfile:
    num_nodes: int = 0
    num_edges: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    max_degree: int = 0
    degree_std: float = 0.0
    clustering_coeff: float = 0.0
    num_communities: int = 0
    modularity: float = 0.0
    homophily: float = 0.0
    feature_dim: int = 0
    feature_sparsity: float = 0.0
    feature_dup_rate: float = 0.0
    spectral_gap: float = 0.0
    is_power_law: bool = False
    properties: dict[str, Any] = field(default_factory=dict)
    domain_context: str = ""

    def __post_init__(self):
        self.properties.update({
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.density,
            "avg_degree": self.avg_degree,
            "clustering_coeff": self.clustering_coeff,
            "num_communities": self.num_communities,
            "modularity": self.modularity,
            "homophily": self.homophily,
            "feature_dim": self.feature_dim,
            "feature_sparsity": self.feature_sparsity,
            "feature_dup_rate": self.feature_dup_rate,
            "spectral_gap": self.spectral_gap,
        })
