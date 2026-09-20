"""Agent workspace — serializable state for the entire analysis session."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from arcade.models import AnalystState, GraphProfile, Paradigm, ArtifactPattern
from arcade.engine.evidence import EvidenceStore
from arcade.engine.scoring import ScoringEngine
from arcade.data.loader import GraphStore


class Workspace:
    """Holds all session state: graph, profile, scores, evidence, extensions."""

    def __init__(self, graph_store: GraphStore):
        self.graph = graph_store
        self.evidence = EvidenceStore()
        self.scoring = ScoringEngine()
        self.state = AnalystState.INIT
        self.profile: GraphProfile | None = None

        # Runtime extensions
        self.custom_paradigms: list[Paradigm] = []
        self.custom_artifacts: list[ArtifactPattern] = []
        self.annotations: dict[str, Any] = {}
        self.weight_overrides: list[dict[str, Any]] = []

        self._session_start = time.time()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def transition(self, new_state: AnalystState) -> str:
        old = self.state
        self.state = new_state
        self.evidence.log_action(
            "state_transition",
            {"from": old.value, "to": new_state.value},
            f"Transitioned from {old.value} to {new_state.value}",
        )
        return f"State: {old.value} → {new_state.value}"

    # ------------------------------------------------------------------
    # Extension points
    # ------------------------------------------------------------------

    def register_custom_paradigm(self, paradigm: Paradigm) -> str:
        paradigm.agent_created = True
        self.custom_paradigms.append(paradigm)
        self.scoring.register_paradigm(paradigm)
        return f"Registered custom paradigm: {paradigm.name}"

    def register_custom_artifact(self, artifact: ArtifactPattern) -> str:
        artifact.agent_created = True
        self.custom_artifacts.append(artifact)
        return f"Registered custom artifact pattern: {artifact.name}"

    def add_annotation(self, key: str, value: Any) -> None:
        self.annotations[key] = value

    # ------------------------------------------------------------------
    # Session info
    # ------------------------------------------------------------------

    def session_info(self) -> dict[str, Any]:
        elapsed = time.time() - self._session_start
        return {
            "state": self.state.value,
            "graph": self.graph.name if self.graph.loaded else "none",
            "num_nodes": self.graph.num_nodes if self.graph.loaded else 0,
            "paradigms_scored": len(self.scoring.available_paradigms),
            "evidence_count": len(self.evidence.all_evidence),
            "findings_count": len(self.evidence.findings),
            "custom_paradigms": len(self.custom_paradigms),
            "custom_artifacts": len(self.custom_artifacts),
            "elapsed_seconds": round(elapsed, 1),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> str:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.evidence.save(directory / "evidence.json")

        scores_data = {}
        for name in self.scoring.available_paradigms:
            s = self.scoring.get_scores(name)
            if s is not None:
                scores_data[name] = s.tolist()

        (directory / "scores.json").write_text(json.dumps(scores_data, default=str))

        meta = {
            "state": self.state.value,
            "graph_name": self.graph.name if self.graph.loaded else "none",
            "session_start": self._session_start,
            "custom_paradigms": [
                {"name": p.name, "assumption": p.assumption, "detector_name": p.detector_name}
                for p in self.custom_paradigms
            ],
            "custom_artifacts": [
                {"name": a.name, "artifact_type": a.artifact_type, "affected_paradigms": a.affected_paradigms}
                for a in self.custom_artifacts
            ],
            "annotations": self.annotations,
            "weights": {
                name: {
                    "alpha": w.alpha, "beta": w.beta, "gamma": w.gamma,
                    "delta": w.delta, "excess_mass": w.excess_mass,
                    "mechanism_flag": w.mechanism_flag, "fusion_mode": w.fusion_mode,
                    "overrides": w.overrides,
                }
                for name, w in self.scoring.weights.items()
            },
        }
        (directory / "workspace.json").write_text(json.dumps(meta, indent=2, default=str))

        return str(directory)
