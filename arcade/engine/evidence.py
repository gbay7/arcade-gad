"""Append-only evidence store — every finding, artifact detection, and convergence record."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from arcade.models import Evidence, EvidenceType, Finding


class EvidenceStore:
    """Immutable, append-only store for all evidence gathered during analysis."""

    def __init__(self):
        self._evidence: list[Evidence] = []
        self._findings: list[Finding] = []
        self._trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Append operations (never delete)
    # ------------------------------------------------------------------

    def add_evidence(
        self,
        source_paradigm: str,
        evidence_type: EvidenceType,
        data: dict[str, Any],
        interpretation: str = "",
        confidence: float = 0.0,
        related: list[str] | None = None,
    ) -> Evidence:
        ev = Evidence(
            source_paradigm=source_paradigm,
            evidence_type=evidence_type,
            data=data,
            interpretation=interpretation,
            confidence=confidence,
            related=related or [],
        )
        self._evidence.append(ev)
        return ev

    def add_signal(
        self, paradigm: str, nodes: list[int], scores: dict[int, float], interpretation: str = ""
    ) -> Evidence:
        return self.add_evidence(
            source_paradigm=paradigm,
            evidence_type=EvidenceType.SIGNAL,
            data={"nodes": nodes, "scores": scores},
            interpretation=interpretation,
            confidence=len(nodes) / max(len(scores), 1),
        )

    def add_artifact(
        self, paradigm: str, artifact_name: str, severity: float, evidence_data: dict[str, Any]
    ) -> Evidence:
        return self.add_evidence(
            source_paradigm=paradigm,
            evidence_type=EvidenceType.ARTIFACT,
            data={"artifact": artifact_name, "severity": severity, **evidence_data},
            interpretation=f"Artifact '{artifact_name}' detected in {paradigm} with severity {severity:.2f}",
            confidence=severity,
        )

    def add_convergence(self, node_ids: list[int], paradigms: list[str], interpretation: str = "") -> Evidence:
        return self.add_evidence(
            source_paradigm="multi-view",
            evidence_type=EvidenceType.CONVERGENCE,
            data={"nodes": node_ids, "paradigms": paradigms, "count": len(paradigms)},
            interpretation=interpretation or f"Nodes {node_ids[:5]}... flagged by {len(paradigms)} paradigms",
            confidence=min(1.0, len(paradigms) / 3),
        )

    def add_finding(self, finding: Finding) -> None:
        self._findings.append(finding)

    def log_action(self, tool: str, params: dict[str, Any], result_summary: str) -> None:
        self._trace.append({
            "timestamp": time.time(),
            "tool": tool,
            "params": params,
            "result_summary": result_summary,
        })

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_evidence_for_node(self, node_id: int) -> list[Evidence]:
        results = []
        for ev in self._evidence:
            nodes = ev.data.get("nodes", [])
            if node_id in nodes:
                results.append(ev)
        return results

    def get_evidence_by_type(self, ev_type: EvidenceType) -> list[Evidence]:
        return [ev for ev in self._evidence if ev.evidence_type == ev_type]

    def get_evidence_by_paradigm(self, paradigm: str) -> list[Evidence]:
        return [ev for ev in self._evidence if ev.source_paradigm == paradigm]

    def check_convergence(self, node_id: int) -> list[str]:
        """Return paradigms that flagged this node."""
        paradigms = set()
        for ev in self._evidence:
            if ev.evidence_type == EvidenceType.SIGNAL:
                if node_id in ev.data.get("nodes", []):
                    paradigms.add(ev.source_paradigm)
        return sorted(paradigms)

    def conflicts(self) -> list[dict[str, Any]]:
        """Find nodes where paradigms disagree (some flag, some explicitly clear)."""
        flagged: dict[int, set[str]] = {}
        cleared: dict[int, set[str]] = {}

        for ev in self._evidence:
            nodes = ev.data.get("nodes", [])
            if ev.evidence_type == EvidenceType.SIGNAL:
                for n in nodes:
                    flagged.setdefault(n, set()).add(ev.source_paradigm)

        conflict_list = []
        for node_id in flagged:
            if node_id in cleared:
                conflict_list.append({
                    "node": node_id,
                    "flagged_by": sorted(flagged[node_id]),
                    "cleared_by": sorted(cleared[node_id]),
                })
        return conflict_list

    def node_dossier(self, node_id: int) -> dict[str, Any]:
        evidence = self.get_evidence_for_node(node_id)
        converging = self.check_convergence(node_id)
        return {
            "node_id": node_id,
            "evidence_count": len(evidence),
            "converging_paradigms": converging,
            "convergence_count": len(converging),
            "evidence": [
                {
                    "id": ev.id,
                    "type": ev.evidence_type.value,
                    "paradigm": ev.source_paradigm,
                    "interpretation": ev.interpretation,
                    "confidence": ev.confidence,
                }
                for ev in evidence
            ],
        }

    @property
    def findings(self) -> list[Finding]:
        return list(self._findings)

    @property
    def trace(self) -> list[dict[str, Any]]:
        return list(self._trace)

    @property
    def all_evidence(self) -> list[Evidence]:
        return list(self._evidence)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence": [
                {
                    "id": ev.id,
                    "source_paradigm": ev.source_paradigm,
                    "evidence_type": ev.evidence_type.value,
                    "data": ev.data,
                    "interpretation": ev.interpretation,
                    "confidence": ev.confidence,
                    "timestamp": ev.timestamp,
                    "related": ev.related,
                }
                for ev in self._evidence
            ],
            "findings": [
                {
                    "nodes": f.nodes,
                    "confidence": f.confidence,
                    "anomaly_type": f.anomaly_type,
                    "evidence_ids": f.evidence_ids,
                    "artifact_checks_passed": f.artifact_checks_passed,
                    "reasoning": f.reasoning,
                }
                for f in self._findings
            ],
            "trace": self._trace,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> EvidenceStore:
        path = Path(path)
        raw = json.loads(path.read_text())
        store = cls()
        for ev_data in raw.get("evidence", []):
            ev = Evidence(
                id=ev_data["id"],
                source_paradigm=ev_data["source_paradigm"],
                evidence_type=EvidenceType(ev_data["evidence_type"]),
                data=ev_data["data"],
                interpretation=ev_data["interpretation"],
                confidence=ev_data["confidence"],
                timestamp=ev_data["timestamp"],
                related=ev_data.get("related", []),
            )
            store._evidence.append(ev)
        for f_data in raw.get("findings", []):
            store._findings.append(Finding(**f_data))
        store._trace = raw.get("trace", [])
        return store
