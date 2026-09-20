"""Decision protocol for the LLM ablation (Mode B).

Every model receives the SAME evidence pack — the label-free context ARCADE's
analyst works from — and answers the same structured question: which paradigms
to compose. The pack is built exclusively from MCP tool results (graph_profile,
applicable_paradigms, run_fast_paradigms / score_distribution stats). It is
deliberately provenance-agnostic: nothing says whether anomalies are injected
or organic, and dataset names are replaced by an opaque id so no model can
pattern-match benchmark lore.

Labels never appear anywhere in the protocol; they are used once, outside the
model, to measure the resulting composition.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

MAX_SELECTION = 4

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selected_paradigms": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": MAX_SELECTION,
            "description": "Names of the paradigms to run and compose, most important first.",
        },
        "rationale": {
            "type": "string",
            "description": "Why these paradigms match the graph's evidence, citing profile fields.",
        },
    },
    "required": ["selected_paradigms", "rationale"],
}

SYSTEM_PROMPT = (
    "You are the analyst of an unsupervised graph anomaly detection framework. "
    "You are given a label-free profile of an attributed graph and the catalog of "
    "applicable detection paradigms, each with the assumption it relies on. "
    "You do not know how the graph was built or what its anomalies look like; "
    "reason only from the evidence given. Select the small set of paradigms "
    f"(1 to {MAX_SELECTION}) whose assumptions the evidence supports, favouring "
    "complementary views over redundant ones. Reply with JSON only."
)

# Checklist protocol (C8): converts free-recall selection into per-paradigm
# verification — the analyst must give a verdict with evidence for EVERY
# applicable paradigm before selecting. Verification is the mode weaker models
# handle better than open-ended recall, and it forces catalog coverage.
CHECKLIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paradigm": {"type": "string"},
                    "evidence": {"type": "string",
                                 "description": "Profile/landmarker facts for or against this paradigm."},
                    "verdict": {"type": "string", "enum": ["select", "reject"]},
                },
                "required": ["paradigm", "evidence", "verdict"],
            },
            "description": "One verdict per applicable paradigm — cover the whole catalog.",
        },
        "selected_paradigms": {
            "type": "array", "items": {"type": "string"},
            "minItems": 1, "maxItems": MAX_SELECTION,
        },
        "rationale": {"type": "string"},
    },
    "required": ["verdicts", "selected_paradigms", "rationale"],
}

CHECKLIST_SYSTEM_PROMPT = (
    "You are the analyst of an unsupervised graph anomaly detection framework. "
    "You are given a label-free profile of an attributed graph and the catalog of "
    "applicable detection paradigms. You do not know how the graph was built; "
    "reason only from the evidence given. Work as a checklist: for EVERY paradigm "
    "in the catalog, state the evidence for or against it and give a verdict "
    "(select or reject). Then list the selected paradigms "
    f"(1 to {MAX_SELECTION}, the ones whose verdict is select, most important "
    "first). Reply with JSON only."
)


DOMAIN_SYSTEM_PROMPT = (
    "You are the analyst of an unsupervised graph anomaly detection framework, "
    "working like a domain expert: you are told what the graph is and what the "
    "practitioner is looking for, and you are given a label-free profile of the "
    "graph and the catalog of applicable detection paradigms, each with the "
    "assumption it relies on. You have no labels. Reason from the domain and the "
    f"evidence together. Select the small set of paradigms (1 to {MAX_SELECTION}) "
    "whose assumptions fit this graph and this task, favouring complementary views "
    "over redundant ones. Reply with JSON only."
)


@dataclass
class EvidencePack:
    dataset_key: str            # internal only; never shown to the model
    profile: dict[str, Any]
    paradigms: list[dict[str, Any]]          # applicable: name/assumption/description
    fast_stats: dict[str, dict[str, Any]]    # per fast paradigm: score-distribution stats
    domain: str | None = None                # optional expert description of the graph and task
    legend: dict[str, Any] | None = None     # optional definitions of every profile/landmarker field

    def to_prompt(self) -> str:
        profile_view = {k: v for k, v in self.profile.items()}
        # Key order is the reading order. The four middle keys are present only in
        # the enriched-catalog arm (ABLATION_RICH=1); the study's packs carry the
        # one-line description alone.
        catalog = [
            {k: v for k, v in {
                "name": p["name"], "assumption": p.get("assumption", ""),
                "computes": p.get("computes", ""),
                "reference_frame": p.get("reference_frame", ""),
                "inputs": p.get("inputs", ""),
                "degenerates_when": p.get("degenerates_when", ""),
                "provenance": p.get("provenance", ""),
                "description": p.get("description", ""),
                # evidence-grounding: WHY the precondition holds, with values
                "applicable_because": p.get("applicable_because", ""),
            }.items() if v}
            for p in self.paradigms
        ]
        parts = []
        if self.domain:
            parts += ["DATASET DESCRIPTION (what this graph is and what the practitioner is looking for; no labels):",
                      self.domain, ""]
        if self.legend:
            parts += ["HOW TO READ THE EVIDENCE (definitions of the fields below; "
                      "read them literally, some names are narrower than they sound):",
                      json.dumps(self.legend, indent=1), ""]
        parts += [
            "GRAPH PROFILE (label-free):",
            json.dumps(profile_view, indent=1),
            "",
            "APPLICABLE PARADIGMS (you may only select from these):",
            json.dumps(catalog, indent=1),
        ]
        if self.fast_stats:
            parts += [
                "",
                # The shipped header tells the analyst what to conclude from the
                # block ("a strongly separated upper tail suggests the mechanism is
                # present"). Traces show small models then rank paradigms by p99:
                # on amazon_fraud qwen3:8b reasoned "looking for paradigms with high
                # p99 values", took spectral (33.1 alone) over local_affinity (59.9)
                # and scored 32.4. It is the same defect as the first legend, which
                # was removed after it cost 39 points on Books. The neutral header
                # describes the block without saying what it implies; the shipped
                # wording is kept for reproducing the frozen study.
                ("FAST-PARADIGM SCORE DISTRIBUTIONS (label-free landmarkers; the score "
                 "distribution of each cheap paradigm on this graph. Normalisation is "
                 "per paradigm, and a distribution cannot show the sign of a signal):"
                 if os.environ.get("ABLATION_NEUTRAL_HEADER") == "1" or
                    os.environ.get("ABLATION_RICH") == "1" else
                 "FAST-PARADIGM SCORE DISTRIBUTIONS (label-free landmarkers; a strongly "
                 "separated upper tail suggests the paradigm's mechanism is present):"),
                json.dumps(self.fast_stats, indent=1),
            ]
        parts += [
            "",
            "Select the paradigms to compose for this graph and justify from the evidence.",
        ]
        return "\n".join(parts)

    def messages(self, checklist: bool = False) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": CHECKLIST_SYSTEM_PROMPT if checklist
             else (DOMAIN_SYSTEM_PROMPT if self.domain else SYSTEM_PROMPT)},
            {"role": "user", "content": self.to_prompt()},
        ]


@dataclass
class Decision:
    selected: list[str]
    rationale: str
    valid_json: bool
    all_applicable: bool
    invalid_names: list[str]

    @classmethod
    def from_response(cls, parsed: dict[str, Any] | None,
                      applicable_names: list[str]) -> "Decision":
        if parsed is None:
            return cls([], "", valid_json=False, all_applicable=False, invalid_names=[])
        raw = parsed.get("selected_paradigms", []) or []
        seen: list[str] = []
        for name in raw:                      # dedupe, preserve order, cap
            if isinstance(name, str) and name not in seen:
                seen.append(name)
        seen = seen[:MAX_SELECTION]
        invalid = [n for n in seen if n not in applicable_names]
        kept = [n for n in seen if n in applicable_names]
        return cls(
            selected=kept,
            rationale=str(parsed.get("rationale", "")),
            valid_json=True,
            all_applicable=not invalid,
            invalid_names=invalid,
        )
