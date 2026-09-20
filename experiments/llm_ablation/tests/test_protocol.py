"""Unit tests for the decision protocol (no network, no GPU)."""
import json

from experiments.llm_ablation.protocol import (DECISION_SCHEMA, MAX_SELECTION,
                                               Decision, EvidencePack)

APPLICABLE = ["reconstruction", "local_affinity", "subgraph_contrast", "clique_density"]


def test_valid_decision_kept():
    d = Decision.from_response(
        {"selected_paradigms": ["subgraph_contrast", "clique_density"],
         "rationale": "dense graph"}, APPLICABLE)
    assert d.valid_json and d.all_applicable
    assert d.selected == ["subgraph_contrast", "clique_density"]


def test_invalid_names_filtered_but_reported():
    d = Decision.from_response(
        {"selected_paradigms": ["subgraph_contrast", "made_up"], "rationale": ""},
        APPLICABLE)
    assert d.valid_json and not d.all_applicable
    assert d.selected == ["subgraph_contrast"]
    assert d.invalid_names == ["made_up"]


def test_duplicates_and_cap():
    many = ["reconstruction"] * 3 + APPLICABLE + APPLICABLE
    d = Decision.from_response({"selected_paradigms": many, "rationale": ""}, APPLICABLE)
    assert len(d.selected) <= MAX_SELECTION
    assert len(set(d.selected)) == len(d.selected)


def test_unparsed_response_is_invalid():
    d = Decision.from_response(None, APPLICABLE)
    assert not d.valid_json and d.selected == []


def test_evidence_pack_hides_dataset_and_provenance():
    pack = EvidencePack(
        dataset_key="inj_cora",
        profile={"num_nodes": 10, "avg_degree": 2.0},
        paradigms=[{"name": "local_affinity", "assumption": "a", "description": "d"}],
        fast_stats={"local_affinity": {"mean": 0.1}})
    prompt = pack.to_prompt()
    assert "inj_cora" not in prompt          # dataset identity hidden
    assert "injected" not in prompt.lower()  # no provenance leak
    json.dumps(DECISION_SCHEMA)              # schema serializable
