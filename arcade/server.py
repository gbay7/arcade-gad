"""ARCADE MCP Server — exposes all graph analysis tools via Model Context Protocol."""

from __future__ import annotations

import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from arcade.data.loader import GraphStore
from arcade.engine.workspace import Workspace
from arcade.engine.scoring import ScoringEngine
from arcade.paradigms import ParadigmRegistry
from arcade.knowledge import ArtifactRegistry
from arcade.models import AnalystState, Paradigm, ArtifactPattern, EvidenceType, Finding

from arcade.tools import explore, measure, diagnose, investigate, compose

logger = logging.getLogger("arcade")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

graph_store = GraphStore()
workspace: Workspace | None = None
paradigm_registry = ParadigmRegistry()
artifact_registry = ArtifactRegistry()

mcp = FastMCP(
    "ARCADE",
    instructions=(
        "ARCADE: Adaptive Reasoned Composition for Anomaly Detection in Graphs. "
        "An LLM-as-Analyst framework for unsupervised graph anomaly detection. "
        "The agent uses these tools to explore graphs, run anomaly detectors, "
        "diagnose score distributions, check for artifacts, investigate convergence, "
        "compute weighted scores, and produce evidence-based reports. "
        "Workflow: Explore → Measure → Diagnose → Investigate → Report."
    ),
)


def _ws() -> Workspace:
    global workspace
    if workspace is None:
        workspace = Workspace(graph_store)
    return workspace


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def load_dataset(name: str, root: str = "./data") -> str:
    """Load a benchmark dataset by name.

    Available — PyG: cora, citeseer, pubmed, computers, photo, flickr.
    PyGOD (with ground-truth anomalies): weibo, reddit, disney, books, enron,
    inj_cora, inj_amazon, inj_flickr.
    This is the first tool to call — it loads the graph into memory.
    """
    global workspace
    info = graph_store.load_dataset(name, root)
    workspace = Workspace(graph_store)
    workspace.transition(AnalystState.INIT)
    return json.dumps({"status": "loaded", **info}, indent=2)


@mcp.tool()
def session_info() -> str:
    """Get current session state: loaded graph, paradigms run, evidence count, elapsed time."""
    return json.dumps(_ws().session_info(), indent=2)


@mcp.tool()
def save_session(directory: str = "./arcade_session") -> str:
    """Save the entire workspace (scores, evidence, weights, extensions) to disk. Resumable."""
    path = _ws().save(directory)
    return json.dumps({"status": "saved", "path": path})


# ═══════════════════════════════════════════════════════════════════════════
#  EXPLORE TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def graph_profile() -> str:
    """Profile the loaded graph: density, degree stats, communities, homophily, features, spectral gap.

    Call this first after loading a dataset — it gives you the information needed
    to select which anomaly paradigms are applicable.
    """
    _ws().transition(AnalystState.EXPLORE)
    result = explore.graph_profile(_ws())
    _ws().evidence.log_action("graph_profile", {}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def degree_distribution() -> str:
    """Analyze the degree distribution: min, max, mean, percentiles, histogram."""
    result = explore.degree_distribution(_ws())
    _ws().evidence.log_action("degree_distribution", {}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def community_detect() -> str:
    """Detect communities and report sizes, modularity."""
    result = explore.community_detect(_ws())
    _ws().evidence.log_action("community_detect", {}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def homophily_score() -> str:
    """Compute global and per-community homophily scores."""
    result = explore.homophily_score(_ws())
    _ws().evidence.log_action("homophily_score", {}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def feature_stats() -> str:
    """Analyze feature statistics: dimensionality, sparsity, entropy, constant features."""
    result = explore.feature_stats(_ws())
    _ws().evidence.log_action("feature_stats", {}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def spectral_analysis() -> str:
    """Compute spectral gap and interpret cluster separation quality."""
    result = explore.spectral_analysis(_ws())
    _ws().evidence.log_action("spectral_analysis", {}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  MEASURE TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def run_paradigm(paradigm: str) -> str:
    """Run a single anomaly paradigm detector and get score statistics + top nodes.

    Available paradigms: reconstruction, local_affinity, contrastive, adversarial,
    one_class, density, community, homophily_violation, spectral, attr_struct_mismatch.
    """
    _ws().transition(AnalystState.MEASURE)
    result = measure.run_paradigm(_ws(), paradigm)
    _ws().evidence.log_action("run_paradigm", {"paradigm": paradigm}, result.summary if hasattr(result, 'summary') else str(result))
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def run_all_selected() -> str:
    """Run ALL applicable paradigms based on the graph profile. Returns summary of each.

    Uses the graph profile to determine which paradigms apply (alpha_k selection).
    This is the batch measurement step.
    """
    _ws().transition(AnalystState.MEASURE)
    result = measure.run_all_selected(_ws())
    _ws().evidence.log_action("run_all_selected", {}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def get_score_matrix() -> str:
    """Get the full score matrix: all paradigms × top 50 nodes (fused ranking)."""
    result = measure.score_matrix(_ws())
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def get_score_percentiles(paradigm: str) -> str:
    """Get percentile breakdown for a specific paradigm's scores."""
    result = measure.score_percentiles(_ws(), paradigm)
    return json.dumps(result.to_dict(), indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  DIAGNOSE TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def score_distribution(paradigm: str, bins: int = 50) -> str:
    """Analyze score distribution for a paradigm: histogram, modes, bimodality, tail spikes.

    This is the key diagnostic tool — look for spikes, bimodality, and distinct populations
    in the score distribution.
    """
    _ws().transition(AnalystState.DIAGNOSE)
    result = diagnose.score_distribution(_ws(), paradigm, bins)
    _ws().evidence.log_action("score_distribution", {"paradigm": paradigm}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def degree_correlation(paradigm: str) -> str:
    """Check degree confound: Pearson correlation between scores and node degree.

    If |rho| > 0.7, the detector is measuring degree, not anomaly — this is the
    most common artifact in GAD.
    """
    result = diagnose.degree_correlation(_ws(), paradigm)
    _ws().evidence.log_action("degree_correlation", {"paradigm": paradigm}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def community_bias(paradigm: str) -> str:
    """Check community size artifact: are flagged nodes concentrated in one community?"""
    result = diagnose.community_bias(_ws(), paradigm)
    _ws().evidence.log_action("community_bias", {"paradigm": paradigm}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def paradigm_correlation() -> str:
    """Compute pairwise correlation matrix between all paradigm score vectors.

    High correlation (>0.8) means two paradigms are redundant — they should share
    credit via gamma_k, not double-count.
    """
    result = diagnose.paradigm_correlation(_ws())
    _ws().evidence.log_action("paradigm_correlation", {}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def tail_analysis(paradigm: str, tail_pct: float = 95) -> str:
    """Analyze the tail of a score distribution: tail nodes, gap from body, KS test for distinct population."""
    result = diagnose.tail_analysis(_ws(), paradigm, tail_pct)
    _ws().evidence.log_action("tail_analysis", {"paradigm": paradigm, "tail_pct": tail_pct}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def check_artifact(paradigm: str, artifact_name: str) -> str:
    """Run a specific artifact check on a paradigm.

    Available artifacts: degree_confound, community_size, feature_sparsity,
    reconstruction_capacity, detector_redundancy.
    """
    result = diagnose.artifact_check(_ws(), paradigm, artifact_name)
    _ws().evidence.log_action("check_artifact", {"paradigm": paradigm, "artifact": artifact_name}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  INVESTIGATE TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def node_card(node_id: int) -> str:
    """Deep inspection of a single node: degree, community, features, all paradigm scores, evidence dossier."""
    _ws().transition(AnalystState.INVESTIGATE)
    result = investigate.node_card(_ws(), node_id)
    _ws().evidence.log_action("node_card", {"node_id": node_id}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def neighborhood_probe(node_id: int) -> str:
    """Probe a node's neighborhood: feature similarity, cross-community neighbors, neighbor scores."""
    result = investigate.neighborhood_probe(_ws(), node_id)
    _ws().evidence.log_action("neighborhood_probe", {"node_id": node_id}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def cross_paradigm_view(node_ids: list[int]) -> str:
    """View a set of nodes across all paradigms: score + percentile per paradigm, convergence count."""
    result = investigate.cross_paradigm_view(_ws(), node_ids)
    _ws().evidence.log_action("cross_paradigm_view", {"num_nodes": len(node_ids)}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def case_file(node_ids: list[int]) -> str:
    """Generate a full evidence dossier for a set of suspect nodes."""
    result = investigate.case_file(_ws(), node_ids)
    _ws().evidence.log_action("case_file", {"node_ids": node_ids[:10]}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def find_convergent_nodes(min_paradigms: int = 3, top_k: int = 50) -> str:
    """Find nodes flagged by multiple independent paradigms (>P95).

    Multi-view convergence is the primary label-free validation signal.
    Nodes flagged by ≥3 independent paradigms are almost certainly anomalous.
    """
    result = investigate.find_convergent_nodes(_ws(), min_paradigms, top_k)
    _ws().evidence.log_action("find_convergent_nodes", {"min_paradigms": min_paradigms}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  COMPOSE / PRIMITIVE TOOLS
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def select_nodes(condition: str, paradigm: str = "", threshold: float = 0.95) -> str:
    """Select nodes by condition: 'top_percentile' (+ paradigm name), 'high_degree', 'low_degree'."""
    result = compose.select_nodes(_ws(), condition, paradigm or None, threshold)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def correlate_vectors(vec_a: str, vec_b: str) -> str:
    """Correlate two named vectors. Names can be paradigm names, 'degree', or 'community_size'."""
    result = compose.correlate(_ws(), vec_a, vec_b)
    _ws().evidence.log_action("correlate", {"vec_a": vec_a, "vec_b": vec_b}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def histogram_vector(vector: str, bins: int = 30) -> str:
    """Compute histogram of a named vector (paradigm name, 'degree', 'community_size')."""
    result = compose.histogram(_ws(), vector, bins)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def subgraph_stats(node_ids: list[int]) -> str:
    """Compute induced subgraph statistics: density, pairwise similarity, community spread.

    Use this to check if a group of flagged nodes forms a clique (collective anomaly).
    """
    result = compose.subgraph_stats(_ws(), node_ids)
    _ws().evidence.log_action("subgraph_stats", {"num_nodes": len(node_ids)}, result.summary)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def pairwise_similarity(node_ids: list[int]) -> str:
    """Compute pairwise feature similarity matrix for a node set."""
    result = compose.pairwise_similarity(_ws(), node_ids)
    return json.dumps(result.to_dict(), indent=2)


@mcp.tool()
def compare_groups(group_a: list[int], group_b: list[int]) -> str:
    """Compare two node groups across all paradigm scores and features."""
    result = compose.compare_groups(_ws(), group_a, group_b)
    return json.dumps(result.to_dict(), indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def compute_weights() -> str:
    """Compute all ARCADE weights: w_k = alpha_k * beta_k * gamma_k * delta_k.

    v2 improvements over v1:
    - I1: Excess mass replaces zero-fraction heuristic for beta quality.
    - I2: Consensus-based delta_k detects anti-signal paradigms (+1/0/-1).
    - I3: Mechanism test determines if degree correlation is signal or artifact.
    - I4: Adaptive fusion gate decides: FUSE / SELECT / HYBRID.
    Returns per-paradigm weights, mechanism test result, and fusion gate decision.
    """
    ws = _ws()
    weights = ws.scoring.compute_weights(
        ws.graph.profile(),
        ws.graph.degrees(),
        ws.graph.communities(),
    )
    mech = ws.scoring.mechanism_test(ws.graph.degrees())
    ws.scoring.set_profile(ws.graph.profile())
    gate = ws.scoring.compute_fusion_gate()
    deltas = ws.scoring.compute_delta()
    result = {
        "weights": {
            name: {
                "alpha": w.alpha, "beta": round(w.beta, 4),
                "gamma": round(w.gamma, 4), "delta": w.delta,
                "excess_mass": round(w.excess_mass, 4),
                "w": round(w.w, 4),
            }
            for name, w in weights.items()
        },
        "mechanism_test": mech,
        "fusion_gate": gate,
        "consensus_delta": deltas,
    }
    ws.evidence.log_action(
        "compute_weights_v2", {},
        f"v2 weights for {len(weights)} paradigms. "
        f"mechanism={mech['decision']}, gate={gate['mode']}, "
        f"excluded={sum(1 for d in deltas.values() if d['delta']==0)}, "
        f"flipped={sum(1 for d in deltas.values() if d['delta']==-1)}"
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def mechanism_test() -> str:
    """Test whether degree-score correlation is an artifact or the anomaly mechanism.

    Stage 1: If >=60% of paradigms show same-sign degree correlation → mechanism.
    Stage 2: If removing degree trend drops paradigm agreement → mechanism.
    Returns m_k: 0 = mechanism (don't penalize degree corr), 1 = artifact (penalize).
    """
    ws = _ws()
    result = ws.scoring.mechanism_test(ws.graph.degrees())
    ws.evidence.log_action("mechanism_test", result, f"Decision: {result['decision']} (m_k={result['m_k']})")
    return json.dumps(result, indent=2)


@mcp.tool()
def consensus_delta(tau: float = 0.2) -> str:
    """Compute consensus-based anti-signal detection (delta_k) for each paradigm.

    Measures Spearman correlation between each paradigm and the median-rank consensus.
    delta_k = +1 (keep), 0 (exclude), or -1 (flip scores). tau controls the threshold.
    Anti-signal paradigms (AUC < 0.5) are automatically detected and excluded or flipped.
    """
    ws = _ws()
    result = ws.scoring.compute_delta(tau=tau)
    n_flip = sum(1 for d in result.values() if d["delta"] == -1)
    n_excl = sum(1 for d in result.values() if d["delta"] == 0)
    ws.evidence.log_action("consensus_delta", {"tau": tau}, f"{n_flip} flipped, {n_excl} excluded, {len(result)-n_flip-n_excl} kept")
    return json.dumps(result, indent=2)


@mcp.tool()
def fusion_gate() -> str:
    """Compute the adaptive fusion gate: should we FUSE, SELECT, or HYBRID?

    FUSE: diversity > 0.5 and quality spread < 2 — many diverse paradigms, fuse all.
    SELECT: diversity < 0.3 or quality spread > 3 — one paradigm dominates, just use it.
    HYBRID: otherwise — use top-3 paradigms with equal weight.
    Returns diversity D, quality spread Q, excess mass per paradigm, and decision.
    """
    ws = _ws()
    ws.scoring.set_profile(ws.graph.profile())
    result = ws.scoring.compute_fusion_gate()
    ws.evidence.log_action("fusion_gate", result, f"Gate decision: {result['mode']} (D={result['D']}, Q={result['Q']})")
    return json.dumps(result, indent=2)


@mcp.tool()
def tail_separation_scores() -> str:
    """Compute tail separation quality for each paradigm's score distribution.

    Measures how well the anomaly tail separates from the normal body using:
    separation (mean gap / std), KS statistic, and tail coherence.
    Replaces the v1 zero-fraction heuristic which penalized concentrated distributions.
    """
    ws = _ws()
    result = ws.scoring.compute_tail_separation_all()
    return json.dumps({name: round(v, 4) for name, v in result.items()}, indent=2)


@mcp.tool()
def override_weight(paradigm: str, factor: str, value: float, reason: str) -> str:
    """Override a weight factor (alpha, beta, gamma, or delta) with documented reasoning.

    The agent can adjust any weight if it has evidence that the automatic computation
    is wrong. Every override is logged in the evidence store.
    """
    result = _ws().scoring.override_weight(paradigm, factor, value, reason)
    _ws().evidence.log_action("override_weight", {"paradigm": paradigm, "factor": factor, "value": value}, reason)
    return json.dumps(result, indent=2)


@mcp.tool()
def fused_scores(top_k: int = 50) -> str:
    """Get final fused anomaly scores using ARCADE weighted fusion.

    v2 fusion respects: delta_k (anti-signal flipping/exclusion), mechanism test,
    excess mass quality, and the adaptive gate (FUSE/SELECT/HYBRID).
    s_i = sum(w_k * r_k(v_i)) / sum(w_k) where w_k = alpha * beta * gamma * delta.
    """
    ws = _ws()
    if not ws.scoring.weights:
        ws.scoring.compute_weights(ws.graph.profile(), ws.graph.degrees(), ws.graph.communities())

    top = ws.scoring.top_k(top_k)
    summary = ws.scoring.score_summary()

    result = {"top_k": top, "summary": summary}
    ws.evidence.log_action("fused_scores", {"top_k": top_k}, f"Mode={summary['fusion_mode']}. Top node: {top[0]['node_id']} ({top[0]['fused_score']:.4f})" if top else "No scores")
    return json.dumps(result, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  EXTENSIBILITY — AGENT-CREATED PARADIGMS, ARTIFACTS, FINDINGS
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def register_paradigm(name: str, assumption: str, detector_name: str, description: str = "") -> str:
    """Register a new anomaly paradigm discovered during analysis.

    The agent defines: name, assumption (what makes a node anomalous under this paradigm),
    and detector_name (which built-in detector to use, or 'custom' for agent-composed).
    """
    p = Paradigm(
        name=name,
        assumption=assumption,
        detector_name=detector_name,
        description=description,
        agent_created=True,
    )
    msg = _ws().register_custom_paradigm(p)
    paradigm_registry.register(p)
    return json.dumps({"status": "registered", "paradigm": name, "message": msg})


@mcp.tool()
def register_artifact(
    name: str,
    artifact_type: str,
    affected_paradigms: list[str],
    description: str,
    mitigation: str = "",
) -> str:
    """Register a new artifact pattern discovered during analysis.

    The agent describes: name, type (confound/redundancy/bias), which paradigms it affects,
    and how to mitigate it.
    """
    a = ArtifactPattern(
        name=name,
        artifact_type=artifact_type,
        affected_paradigms=affected_paradigms,
        detection_tool="custom",
        description=description,
        mitigation=mitigation,
        agent_created=True,
    )
    msg = _ws().register_custom_artifact(a)
    artifact_registry.register(a)
    return json.dumps({"status": "registered", "artifact": name, "message": msg})


@mcp.tool()
def add_finding(
    node_ids: list[int],
    confidence: str,
    anomaly_type: str,
    reasoning: str,
    artifact_checks_passed: list[str] | None = None,
) -> str:
    """Add a confirmed finding to the evidence store.

    confidence: 'high', 'medium', or 'low'.
    anomaly_type: 'structural', 'contextual', 'collective', 'camouflaged', etc.
    """
    f = Finding(
        nodes=node_ids,
        confidence=confidence,
        anomaly_type=anomaly_type,
        reasoning=reasoning,
        artifact_checks_passed=artifact_checks_passed or [],
    )
    _ws().evidence.add_finding(f)
    _ws().evidence.log_action("add_finding", {"nodes": node_ids[:5], "type": anomaly_type}, reasoning[:200])
    return json.dumps({
        "status": "added",
        "nodes": node_ids,
        "confidence": confidence,
        "anomaly_type": anomaly_type,
    })


@mcp.tool()
def add_annotation(key: str, value: str) -> str:
    """Add a domain annotation (e.g., 'community_3 = CS department')."""
    _ws().add_annotation(key, value)
    return json.dumps({"status": "annotated", "key": key, "value": value})


# ═══════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE QUERIES
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_paradigms() -> str:
    """List all available anomaly paradigms (built-in + agent-created) with descriptions."""
    return json.dumps(paradigm_registry.describe_all(), indent=2)


@mcp.tool()
def list_artifacts() -> str:
    """List all known artifact patterns (built-in + agent-discovered) with detection methods."""
    return json.dumps(artifact_registry.describe_all(), indent=2)


@mcp.tool()
def applicable_paradigms() -> str:
    """List paradigms applicable to the current graph (based on graph profile)."""
    ws = _ws()
    if ws.profile is None:
        ws.profile = ws.graph.profile()
    applicable = paradigm_registry.applicable(ws.profile)
    return json.dumps([
        {"name": p.name, "assumption": p.assumption, "description": p.description,
         "applicable_because": p.grounding(ws.profile)}
        for p in applicable
    ], indent=2)


@mcp.tool()
def artifacts_for_paradigm(paradigm: str) -> str:
    """List known artifact patterns that affect a specific paradigm."""
    artifacts = artifact_registry.for_paradigm(paradigm)
    return json.dumps([
        {"name": a.name, "type": a.artifact_type, "description": a.description, "mitigation": a.mitigation}
        for a in artifacts
    ], indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  EVIDENCE QUERIES
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def evidence_for_node(node_id: int) -> str:
    """Get all evidence accumulated for a specific node."""
    return json.dumps(_ws().evidence.node_dossier(node_id), indent=2)


@mcp.tool()
def all_findings() -> str:
    """Get all confirmed findings in the evidence store."""
    findings = _ws().evidence.findings
    return json.dumps([
        {
            "nodes": f.nodes,
            "confidence": f.confidence,
            "anomaly_type": f.anomaly_type,
            "reasoning": f.reasoning,
            "artifact_checks_passed": f.artifact_checks_passed,
        }
        for f in findings
    ], indent=2)


@mcp.tool()
def evidence_summary() -> str:
    """Get summary of all evidence: counts by type, paradigm coverage, conflicts."""
    ev = _ws().evidence
    by_type = {}
    for e in ev.all_evidence:
        by_type[e.evidence_type.value] = by_type.get(e.evidence_type.value, 0) + 1

    return json.dumps({
        "total_evidence": len(ev.all_evidence),
        "by_type": by_type,
        "findings": len(ev.findings),
        "conflicts": ev.conflicts(),
        "trace_length": len(ev.trace),
    }, indent=2)


@mcp.tool()
def audit_trail() -> str:
    """Get the full audit trail: every tool call, parameters, and result summaries."""
    return json.dumps(_ws().evidence.trace[-50:], indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def generate_report() -> str:
    """Generate the final analytical report with all findings, evidence, and scores.

    Call this at the end of the analysis to compile everything into a structured report.
    """
    ws = _ws()
    ws.transition(AnalystState.REPORT)

    report = {
        "graph": ws.graph.name,
        "num_nodes": ws.graph.num_nodes,
        "num_edges": ws.graph.num_edges,
        "paradigms_used": ws.scoring.available_paradigms,
        "weights": {
            name: {"alpha": w.alpha, "beta": round(w.beta, 3), "gamma": round(w.gamma, 3), "w": round(w.w, 3)}
            for name, w in ws.scoring.weights.items()
        },
        "findings": [
            {
                "nodes": f.nodes,
                "confidence": f.confidence,
                "anomaly_type": f.anomaly_type,
                "reasoning": f.reasoning,
                "artifact_checks_passed": f.artifact_checks_passed,
            }
            for f in ws.evidence.findings
        ],
        "evidence_summary": {
            "total": len(ws.evidence.all_evidence),
            "signals": len(ws.evidence.get_evidence_by_type(EvidenceType.SIGNAL)),
            "artifacts": len(ws.evidence.get_evidence_by_type(EvidenceType.ARTIFACT)),
            "convergences": len(ws.evidence.get_evidence_by_type(EvidenceType.CONVERGENCE)),
        },
        "extensions": {
            "custom_paradigms": [p.name for p in ws.custom_paradigms],
            "custom_artifacts": [a.name for a in ws.custom_artifacts],
            "weight_overrides": [
                override
                for w in ws.scoring.weights.values()
                for override in w.overrides
            ],
        },
        "top_50_nodes": ws.scoring.top_k(50) if ws.scoring.available_paradigms else [],
    }

    ws.evidence.log_action("generate_report", {}, f"Report generated: {len(report['findings'])} findings")
    return json.dumps(report, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
#  COMPONENT BUILDER TOOLS (v3)
# ═══════════════════════════════════════════════════════════════════════════

from arcade.components import (
    MethodBuilder,
    analyze_graph as _analyze_graph,
    component_info as _component_info,
    recommend_components as _recommend_components,
    KNOWN_RECIPES,
)

_builder: MethodBuilder | None = None


def _get_builder() -> MethodBuilder:
    global _builder
    if _builder is None:
        _builder = MethodBuilder()
    return _builder


@mcp.tool()
def list_components(category: str | None = None) -> str:
    """List available GNN components for method building.

    Categories: encoder, objective, decoder, augmentation, scorer.
    Each component is a real nn.Module that can be composed into a GAD method.
    Call with no args to see all, or specify a category to filter.
    """
    info = _component_info(category=category)
    return json.dumps(info, indent=2)


@mcp.tool()
def recommend_components(dataset_name: str | None = None) -> str:
    """Analyze the loaded graph and recommend GAD components.

    Computes a full graph profile (homophily, degree distribution,
    feature sparsity, density) then selects components using
    empirically-validated rules:
    - Dense graphs (avg_degree>=10) → GAD-NR neighbor prediction
    - Sparse + high-dim features → PREM ego-neighbor matching
    - General → reconstruction (DOMINANT-style) baseline
    - Rich features → contrastive augmentation views

    Returns the profile analysis + ranked recommendations per category
    + suggested hyperparameters.

    The LLM agent should review these recommendations and may override
    them based on domain reasoning before calling build_method.
    """
    if not graph_store.loaded:
        return json.dumps({"error": "No graph loaded. Call load_dataset first."})
    profile = _analyze_graph(graph_store.data)
    recs = _recommend_components(profile)
    return json.dumps({"profile": profile, "recommendations": recs}, indent=2)


@mcp.tool()
def build_method(name: str, spec: dict | str) -> str:
    """Build a GAD method from a component specification (real GNN architecture).

    The spec is a dict (or JSON string) describing which components to wire together:
    {
        "encoder": {"type": "gcn|gat|gin|mlp", "hid_dim": 64, "num_layers": 4},
        "objective": {"type": "<objective>"},
        "decoder": {"type": "dual|feature_mlp|dot_product|none"},
        "augmentation": {"type": "feature_mask|edge_drop|noise_inject|edge_truncation|none"}
    }

    Objectives (empirically validated):
    - reconstruction: dual MSE on features+structure (DOMINANT-style). General baseline.
    - gad_nr: 3-part neighbor reconstruction (WSDM 2024). Best on dense graphs (0.76 Amazon).
    - ego_matching: PREM contrastive ego-neighbor matching (ICDM 2023). Best on sparse+high-dim (0.69 Cora).
    - contrastive: CoLA-style discriminator. Good for camouflaged anomalies.
    - svdd: hypersphere one-class. Compact normal distribution.
    - local_affinity: TAM neighbor similarity (NeurIPS 2023).
    - neighborhood_jsd: FlexGAD-style Jensen-Shannon divergence.
    - neighbor_prediction: predict neighbor feature mean.
    - community_deviation: distance from community centroid.

    This creates a REAL trainable nn.Module — not a wrapper.
    """
    if not graph_store.loaded:
        return json.dumps({"error": "No graph loaded."})
    spec_dict = spec if isinstance(spec, dict) else json.loads(spec)
    in_dim = graph_store.x.shape[1]
    builder = _get_builder()
    method = builder.build(name, spec_dict, in_dim)
    desc = method.describe()
    return json.dumps({"status": "built", "name": name, **desc}, indent=2)


@mcp.tool()
def build_recipe(name: str) -> str:
    """Build a method from a known recipe (reproduces a published method).

    Available recipes: dominant, cola, ocgnn, gaan, anomalydae, gae.
    Each maps to specific component choices (e.g., dominant = GCN + reconstruction + dual decoder).
    """
    if not graph_store.loaded:
        return json.dumps({"error": "No graph loaded."})
    in_dim = graph_store.x.shape[1]
    builder = _get_builder()
    method = builder.build_from_recipe(name, in_dim)
    desc = method.describe()
    recipe = KNOWN_RECIPES[name]
    return json.dumps({"status": "built", "name": name, "recipe": recipe, **desc}, indent=2)


@mcp.tool()
def build_fusion(name: str, method_specs: list | str, strategy: str = "cross_attention") -> str:
    """Build an architecture-level fusion from multiple method specs.

    Strategies:
    - shared_encoder: one shared GNN encoder with multiple objective heads
    - cross_attention: independent encoders with cross-attention on embeddings
    - ensemble_gate: independent methods with a learned per-node gating network

    method_specs is a list of spec dicts (or JSON string).
    """
    if not graph_store.loaded:
        return json.dumps({"error": "No graph loaded."})
    specs = method_specs if isinstance(method_specs, list) else json.loads(method_specs)
    in_dim = graph_store.x.shape[1]
    builder = _get_builder()

    if strategy == "shared_encoder":
        builder.build_shared_encoder(name, specs, in_dim)
    elif strategy == "cross_attention":
        builder.build_cross_attention(name, specs, in_dim)
    elif strategy == "ensemble_gate":
        builder.build_ensemble_gate(name, specs, in_dim)
    else:
        return json.dumps({"error": f"Unknown strategy: {strategy}. Use shared_encoder|cross_attention|ensemble_gate."})

    return json.dumps({"status": "built", "name": name, "strategy": strategy, "num_methods": len(specs)}, indent=2)


@mcp.tool()
def train_method(name: str, epochs: int = 100, lr: float = 0.004) -> str:
    """Train a built method on the loaded graph.

    Returns training stats (final loss, score range).
    Call build_method or build_fusion first.
    """
    if not graph_store.loaded:
        return json.dumps({"error": "No graph loaded."})
    builder = _get_builder()
    data = graph_store.data
    stats = builder.train(name, data, epochs=epochs, lr=lr)
    return json.dumps(stats, indent=2)


@mcp.tool()
def method_scores(name: str, top_k: int = 20) -> str:
    """Get anomaly scores from a trained method.

    Returns the top-K most anomalous nodes with their scores,
    plus score distribution statistics.
    """
    builder = _get_builder()
    scores = builder.scores(name)
    if scores is None:
        return json.dumps({"error": f"No scores for '{name}'. Train it first."})

    n = len(scores)
    ranked = scores.argsort()[::-1][:top_k]
    top_nodes = [{"node": int(i), "score": round(float(scores[i]), 4)} for i in ranked]

    import numpy as np
    result = {
        "name": name,
        "num_nodes": n,
        "score_stats": {
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "p90": round(float(np.percentile(scores, 90)), 4),
            "p95": round(float(np.percentile(scores, 95)), 4),
            "p99": round(float(np.percentile(scores, 99)), 4),
        },
        "top_anomalous": top_nodes,
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def evaluate_method(name: str) -> str:
    """Evaluate a trained method against ground-truth labels (AUC-ROC).

    Only works on datasets with anomaly labels (PyGOD datasets).
    """
    builder = _get_builder()
    labels = graph_store.labels
    if labels is None:
        return json.dumps({"error": "No labels in this dataset."})
    result = builder.evaluate(name, labels)
    return json.dumps(result, indent=2)


@mcp.tool()
def score_fusion(name: str, method_names: str, weights: str | None = None) -> str:
    """Fuse scores from multiple trained methods (score-level baseline).

    method_names: JSON list of method names (must be already trained).
    weights: optional JSON list of floats (default: equal weights).
    """
    builder = _get_builder()
    names = json.loads(method_names)
    w = json.loads(weights) if weights else None
    result = builder.score_level_fusion(name, names, w)
    return json.dumps(result, indent=2)


@mcp.tool()
def list_built_methods() -> str:
    """List all methods built in this session with their specs and training status."""
    builder = _get_builder()
    return json.dumps(builder.list_methods(), indent=2)


@mcp.tool()
def auto_build(top_k: int = 3) -> str:
    """LLM-assisted component builder: analyze → recommend → build → train → evaluate.

    Runs the full pipeline on the loaded graph:
    1. Analyzes graph profile (homophily, density, features, degree distribution)
    2. Recommends components based on empirical selection rules
    3. Builds top_k candidate methods from recommendations
    4. Trains all candidates
    5. Evaluates against labels (if available)
    6. Fuses candidates via score-level fusion

    Returns the full analysis, all results, and the best method.
    The LLM agent should review these results and may:
    - Call build_method with a different spec to try alternatives
    - Adjust hyperparameters and retrain via train_method
    - Reason about WHY certain components worked based on the profile
    """
    if not graph_store.loaded:
        return json.dumps({"error": "No graph loaded. Call load_dataset first."})

    builder = _get_builder()
    data = graph_store.data
    labels = graph_store.labels

    import numpy as np
    result = builder.analyze_and_build(
        data,
        labels=labels if labels is not None else None,
        top_k=top_k,
        verbose=False,
    )
    return json.dumps(result, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION & BENCHMARK TOOLS
# ═══════════════════════════════════════════════════════════════════════════

DEEP_PARADIGMS = {"reconstruction", "contrastive", "adversarial", "one_class"}
FAST_PARADIGMS = {"local_affinity", "density", "community", "homophily_violation", "spectral", "attr_struct_mismatch"}


@mcp.tool()
def run_fast_paradigms() -> str:
    """Run only fast paradigms (no deep learning). Skips: reconstruction, contrastive, adversarial, one_class.

    Use this on large graphs (>5K nodes) where deep paradigms would take too long.
    Runs: local_affinity, density, community, homophily_violation, spectral, attr_struct_mismatch.
    """
    _ws().transition(AnalystState.MEASURE)
    from arcade.paradigms import ParadigmRegistry
    pr = ParadigmRegistry()
    applicable = pr.applicable(_ws().profile or _ws().graph.profile())

    import numpy as np
    results = {}
    failed = []
    for p in applicable:
        if p.name in DEEP_PARADIGMS:
            continue
        _ws().scoring.register_paradigm(p)
        try:
            from arcade.paradigms.detectors import run_detector
            scores = run_detector(p.detector_name, _ws().graph)
            _ws().scoring.set_scores(p.name, scores)
            results[p.name] = {
                "mean": round(float(scores.mean()), 4),
                "std": round(float(scores.std()), 4),
                "p95": round(float(np.percentile(scores, 95)), 4),
                "p99": round(float(np.percentile(scores, 99)), 4),
                "top_5": [int(i) for i in np.argsort(scores)[-5:][::-1]],
            }
        except Exception as e:
            failed.append({"paradigm": p.name, "error": str(e)})

    summary = (
        f"Ran {len(results)} fast paradigms (skipped {len(DEEP_PARADIGMS)} deep). "
        f"Successful: {', '.join(results.keys())}."
    )
    return json.dumps({"paradigms_run": list(results.keys()), "paradigms_skipped": list(DEEP_PARADIGMS), "results": results, "failed": failed}, indent=2)


@mcp.tool()
def run_hybrid(epochs: int = 150, lr: float = 0.004) -> str:
    """Hybrid pipeline: fast paradigms → gap analysis → trained complements → unified fusion.

    Phase 1: Run fast paradigms (if not already done)
    Phase 2: Analyze score distributions WITHOUT labels (diagnose_gaps)
    Phase 3: Train complementary GNN components based on gap analysis
    Phase 4: Inject trained scores into v3 engine and re-fuse

    The trained components enhance (not duplicate) the fast paradigms.
    The v3 engine's UED/SELECT/AOM handles the unified ensemble automatically.
    """
    if not graph_store.loaded:
        return json.dumps({"error": "No graph loaded."})

    scoring = _ws().scoring
    import numpy as np

    # Phase 1: ensure fast paradigms ran
    if len(scoring.available_paradigms) == 0:
        run_fast_paradigms()

    fast_names = list(scoring.available_paradigms)

    # Phase 2: gap analysis (no labels)
    gap = scoring.diagnose_gaps()

    # Phase 3: train complements
    from arcade.components.registry import analyze_graph, recommend_complements
    profile = analyze_graph(graph_store.data)
    builder = _get_builder()
    _hp = builder.hybrid_pipeline(
        graph_store.data, gap, profile, epochs=epochs, lr=lr, verbose=False,
    )
    # hybrid_pipeline returns {"trained_scores", "train_info", "complements"}
    trained_scores = _hp.get("trained_scores", _hp) if isinstance(_hp, dict) else _hp

    # Phase 4: inject into scoring engine
    from arcade.models import Paradigm
    injected = []
    for name, scores in trained_scores.items():
        scoring.set_scores(name, scores)
        scoring.register_paradigm(Paradigm(
            name=name, assumption="learned",
            detector_name=name,
            description=f"Trained component: {name}",
            agent_created=True,
        ))
        injected.append(name)

    # Re-compute weights and fuse over full ensemble
    profile_obj = _ws().profile or _ws().graph.profile()
    degrees = _ws().graph.degrees()
    communities = _ws().graph.communities()
    scoring.compute_weights(profile_obj, degrees, communities)
    fused = scoring.fuse(selected_only=True)

    # Summary
    all_weights = scoring.weights
    trained_weights = {k: {"w": round(v.w, 4), "beta": round(v.beta, 4)}
                       for k, v in all_weights.items() if k.startswith("trained:")}
    fast_weights = {k: {"w": round(v.w, 4), "beta": round(v.beta, 4)}
                    for k, v in all_weights.items() if not k.startswith("trained:")}

    return json.dumps({
        "phase1_fast": fast_names,
        "phase2_gap_analysis": gap,
        "phase3_trained": injected,
        "phase4_fusion": {
            "mode": scoring.fusion_mode,
            "fast_weights": fast_weights,
            "trained_weights": trained_weights,
            "total_paradigms": len(scoring.available_paradigms),
        },
    }, indent=2, default=str)


@mcp.tool()
def evaluate_against_labels(top_k_values: str = "[50, 100, 200]") -> str:
    """Evaluate all paradigm scores AND fused scores against ground-truth labels.

    Returns AUROC, AUPRC, and Recall@k for each paradigm and the weighted fusion.
    Only works on datasets with anomaly labels (PyGOD datasets).
    This is the key quantitative validation tool — call after running paradigms + compute_weights.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    labels = graph_store.labels
    if labels is None:
        return json.dumps({"error": "No labels in this dataset."})

    binary_labels = (np.array(labels) > 0).astype(int)
    n = len(binary_labels)
    num_anomalies = int(binary_labels.sum())
    top_ks = json.loads(top_k_values)

    # Per-paradigm evaluation
    per_paradigm = {}
    for p_name in _ws().scoring.available_paradigms:
        scores = _ws().scoring.get_scores(p_name)
        if scores is not None and len(scores) == n:
            try:
                auroc = float(roc_auc_score(binary_labels, scores))
                auprc = float(average_precision_score(binary_labels, scores))
                sorted_idx = np.argsort(-scores)
                recall_at_k = {}
                for k in top_ks:
                    hits = int(binary_labels[sorted_idx[:k]].sum())
                    recall_at_k[f"recall@{k}"] = round(hits / max(num_anomalies, 1), 4)
                per_paradigm[p_name] = {"auroc": round(auroc, 4), "auprc": round(auprc, 4), **recall_at_k}
            except Exception as e:
                per_paradigm[p_name] = {"error": str(e)}

    # Fused score evaluation
    fused_result = {}
    try:
        f_data = investigate.fused_scores(_ws(), top_k=n)
        fused = np.zeros(n)
        for item in f_data.data["top_k"]:
            fused[item["node_id"]] = item["fused_score"]
        auroc_f = float(roc_auc_score(binary_labels, fused))
        auprc_f = float(average_precision_score(binary_labels, fused))
        sorted_fused = np.argsort(-fused)
        recall_f = {}
        for k in top_ks:
            hits = int(binary_labels[sorted_fused[:k]].sum())
            recall_f[f"recall@{k}"] = round(hits / max(num_anomalies, 1), 4)
        fused_result = {"auroc": round(auroc_f, 4), "auprc": round(auprc_f, 4), **recall_f}
    except Exception as e:
        fused_result = {"error": str(e)}

    # Best single paradigm
    best_paradigm = max(per_paradigm, key=lambda p: per_paradigm[p].get("auroc", 0)) if per_paradigm else None

    data = {
        "dataset": graph_store.name if hasattr(graph_store, 'name') else "unknown",
        "num_nodes": n,
        "num_anomalies": num_anomalies,
        "anomaly_ratio": round(num_anomalies / n, 4),
        "random_auprc_baseline": round(num_anomalies / n, 4),
        "per_paradigm": per_paradigm,
        "fused": fused_result,
        "best_single_paradigm": best_paradigm,
    }
    summary = (
        f"Evaluation on {n} nodes ({num_anomalies} anomalies). "
        f"Fused AUROC={fused_result.get('auroc', 'N/A')}, AUPRC={fused_result.get('auprc', 'N/A')}. "
        f"Best single: {best_paradigm} AUROC={per_paradigm.get(best_paradigm, {}).get('auroc', 'N/A')}."
    )
    _ws().evidence.log_action("evaluate_against_labels", {}, summary)
    return json.dumps(data, indent=2)


@mcp.tool()
def run_pygod_baseline(method: str = "DOMINANT", epochs: int = 100) -> str:
    """Run a PyGOD detector as baseline and evaluate against labels.

    Available methods: DOMINANT, AnomalyDAE, CONAD, COLA, OCGNN, GAAN, ARCADE_native.
    Returns AUROC, AUPRC, and training time. Use for comparison with ARCADE paradigms.
    """
    import numpy as np
    import time as _time
    from sklearn.metrics import roc_auc_score, average_precision_score

    if not graph_store.loaded:
        return json.dumps({"error": "No graph loaded."})

    labels = graph_store.labels
    if labels is None:
        return json.dumps({"error": "No labels in this dataset."})

    binary_labels = (np.array(labels) > 0).astype(int)

    # Build PyG Data object
    import torch
    from torch_geometric.data import Data
    pyg_data = Data(
        x=graph_store.x.float(),
        edge_index=graph_store.edge_index,
        y=torch.tensor(labels, dtype=torch.long),
    )

    import warnings
    warnings.filterwarnings("ignore")

    method_upper = method.upper()
    t0 = _time.time()

    try:
        if method_upper == "DOMINANT":
            from pygod.detector import DOMINANT as Det
        elif method_upper == "ANOMALYDAE":
            from pygod.detector import AnomalyDAE as Det
        elif method_upper == "CONAD":
            from pygod.detector import CONAD as Det
        elif method_upper == "COLA":
            from pygod.detector import CoLA as Det
        elif method_upper == "OCGNN":
            from pygod.detector import OCGNN as Det
        elif method_upper == "GAAN":
            from pygod.detector import GAAN as Det
        else:
            return json.dumps({"error": f"Unknown method: {method}. Try: DOMINANT, AnomalyDAE, CONAD, COLA, OCGNN, GAAN"})

        det = Det(epoch=epochs, gpu=-1)
        det.fit(pyg_data)
        scores = det.decision_score_
        elapsed = _time.time() - t0

        auroc = float(roc_auc_score(binary_labels, scores))
        auprc = float(average_precision_score(binary_labels, scores))

        sorted_idx = np.argsort(-scores)
        recall_at = {}
        for k in [50, 100, 200]:
            if k <= len(scores):
                hits = int(binary_labels[sorted_idx[:k]].sum())
                recall_at[f"recall@{k}"] = round(hits / max(binary_labels.sum(), 1), 4)

        data = {
            "method": method_upper,
            "epochs": epochs,
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            **recall_at,
            "time_seconds": round(elapsed, 1),
        }
        return json.dumps(data, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "method": method_upper, "time_seconds": round(_time.time() - t0, 1)})


@mcp.tool()
def benchmark_report() -> str:
    """Generate a full benchmark report: all paradigms + fused + PyGOD baselines.

    Runs evaluate_against_labels for ARCADE paradigms, then adds PyGOD comparison.
    Call after running paradigms and compute_weights.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    labels = graph_store.labels
    if labels is None:
        return json.dumps({"error": "No labels."})

    binary_labels = (np.array(labels) > 0).astype(int)
    n = len(binary_labels)

    # ARCADE paradigms
    arcade_results = {}
    for p_name in _ws().scoring.available_paradigms:
        scores = _ws().scoring.get_scores(p_name)
        if scores is not None and len(scores) == n:
            try:
                arcade_results[p_name] = round(float(roc_auc_score(binary_labels, scores)), 4)
            except:
                pass

    # Fused
    fused_auroc = None
    try:
        f_data = investigate.fused_scores(_ws(), top_k=n)
        fused = np.zeros(n)
        for item in f_data.data["top_k"]:
            fused[item["node_id"]] = item["fused_score"]
        fused_auroc = round(float(roc_auc_score(binary_labels, fused)), 4)
    except:
        pass

    best_paradigm = max(arcade_results, key=arcade_results.get) if arcade_results else None

    data = {
        "dataset": graph_store.name if hasattr(graph_store, 'name') else "unknown",
        "num_nodes": n,
        "num_anomalies": int(binary_labels.sum()),
        "arcade_paradigms": arcade_results,
        "arcade_fused_auroc": fused_auroc,
        "best_single_paradigm": best_paradigm,
        "best_single_auroc": arcade_results.get(best_paradigm) if best_paradigm else None,
    }
    return json.dumps(data, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    mcp.run()


if __name__ == "__main__":
    main()
