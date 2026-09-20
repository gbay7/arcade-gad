"""Paradigm registry — all 12 built-in paradigms and runtime registration."""

from __future__ import annotations

from arcade.models import Paradigm


BUILTIN_PARADIGMS: list[Paradigm] = [
    Paradigm(
        name="reconstruction",
        assumption="Anomalous nodes are hard to reconstruct from their neighborhood",
        detector_name="reconstruction_detector",
        known_artifacts=["degree_confound", "reconstruction_capacity"],
        applicability={"feature_dim": {"op": ">=", "threshold": 2}},
        description="Autoencoder-based: high reconstruction error → anomalous. Methods: DOMINANT, AnomalyDAE.",
    ),
    Paradigm(
        name="local_affinity",
        assumption="Anomalous nodes have low agreement with their neighbors",
        detector_name="local_affinity_detector",
        known_artifacts=["degree_confound"],
        applicability={"homophily": {"op": ">=", "threshold": 0.3}},
        description="Neighbor feature similarity: low local affinity → anomalous. Methods: CoLA, SL-GAD.",
    ),
    Paradigm(
        name="contrastive",
        assumption="Anomalous nodes show low cross-view agreement",
        detector_name="contrastive_detector",
        known_artifacts=["detector_redundancy"],
        applicability={"num_nodes": {"op": ">=", "threshold": 50}},
        description="Multi-scale contrastive: low agreement across views → anomalous. Methods: ANEMONE, GRADATE.",
    ),
    Paradigm(
        name="adversarial",
        assumption="Anomalous nodes look fake to a discriminator",
        detector_name="adversarial_detector",
        known_artifacts=["degree_confound"],
        applicability={"num_nodes": {"op": ">=", "threshold": 500}},
        description="GAN-based: generator–discriminator score as anomaly signal. Methods: GAAN.",
    ),
    Paradigm(
        name="subgraph_contrast",
        assumption="Anomalous nodes do not fit a readout of their own RWR-sampled local subgraph",
        detector_name="subgraph_contrast_detector",
        known_artifacts=["detector_redundancy"],
        applicability={"feature_dim": {"op": ">=", "threshold": 100}},
        description="Local subgraph-contrastive with an auxiliary generative branch (CoLA, SL-GAD family): each node is contrasted against its RWR subgraph via a bilinear discriminator, plus feature reconstruction of the target. Strong on injected/contextual anomalies. Fuse with clique_density (rank or mean) to cover the structural population that reconstruction misses.",
    ),
    Paradigm(
        name="one_class",
        assumption="Anomalous nodes fall outside the normal boundary in embedding space",
        detector_name="one_class_detector",
        known_artifacts=["feature_sparsity"],
        applicability={"feature_dim": {"op": ">=", "threshold": 2}},
        description="SVDD on graph embeddings: distance from center → anomalous. Methods: OCGNN, DeepSAD.",
    ),
    Paradigm(
        name="density",
        assumption="Anomalous nodes lie in low-density regions of the embedding space",
        detector_name="density_detector",
        known_artifacts=["feature_sparsity", "degree_confound"],
        applicability={"num_nodes": {"op": ">=", "threshold": 30}},
        description="LOF/kNN in embedding space: low local density → anomalous.",
    ),
    Paradigm(
        name="community",
        assumption="Anomalous nodes don't fit any community well",
        detector_name="community_detector",
        known_artifacts=["community_size"],
        applicability={"num_communities": {"op": ">=", "threshold": 2}},
        description="Modularity residual: low community adherence → anomalous.",
    ),
    Paradigm(
        name="homophily_violation",
        assumption="Anomalous nodes have features that mismatch their neighbors",
        detector_name="homophily_detector",
        known_artifacts=["feature_sparsity"],
        applicability={"homophily": {"op": ">=", "threshold": 0.4}},
        description="Feature–neighbor mismatch: high homophily violation → anomalous.",
    ),
    Paradigm(
        name="spectral",
        assumption="Anomalous nodes produce high-frequency signals on the graph",
        detector_name="spectral_detector",
        known_artifacts=["degree_confound"],
        applicability={"num_nodes": {"op": ">=", "threshold": 30}},
        description="Graph wavelet / spectral response: high-frequency → anomalous.",
    ),
    Paradigm(
        name="attr_struct_mismatch",
        assumption="Anomalous nodes show divergence between attribute-based and structure-based rankings",
        detector_name="attr_struct_detector",
        known_artifacts=["feature_sparsity", "reconstruction_capacity"],
        applicability={"feature_dim": {"op": ">=", "threshold": 2}},
        description="Cross-modal rank divergence: attributes and structure disagree → anomalous.",
    ),
    Paradigm(
        name="conformity",
        assumption="Anomalous actors hide behind template/minimal profiles — suspiciously conformant to the dominant feature pattern",
        detector_name="conformity_detector",
        known_artifacts=["feature_sparsity"],
        applicability={"feature_dup_rate": {"op": ">=", "threshold": 0.4}},
        description="Camouflage-by-conformity: NEGATIVE linear reconstruction residual. Only meaningful on template-dominated feature matrices (dup_rate >= 0.4), where anomalies are feature-space in-liers. Deterministic.",
    ),
    Paradigm(
        name="peripheral_mismatch",
        assumption="On tiny graphs, anomalies are peripheral, overly cliquish nodes whose features mismatch both globally and locally",
        detector_name="peripheral_mismatch_detector",
        known_artifacts=["degree_confound"],
        applicability={"num_nodes": {"op": "<=", "threshold": 500}},
        description="Tiny-graph robust composite: rank-fusion of low-degree, high-clustering, SVD residual, and feature-vs-neighborhood distance. Deterministic — replaces seed-dominated trained detectors when n <= 500.",
    ),
    Paradigm(
        name="attribute_inflation",
        assumption="Manipulation inflates count-like attributes across many dimensions simultaneously",
        detector_name="attribute_inflation_detector",
        known_artifacts=["feature_sparsity"],
        applicability={"avg_degree": {"op": "<=", "threshold": 5},
                        "num_nodes": {"op": ">=", "threshold": 500},
                        "feature_dim": {"op": "<=", "threshold": 100}},
        description="Per-dimension high-rank aggregation: nodes uniformly high across low-dimensional metadata (counts, ratings) are manipulation suspects. Deterministic; suited to sparse e-commerce graphs with few count-like features.",
    ),
    Paradigm(
        name="clique_density",
        assumption="Members of abnormally dense subgraphs (injected cliques, collusion rings) show extreme closure-weighted triangle participation",
        detector_name="clique_density_detector",
        known_artifacts=["community_size"],
        applicability={"feature_sparsity": {"op": ">=", "threshold": 0.9},
                        "feature_dim": {"op": ">=", "threshold": 100}},
        description="Structural companion for sparse high-dim graphs: triangles(v) * clustering(v). Pair with local_affinity via per-node rank-max (AOM) to cover contextual + structural anomaly populations. Deterministic.",
    ),
    Paradigm(
        name="linear_residual",
        assumption="Anomalous attributes cannot be linearly explained by other nodes' attributes plus graph coherence (Radar)",
        detector_name="linear_residual_detector",
        known_artifacts=["degree_confound"],
        applicability={"feature_sparsity": {"op": "<=", "threshold": 0.3},
                        "feature_dim": {"op": ">=", "threshold": 100},
                        "num_nodes": {"op": "<=", "threshold": 30000}},
        description="Radar-style residual analysis (Li et al., IJCAI 2017): X = XW + R with Laplacian coherence; score = ||R_i||. The linear model presumes DENSE continuous attributes (sparsity <= 0.3, dim >= 100) — on sparse binary features the residual is dominated by sparsity noise. Deterministic.",
    ),
]


class ParadigmRegistry:
    """Registry for paradigms — built-in + agent-created."""

    def __init__(self):
        self._paradigms: dict[str, Paradigm] = {}
        for p in BUILTIN_PARADIGMS:
            self._paradigms[p.name] = p

    def register(self, paradigm: Paradigm) -> None:
        self._paradigms[paradigm.name] = paradigm

    def get(self, name: str) -> Paradigm | None:
        return self._paradigms.get(name)

    def all(self) -> list[Paradigm]:
        return list(self._paradigms.values())

    def names(self) -> list[str]:
        return list(self._paradigms.keys())

    def applicable(self, profile) -> list[Paradigm]:
        return [p for p in self._paradigms.values() if p.is_applicable(profile)]

    def describe_all(self) -> list[dict]:
        return [
            {
                "name": p.name,
                "assumption": p.assumption,
                "description": p.description,
                "known_artifacts": p.known_artifacts,
                "agent_created": p.agent_created,
            }
            for p in self._paradigms.values()
        ]
