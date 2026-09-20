"""Component registry — encoders, objectives, decoders, augmentations, scorers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComponentSpec:
    """Metadata for a single component."""
    name: str
    category: str  # encoder, objective, decoder, augmentation, scorer
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    best_for: list[str] = field(default_factory=list)
    incompatible_with: list[str] = field(default_factory=list)
    reliability: str = "reliable"  # reliable, conditional, broken


COMPONENTS: dict[str, list[ComponentSpec]] = {
    "encoder": [
        ComponentSpec(
            name="gcn",
            category="encoder",
            description="GCN encoder — mean-pool neighbor aggregation. Best for homophilic graphs.",
            params={"hid_dim": 64, "num_layers": 4, "dropout": 0.0, "act": "relu"},
            best_for=["high_homophily", "smooth_signals", "general"],
        ),
        ComponentSpec(
            name="gat",
            category="encoder",
            description="GAT encoder — attention-weighted neighbor aggregation. Best for heterogeneous neighborhoods.",
            params={"hid_dim": 64, "num_layers": 4, "dropout": 0.0, "act": "relu", "heads": 4},
            best_for=["heterogeneous_neighborhoods", "varying_importance"],
        ),
        ComponentSpec(
            name="gin",
            category="encoder",
            description="GIN encoder — sum aggregation with MLP. Maximally expressive for WL-test.",
            params={"hid_dim": 64, "num_layers": 4, "dropout": 0.0, "act": "relu"},
            best_for=["structural_discrimination", "isomorphism_sensitivity"],
        ),
        ComponentSpec(
            name="mlp",
            category="encoder",
            description="MLP encoder — ignores graph structure, pure feature-based.",
            params={"hid_dim": 64, "num_layers": 4, "dropout": 0.0, "act": "relu"},
            best_for=["attribute_dominant_anomalies", "disconnected_graphs"],
        ),
    ],
    "objective": [
        ComponentSpec(
            name="reconstruction",
            category="objective",
            description="Dual reconstruction — minimize ||X - X'||^2 + alpha*||A - A'||^2. Anomaly = high recon error.",
            params={"alpha": 0.5},
            best_for=["general", "structural_anomalies", "attribute_anomalies"],
        ),
        ComponentSpec(
            name="contrastive",
            category="objective",
            description="Contrastive learning — maximize agreement between original and augmented views. Anomaly = low agreement.",
            params={},
            best_for=["camouflaged_anomalies", "contextual_anomalies"],
            incompatible_with=["decoder_dot_product", "decoder_feature_mlp", "decoder_dual"],
        ),
        ComponentSpec(
            name="svdd",
            category="objective",
            description="SVDD — minimize hypersphere volume containing normals. Anomaly = far from center.",
            params={},
            best_for=["compact_normal_distribution", "global_anomalies"],
            incompatible_with=["decoder_dot_product", "decoder_feature_mlp", "decoder_dual"],
            reliability="conditional",
        ),
        ComponentSpec(
            name="adversarial",
            category="objective",
            description="Adversarial — generator vs discriminator. Anomaly = discriminator says 'fake'.",
            params={"noise_dim": 16},
            best_for=["generative_anomalies", "complex_distributions"],
        ),
        ComponentSpec(
            name="neighborhood_jsd",
            category="objective",
            description="Neighborhood JSD — model neighbourhood as Gaussian, score = JSD between predicted and actual distribution. FlexGAD-inspired.",
            params={},
            best_for=["heterogeneous_neighborhoods", "contextual_anomalies", "low_homophily"],
            incompatible_with=["decoder_dot_product", "decoder_feature_mlp", "decoder_dual"],
        ),
        ComponentSpec(
            name="neighbor_prediction",
            category="objective",
            description="Neighbor prediction — predict mean of neighbor features from node embedding. Anomaly = high prediction error. GAD-NR-inspired.",
            params={},
            best_for=["homophilic_graphs", "contextual_anomalies", "feature_rich"],
            incompatible_with=["decoder_dot_product", "decoder_feature_mlp", "decoder_dual"],
        ),
        ComponentSpec(
            name="community_deviation",
            category="objective",
            description="Community deviation — distance from community centroid in embedding space. Anomaly = far from its community.",
            params={},
            best_for=["strong_communities", "global_anomalies", "membership_anomalies"],
            incompatible_with=["decoder_dot_product", "decoder_feature_mlp", "decoder_dual"],
        ),
        ComponentSpec(
            name="gad_nr",
            category="objective",
            description="GAD-NR 3-part loss (WSDM 2024): self-attribute MSE + degree MSE + neighbor Gaussian KL. Works on dense injected-anomaly graphs. Anti-signal on organic anomalies.",
            params={},
            best_for=["injected_anomalies", "dense_graphs"],
            incompatible_with=["decoder_dot_product", "decoder_feature_mlp", "decoder_dual"],
            reliability="conditional",
        ),
        ComponentSpec(
            name="local_affinity",
            category="objective",
            description="TAM local affinity (NeurIPS 2023): semi-hard triplet margin loss. Redundant with fast local_affinity_detector — use only when fast paradigm is unavailable.",
            params={},
            best_for=["homophilic_graphs", "one_class"],
            incompatible_with=["decoder_dot_product", "decoder_feature_mlp", "decoder_dual"],
            reliability="broken",
        ),
        ComponentSpec(
            name="residual",
            category="objective",
            description="Residual analysis (Radar-inspired, IJCAI 2017): reconstruct attributes from a graph-smoothed embedding; anomaly = the residual the network cannot explain. Strong on high-homophily graphs with injected feature perturbations (e.g. Weibo h=1.0, Amazon).",
            params={"lam": 0.5},
            best_for=["high_homophily", "injected_anomalies", "feature_perturbation"],
            incompatible_with=["decoder_dot_product", "decoder_feature_mlp", "decoder_dual"],
            reliability="conditional",
        ),
        ComponentSpec(
            name="ego_matching",
            category="objective",
            description="PREM ego-neighbor matching (ICDM 2023): InfoNCE ego vs neighbor mean. Works on sparse high-dim injected anomalies (Cora 0.69). Anti-signal on organic/dense.",
            params={},
            best_for=["injected_anomalies", "sparse_high_dim"],
            incompatible_with=["decoder_dot_product", "decoder_feature_mlp", "decoder_dual"],
            reliability="conditional",
        ),
    ],
    "decoder": [
        ComponentSpec(
            name="dot_product",
            category="decoder",
            description="DotProduct decoder — A' = Z @ Z.T. Reconstructs adjacency.",
            params={"sigmoid_s": False},
            best_for=["structure_reconstruction"],
        ),
        ComponentSpec(
            name="feature_mlp",
            category="decoder",
            description="Feature MLP decoder — X' = Backbone(Z). Reconstructs node features.",
            params={},
            best_for=["feature_reconstruction"],
        ),
        ComponentSpec(
            name="dual",
            category="decoder",
            description="Dual decoder — separate decoders for structure (Z@Z.T) and features (MLP).",
            params={"alpha": 0.5, "sigmoid_s": False},
            best_for=["dual_modality", "general"],
        ),
        ComponentSpec(
            name="none",
            category="decoder",
            description="No decoder — score derived directly from embeddings (contrastive, SVDD).",
            params={},
            best_for=["contrastive", "svdd", "embedding_based"],
        ),
    ],
    "augmentation": [
        ComponentSpec(
            name="feature_mask",
            category="augmentation",
            description="Feature masking — randomly zero out feature dimensions to create contrastive views.",
            params={"mask_rate": 0.3},
            best_for=["contrastive", "high_dimensional_features"],
        ),
        ComponentSpec(
            name="edge_drop",
            category="augmentation",
            description="Edge dropping — remove edges with probability p. Tests structural robustness.",
            params={"drop_rate": 0.2},
            best_for=["structural_robustness", "sparse_graphs"],
        ),
        ComponentSpec(
            name="noise_inject",
            category="augmentation",
            description="Noise injection — X~ = X + N(0, sigma^2). Adversarial perturbation.",
            params={"noise_std": 0.1},
            best_for=["adversarial", "dense_features"],
        ),
        ComponentSpec(
            name="community_smooth",
            category="augmentation",
            description="Community smoothing — replace features with 1-hop neighborhood mean. FlexGAD-inspired. Creates a smoothed view for contrastive learning.",
            params={},
            best_for=["contrastive", "homophilic_graphs", "noisy_features"],
        ),
        ComponentSpec(
            name="edge_truncation",
            category="augmentation",
            description="TAM NSGT-inspired edge truncation — remove non-homophily edges based on feature cosine similarity. Reduces noise from heterophilic connections.",
            params={"threshold": 0.3},
            best_for=["low_homophily", "organic_anomalies", "noisy_structure"],
        ),
        ComponentSpec(
            name="none",
            category="augmentation",
            description="No augmentation — use original data only.",
            params={},
            best_for=["reconstruction", "svdd"],
        ),
    ],
    "scorer": [
        ComponentSpec(
            name="mse",
            category="scorer",
            description="MSE scorer — per-node mean squared error between input and reconstruction.",
            params={},
            best_for=["reconstruction"],
        ),
        ComponentSpec(
            name="bce",
            category="scorer",
            description="BCE scorer — binary cross-entropy on adjacency reconstruction.",
            params={},
            best_for=["structure_reconstruction", "adversarial"],
        ),
        ComponentSpec(
            name="distance",
            category="scorer",
            description="Distance scorer — ||z_i - c||^2. Distance from SVDD center.",
            params={},
            best_for=["svdd"],
        ),
        ComponentSpec(
            name="margin",
            category="scorer",
            description="Margin scorer — f_neg(i) - f_pos(i). Contrastive margin.",
            params={},
            best_for=["contrastive"],
        ),
    ],
}


def component_info(category: str | None = None, name: str | None = None) -> list[dict]:
    """Return component metadata, optionally filtered."""
    results = []
    categories = [category] if category else list(COMPONENTS.keys())
    for cat in categories:
        for spec in COMPONENTS.get(cat, []):
            if name and spec.name != name:
                continue
            results.append({
                "name": spec.name,
                "category": spec.category,
                "description": spec.description,
                "params": spec.params,
                "best_for": spec.best_for,
                "incompatible_with": spec.incompatible_with,
            })
    return results


def analyze_graph(data) -> dict:
    """Compute graph profile properties for component selection.

    Accepts a PyG Data object, returns a profile dict with all
    properties needed by recommend_components().
    """
    import torch
    from torch_geometric.utils import homophily as _homophily

    x = data.x
    edge_index = data.edge_index
    n = data.num_nodes
    m = edge_index.size(1)

    degrees = torch.zeros(n, dtype=torch.long)
    degrees.scatter_add_(0, edge_index[1], torch.ones(m, dtype=torch.long))
    avg_deg = degrees.float().mean().item()
    max_deg = int(degrees.max().item())
    degree_std = degrees.float().std().item()

    density = m / max(n * (n - 1), 1)

    # Feature statistics
    feature_dim = int(x.size(1))
    feature_sparsity = float((x == 0).float().mean().item())
    feature_std_mean = float(x.std(dim=0).mean().item())
    # Row-duplication rate — template-dominated features signal that
    # conformity (camouflage) anomalies are plausible (see conformity paradigm)
    import numpy as _np
    _, _counts = _np.unique(x.cpu().numpy().round(6), axis=0, return_counts=True)
    feature_dup_rate = float(_counts[_counts > 1].sum() / max(n, 1))

    # Degree distribution shape
    deg_sorted = degrees.float().sort(descending=True).values
    is_power_law = (deg_sorted[0] > 10 * deg_sorted[n // 2]) if n > 10 else False

    # Homophily (edge-level, based on 1-hop label propagation communities)
    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(5):
        new_labels = labels.clone()
        new_labels.scatter_reduce_(0, edge_index[1], labels[edge_index[0]],
                                   reduce="amin")
        if (new_labels == labels).all():
            break
        labels = new_labels
    _, labels = labels.unique(return_inverse=True)
    h = _homophily(edge_index, labels, method="edge")

    return {
        "num_nodes": n,
        "num_edges": m,
        "density": round(density, 6),
        "avg_degree": round(avg_deg, 2),
        "max_degree": max_deg,
        "degree_std": round(degree_std, 2),
        "feature_dim": feature_dim,
        "feature_sparsity": round(feature_sparsity, 4),
        "feature_std_mean": round(feature_std_mean, 4),
        "feature_dup_rate": round(feature_dup_rate, 4),
        "is_power_law": bool(is_power_law),
        "homophily": round(float(h), 4),
    }


def recommend_components(profile: dict) -> dict:
    """Recommend components based on graph profile.

    Selection rules are empirically grounded:
    - GAD-NR: strong on dense graphs with structural anomalies (Amazon 0.694)
    - PREM ego-matching: strong on high-dim sparse features (Cora 0.659)
    - Reconstruction: safe general baseline (DOMINANT)
    - Contrastive: good for camouflaged contextual anomalies
    - Fast paradigm ensemble: best for organic hard anomalies (Reddit 0.599)

    Returns {category: [(name, reason), ...]}, first item = top recommendation.
    """
    recs: dict[str, list[tuple[str, str]]] = {}
    homophily = profile.get("homophily", 0.5)
    feature_dim = profile.get("feature_dim", 0)
    num_nodes = profile.get("num_nodes", 0)
    density = profile.get("density", 0.0)
    avg_degree = profile.get("avg_degree", 0.0)
    feature_sparsity = profile.get("feature_sparsity", 0.0)
    is_power_law = profile.get("is_power_law", False)

    # ── Encoder ──
    enc_recs = []
    if homophily >= 0.7:
        enc_recs.append(("gcn", f"homophily={homophily:.2f} — neighbors are informative"))
    elif homophily < 0.4:
        enc_recs.append(("gat", f"homophily={homophily:.2f} — attention filters noisy neighbors"))
        enc_recs.append(("mlp", f"homophily={homophily:.2f} — structure may mislead"))
    else:
        enc_recs.append(("gcn", "moderate homophily — GCN is robust"))
        enc_recs.append(("gin", "moderate homophily — GIN is more expressive"))
    recs["encoder"] = enc_recs

    # ── Objective ── (reliability-gated, empirically-validated)
    # Conditional objectives only allowed under specific conditions:
    #   gad_nr: injected anomalies on dense graphs (feature_sparsity < 0.3)
    #   ego_matching: injected anomalies on sparse high-dim (sparsity >= 0.5, dim >= 100)
    #   svdd: always allowed (fixed with trimmed center)
    likely_injected = feature_sparsity >= 0.4 and feature_dim >= 50

    obj_recs = []
    if avg_degree >= 10 and likely_injected:
        obj_recs.append(("gad_nr",
            f"dense + likely injected anomalies — GAD-NR neighbor prediction"))
    if feature_dim >= 100 and feature_sparsity >= 0.5 and likely_injected:
        obj_recs.append(("ego_matching",
            f"sparse high-dim features — ego-neighbor matching"))
    obj_recs.append(("reconstruction",
        "dual reconstruction — reliable general-purpose baseline"))
    if feature_dim >= 20:
        obj_recs.append(("contrastive",
            f"feature_dim={feature_dim} — contrastive for camouflaged anomalies"))
    obj_recs.append(("svdd",
        "one-class hypersphere — orthogonal to reconstruction"))
    recs["objective"] = obj_recs

    # ── Decoder ──
    dec_recs = []
    primary_obj = obj_recs[0][0] if obj_recs else "reconstruction"
    if primary_obj in ("gad_nr", "ego_matching", "contrastive", "svdd",
                       "local_affinity"):
        dec_recs.append(("none", f"{primary_obj} does not need a decoder"))
    elif feature_dim >= 5 and avg_degree >= 2:
        dec_recs.append(("dual", "both features and structure available"))
    elif feature_dim < 5:
        dec_recs.append(("dot_product", "low feature dim — reconstruct structure"))
    else:
        dec_recs.append(("feature_mlp", "rich features — reconstruct features"))
    recs["decoder"] = dec_recs

    # ── Augmentation ──
    aug_recs = [("none", "no augmentation — safe default")]
    if primary_obj == "contrastive" and feature_dim >= 20:
        aug_recs.insert(0, ("feature_mask",
            f"feature_dim={feature_dim} — masking creates meaningful contrastive views"))
    if homophily < 0.4:
        aug_recs.insert(0, ("edge_truncation",
            f"homophily={homophily:.2f} — truncate heterophilic edges"))
    recs["augmentation"] = aug_recs

    # ── Hyperparameters (data-driven) ──
    hid_dim = 64 if feature_dim <= 200 else 128
    num_layers = 2 if num_nodes < 3000 else 4
    lr = 0.004 if num_nodes <= 15000 else 0.001
    epochs = 200 if num_nodes <= 15000 else 100
    batch_size = 0 if num_nodes <= 20000 else 4096
    num_neigh = -1 if num_nodes <= 20000 else 10
    recs["_hyperparams"] = {
        "hid_dim": hid_dim, "num_layers": num_layers,
        "lr": lr, "epochs": epochs,
        "batch_size": batch_size, "num_neigh": num_neigh,
    }

    return recs


def recommend_complements(gap_analysis: dict, profile: dict) -> list[dict]:
    """Recommend trained components that complement fast paradigms.

    Uses unsupervised gap analysis (no labels) to select objectives
    that fill holes in the fast paradigm coverage.
    """
    signal_count = gap_analysis.get("signal_count", 0)
    bimod = gap_analysis.get("bimodality", {})
    avg_bimod = sum(bimod.values()) / max(len(bimod), 1)
    feature_dim = profile.get("feature_dim", 0)
    num_nodes = profile.get("num_nodes", 0)
    avg_degree = profile.get("avg_degree", 0)

    hid_dim = 64 if feature_dim <= 200 else 128
    num_layers = 2 if num_nodes < 3000 else 4
    large = num_nodes > 20000

    complements = []

    # Always: reconstruction (balanced) as reliable anchor
    complements.append({
        "name": "trained:recon-gcn",
        "spec": {
            "encoder": {"type": "gcn", "hid_dim": hid_dim, "num_layers": num_layers},
            "objective": {"type": "reconstruction", "alpha": 0.5},
            "decoder": {"type": "feature_mlp" if large else "dual"},
            "augmentation": {"type": "none"},
        },
        "reason": "reconstruction — reliable anchor, orthogonal to fast paradigms",
    })

    # Structure-heavy reconstruction if structure dominates (low feature_dim)
    if feature_dim <= 64 and not large:
        complements.append({
            "name": "trained:recon-struct",
            "spec": {
                "encoder": {"type": "gcn", "hid_dim": hid_dim, "num_layers": num_layers},
                "objective": {"type": "reconstruction", "alpha": 0.2},
                "decoder": {"type": "dual"},
                "augmentation": {"type": "none"},
            },
            "reason": f"feature_dim={feature_dim} — structure-weighted recon captures topology anomalies",
        })

    # Contrastive: always when features are rich enough
    if feature_dim >= 20:
        complements.append({
            "name": "trained:contrastive-gcn",
            "spec": {
                "encoder": {"type": "gcn", "hid_dim": hid_dim, "num_layers": num_layers},
                "objective": {"type": "contrastive"},
                "decoder": {"type": "none"},
                "augmentation": {"type": "feature_mask", "mask_rate": 0.3},
            },
            "reason": f"feature_dim={feature_dim} — contrastive catches camouflaged anomalies",
        })

    # SVDD if few paradigms are working
    if signal_count < 4:
        complements.append({
            "name": "trained:svdd-gcn",
            "spec": {
                "encoder": {"type": "gcn", "hid_dim": hid_dim, "num_layers": num_layers},
                "objective": {"type": "svdd"},
                "decoder": {"type": "none"},
                "augmentation": {"type": "none"},
            },
            "reason": f"only {signal_count} strong paradigms — SVDD adds one-class view",
        })

    return complements


KNOWN_RECIPES: dict[str, dict[str, str]] = {
    "dominant": {"encoder": "gcn", "objective": "reconstruction", "decoder": "dual", "augmentation": "none", "scorer": "mse"},
    "cola": {"encoder": "gcn", "objective": "contrastive", "decoder": "none", "augmentation": "feature_mask", "scorer": "margin"},
    "ocgnn": {"encoder": "gcn", "objective": "svdd", "decoder": "none", "augmentation": "none", "scorer": "distance"},
    "gaan": {"encoder": "mlp", "objective": "adversarial", "decoder": "dot_product", "augmentation": "noise_inject", "scorer": "bce"},
    "anomalydae": {"encoder": "gat", "objective": "reconstruction", "decoder": "dual", "augmentation": "none", "scorer": "mse"},
    "gae": {"encoder": "gcn", "objective": "reconstruction", "decoder": "feature_mlp", "augmentation": "none", "scorer": "mse"},
    "gad_nr": {"encoder": "gcn", "objective": "gad_nr", "decoder": "none", "augmentation": "none", "scorer": "mse"},
    "tam": {"encoder": "gcn", "objective": "local_affinity", "decoder": "none", "augmentation": "edge_truncation", "scorer": "margin"},
    "prem": {"encoder": "gcn", "objective": "ego_matching", "decoder": "none", "augmentation": "none", "scorer": "margin"},
}
