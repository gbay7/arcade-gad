"""Detector implementations for all 12 built-in anomaly paradigms.

Each detector takes a GraphStore and returns a score vector (np.ndarray)
with one score per node, higher = more anomalous.

v2: vectorized — no per-node Python loops. Uses sparse matrix ops for
neighbor aggregation (D_inv @ A @ X) instead of `for i in range(n)`.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from scipy import sparse
from typing import Any

from arcade.data.loader import GraphStore


def run_detector(name: str, graph: GraphStore, **kwargs: Any) -> np.ndarray:
    """Dispatch to the named detector."""
    detectors = {
        "reconstruction_detector": reconstruction_detector,
        "local_affinity_detector": local_affinity_detector,
        "contrastive_detector": contrastive_detector,
        "adversarial_detector": adversarial_detector,
        "one_class_detector": one_class_detector,
        "density_detector": density_detector,
        "community_detector": community_detector,
        "homophily_detector": homophily_detector,
        "spectral_detector": spectral_detector,
        "attr_struct_detector": attr_struct_detector,
        "conformity_detector": conformity_detector,
        "linear_residual_detector": linear_residual_detector,
        "peripheral_mismatch_detector": peripheral_mismatch_detector,
        "attribute_inflation_detector": attribute_inflation_detector,
        "clique_density_detector": clique_density_detector,
        "subgraph_contrast_detector": subgraph_contrast_detector,
    }
    fn = detectors.get(name)
    if fn is None:
        raise ValueError(f"Unknown detector: {name}. Available: {list(detectors.keys())}")
    return fn(graph, **kwargs)


# ---------------------------------------------------------------------------
# Sparse adjacency helpers (cached on first call)
# ---------------------------------------------------------------------------

def _sparse_adj(graph: GraphStore) -> tuple[sparse.csr_matrix, sparse.dia_matrix]:
    """Return (A, D_inv) — sparse adjacency and inverse-degree diagonal."""
    ei = graph.edge_index.cpu().numpy()
    n = graph.num_nodes
    data = np.ones(ei.shape[1], dtype=np.float32)
    A = sparse.csr_matrix((data, (ei[0], ei[1])), shape=(n, n))
    deg = np.array(A.sum(axis=1)).ravel()
    deg_inv = np.where(deg > 0, 1.0 / deg, 0.0).astype(np.float32)
    D_inv = sparse.diags(deg_inv)
    return A, D_inv


# ---------------------------------------------------------------------------
# 1. Reconstruction: autoencoder reconstruction error
# ---------------------------------------------------------------------------

def reconstruction_detector(graph: GraphStore, hidden_dim: int = 64, epochs: int = 100, **kw) -> np.ndarray:
    x = graph.x.float()
    edge_index = graph.edge_index
    in_dim = x.shape[1]

    from torch_geometric.nn import GCNConv

    # The decoder GCNConv(hidden -> in_dim) projects to full feature width and
    # THEN aggregates, so gather-scatter materialises an (edges x in_dim)
    # message tensor — tens of GB on wide-feature, edge-heavy graphs
    # (BlogCatalog 23GB, Flickr 46GB: the OOM killer takes the process). When
    # that tensor would be large, pass the adjacency as a SparseTensor so PyG
    # uses fused sparse-dense matmul: the same normalised aggregation with no
    # per-edge materialisation. Small graphs keep the original path unchanged.
    est_message_bytes = 2 * edge_index.shape[1] * in_dim * 4
    if est_message_bytes > 4e9:
        from torch_sparse import SparseTensor
        n = graph.num_nodes
        adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                           sparse_sizes=(n, n)).to(x.device)
    else:
        adj = edge_index

    class AE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = GCNConv(in_dim, hidden_dim)
            self.dec = GCNConv(hidden_dim, in_dim)

        def forward(self, x, ei):
            z = torch.relu(self.enc(x, ei))
            return self.dec(z, ei)

    device = x.device
    model = AE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        x_hat = model(x, adj)
        loss = torch.nn.functional.mse_loss(x_hat, x)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        x_hat = model(x, adj)
        errors = torch.mean((x - x_hat) ** 2, dim=1).cpu().numpy()

    return _normalize(errors)


# ---------------------------------------------------------------------------
# 2. Local affinity: neighbor feature similarity  [VECTORIZED]
# ---------------------------------------------------------------------------

def local_affinity_detector(graph: GraphStore, **kw) -> np.ndarray:
    x = graph.node_features()
    n = graph.num_nodes
    A, D_inv = _sparse_adj(graph)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / np.maximum(norms, 1e-8)

    nbr_mean_norm = (D_inv @ A @ x_norm).astype(np.float32)
    similarity = np.sum(x_norm * nbr_mean_norm, axis=1)
    scores = 1.0 - similarity

    deg = np.array(A.sum(axis=1)).ravel()
    scores[deg == 0] = 1.0

    return _normalize(scores)


# ---------------------------------------------------------------------------
# 3. Contrastive: cross-view disagreement via two augmented views
# ---------------------------------------------------------------------------

def contrastive_detector(graph: GraphStore, hidden_dim: int = 64, epochs: int = 100, **kw) -> np.ndarray:
    x = graph.x.float()
    edge_index = graph.edge_index

    from torch_geometric.nn import GCNConv

    class Encoder(torch.nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        def forward(self, x, ei):
            z = torch.relu(self.conv1(x, ei))
            return self.conv2(z, ei)

    device = x.device
    enc = Encoder(x.shape[1]).to(device)
    opt = torch.optim.Adam(enc.parameters(), lr=0.005)

    def augment_features(x, drop_rate=0.2):
        mask = torch.bernoulli(torch.full_like(x, 1 - drop_rate))
        return x * mask

    enc.train()
    for _ in range(epochs):
        opt.zero_grad()
        z1 = enc(augment_features(x), edge_index)
        z2 = enc(augment_features(x), edge_index)
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        sim = (z1 * z2).sum(dim=1)
        loss = -sim.mean()
        loss.backward()
        opt.step()

    enc.eval()
    with torch.no_grad():
        z1 = torch.nn.functional.normalize(enc(x, edge_index), dim=1)
        z2 = torch.nn.functional.normalize(enc(augment_features(x, 0.1), edge_index), dim=1)
        agreement = (z1 * z2).sum(dim=1).cpu().numpy()

    return _normalize(1.0 - agreement)


# ---------------------------------------------------------------------------
# 4. Adversarial: discriminator score (simplified)
# ---------------------------------------------------------------------------

def adversarial_detector(graph: GraphStore, hidden_dim: int = 64, epochs: int = 100, **kw) -> np.ndarray:
    x = graph.x.float()
    edge_index = graph.edge_index

    from torch_geometric.nn import GCNConv

    class Disc(torch.nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.conv = GCNConv(in_dim, hidden_dim)
            self.lin = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x, ei):
            z = torch.relu(self.conv(x, ei))
            return torch.sigmoid(self.lin(z)).squeeze(-1)

    device = x.device
    disc = Disc(x.shape[1]).to(device)
    opt = torch.optim.Adam(disc.parameters(), lr=0.005)

    disc.train()
    for _ in range(epochs):
        opt.zero_grad()
        real_scores = disc(x, edge_index)
        noise = torch.randn_like(x) * 0.5
        fake_scores = disc(x + noise, edge_index)
        loss = -torch.log(real_scores + 1e-8).mean() - torch.log(1 - fake_scores + 1e-8).mean()
        loss.backward()
        opt.step()

    disc.eval()
    with torch.no_grad():
        scores = 1.0 - disc(x, edge_index).cpu().numpy()

    return _normalize(scores)


# ---------------------------------------------------------------------------
# 5. One-class: SVDD distance in embedding space
# ---------------------------------------------------------------------------

def one_class_detector(graph: GraphStore, hidden_dim: int = 64, epochs: int = 100, **kw) -> np.ndarray:
    x = graph.x.float()
    edge_index = graph.edge_index

    from torch_geometric.nn import GCNConv

    class SVDD(torch.nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        def forward(self, x, ei):
            z = torch.relu(self.conv1(x, ei))
            return self.conv2(z, ei)

    device = x.device
    model = SVDD(x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)

    model.train()
    with torch.no_grad():
        center = model(x, edge_index).mean(dim=0)

    for _ in range(epochs):
        opt.zero_grad()
        z = model(x, edge_index)
        loss = ((z - center) ** 2).sum(dim=1).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            center = 0.9 * center + 0.1 * model(x, edge_index).mean(dim=0)

    model.eval()
    with torch.no_grad():
        z = model(x, edge_index)
        dists = ((z - center) ** 2).sum(dim=1).cpu().numpy()

    return _normalize(dists)


# ---------------------------------------------------------------------------
# 6. Density: Local Outlier Factor in embedding space
# ---------------------------------------------------------------------------

def density_detector(graph: GraphStore, n_neighbors: int = 20, **kw) -> np.ndarray:
    x = graph.node_features()
    from sklearn.neighbors import LocalOutlierFactor
    k = min(n_neighbors, len(x) - 1)
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False, contamination="auto")
    lof.fit(x)
    scores = -lof.negative_outlier_factor_
    return _normalize(scores)


# ---------------------------------------------------------------------------
# 7. Community: modularity residual  [VECTORIZED]
# ---------------------------------------------------------------------------

def community_detector(graph: GraphStore, **kw) -> np.ndarray:
    comms = graph.communities()
    x = graph.node_features()
    n = graph.num_nodes

    unique_comms = np.unique(comms)
    centroids = np.zeros_like(x)
    stds = np.zeros(n, dtype=np.float32)

    for c in unique_comms:
        mask = comms == c
        if mask.sum() < 2:
            centroids[mask] = x[mask]
            stds[mask] = 1.0
            continue
        center = x[mask].mean(axis=0)
        comm_dists = np.linalg.norm(x[mask] - center, axis=1)
        centroids[mask] = center
        stds[mask] = max(float(comm_dists.mean()), 1e-8)

    scores = np.linalg.norm(x - centroids, axis=1) / stds
    return _normalize(scores)


# ---------------------------------------------------------------------------
# 8. Homophily violation: feature–neighbor mismatch  [VECTORIZED]
# ---------------------------------------------------------------------------

def homophily_detector(graph: GraphStore, **kw) -> np.ndarray:
    x = graph.node_features()
    n = graph.num_nodes
    A, D_inv = _sparse_adj(graph)

    nbr_mean = (D_inv @ A @ x).astype(np.float32)
    scores = np.linalg.norm(x - nbr_mean, axis=1)

    deg = np.array(A.sum(axis=1)).ravel()
    scores[deg == 0] = float(np.median(scores[deg > 0])) if (deg > 0).any() else 0.5

    return _normalize(scores)


# ---------------------------------------------------------------------------
# 9. Spectral: graph wavelet response
# ---------------------------------------------------------------------------

def spectral_detector(graph: GraphStore, scales: int = 4, **kw) -> np.ndarray:
    n = graph.num_nodes
    A, D_inv = _sparse_adj(graph)
    deg = np.array(A.sum(axis=1)).ravel()

    # For large graphs, use Chebyshev polynomial approximation instead of eigsh
    if n > 3000:
        return _spectral_chebyshev(A, D_inv, deg, graph, n)

    from scipy.sparse.linalg import eigsh
    L = sparse.diags(deg) - A
    k = min(50, n - 2)

    try:
        eigvals, eigvecs = eigsh(L.astype(np.float64), k=k, which="SM", maxiter=n * 5)
    except Exception:
        return _spectral_chebyshev(A, D_inv, deg, graph, n)

    x = graph.node_features()
    signal = np.linalg.norm(x, axis=1)

    spectral_coeffs = eigvecs.T @ signal
    high_freq_energy = np.zeros(n)
    half = max(1, k // 2)
    for j in range(half, k):
        high_freq_energy += (spectral_coeffs[j] ** 2) * (eigvecs[:, j] ** 2)

    return _normalize(high_freq_energy)


def _spectral_chebyshev(A, D_inv, deg, graph, n):
    """Fast spectral anomaly detection via graph high-pass filter.

    Instead of eigendecomposition, applies a high-pass filter using
    the normalized Laplacian: h(L) = I - D_inv @ A (one-hop smoothing).
    Residual energy after smoothing = high-frequency content.
    """
    x = graph.node_features().astype(np.float64)
    signal = np.linalg.norm(x, axis=1)

    # Normalized Laplacian high-pass: residual = signal - smoothed(signal)
    smoothed = np.array(D_inv @ A @ signal).ravel()
    residual_1 = np.abs(signal - smoothed)

    # 2-hop smoothing for deeper frequency separation
    smoothed_2 = np.array(D_inv @ A @ smoothed).ravel()
    residual_2 = np.abs(signal - smoothed_2)

    scores = 0.5 * residual_1 + 0.5 * residual_2
    return _normalize(scores)


# ---------------------------------------------------------------------------
# 10. Attribute-structure mismatch: cross-modal rank divergence  [VECTORIZED]
# ---------------------------------------------------------------------------

def attr_struct_detector(graph: GraphStore, **kw) -> np.ndarray:
    x = graph.node_features()
    n = graph.num_nodes

    from sklearn.neighbors import NearestNeighbors
    k = min(10, n - 1)
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(x)
    attr_dists, _ = nn.kneighbors(x)
    attr_rank = np.argsort(np.argsort(attr_dists.mean(axis=1)))

    A, D_inv = _sparse_adj(graph)
    nbr_mean = (D_inv @ A @ x).astype(np.float32)
    struct_scores = np.linalg.norm(x - nbr_mean, axis=1)

    deg = np.array(A.sum(axis=1)).ravel()
    struct_scores[deg == 0] = float(np.mean(attr_dists[deg == 0].mean(axis=1))) if (deg == 0).any() else 0.0
    struct_rank = np.argsort(np.argsort(struct_scores))

    rank_diff = np.abs(attr_rank.astype(float) - struct_rank.astype(float))
    return _normalize(rank_diff)


# ---------------------------------------------------------------------------
# 11. Conformity: anomalies hide behind template/minimal profiles
# ---------------------------------------------------------------------------

def conformity_detector(graph: GraphStore, n_components: int = 5, **kw) -> np.ndarray:
    """Camouflage-by-conformity: score = NEGATIVE linear reconstruction residual.

    On template-dominated feature matrices (high feature_dup_rate), anomalous
    actors often carry scrubbed/minimal profiles that sit suspiciously close
    to the dominant linear structure — they are feature-space IN-liers, and
    every standard outlier direction inverts. Deterministic (fixed SVD),
    zero seed variance. Gated by the feature_dup_rate applicability trigger.
    """
    from sklearn.decomposition import TruncatedSVD
    x = graph.node_features().astype(np.float64)
    k = min(n_components, x.shape[1] - 1)
    if k < 1:
        return np.zeros(x.shape[0])
    svd = TruncatedSVD(n_components=k, random_state=0)
    x_hat = svd.inverse_transform(svd.fit_transform(x))
    resid = np.linalg.norm(x - x_hat, axis=1)
    return _normalize(-resid)


# ---------------------------------------------------------------------------
# 12. Linear residual: Radar-style residual analysis (Li et al., IJCAI 2017)
# ---------------------------------------------------------------------------

def linear_residual_detector(graph: GraphStore, **kw) -> np.ndarray:
    """Anomalous attributes cannot be linearly explained by the other nodes.

    Wraps Radar's residual decomposition (X = XW + R with graph-Laplacian
    coherence); score = ||R_i||. Deterministic given the graph. Matches the
    high-homophily / rich-continuous-feature regime (e.g. Weibo).
    """
    from pygod.detector import Radar
    det = Radar(gpu=0 if torch.cuda.is_available() else -1)
    det.fit(graph.data)
    s = det.decision_score_
    s = s.cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
    return _normalize(s)


# ---------------------------------------------------------------------------
# 13. Peripheral mismatch: tiny-graph robust composite (deterministic)
# ---------------------------------------------------------------------------

def peripheral_mismatch_detector(graph: GraphStore, **kw) -> np.ndarray:
    """Rank-fusion of four unsupervised outlier directions for TINY graphs.

    On graphs of a few hundred nodes, trained detectors are dominated by seed
    variance. This composite is deterministic: anomalies are peripheral
    (low degree), overly cliquish (high clustering), and mismatched both
    globally (SVD residual) and locally (feature-vs-neighborhood distance).
    Score = mean of the four percentile ranks.
    """
    import networkx as nx
    from sklearn.decomposition import TruncatedSVD

    x = graph.node_features().astype(np.float64)
    n = x.shape[0]
    ei = graph.edge_index.cpu().numpy()
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(ei.T.tolist())
    deg = np.array([G.degree(i) for i in range(n)], float)
    clus = nx.clustering(G)
    clus_a = np.array([clus[i] for i in range(n)])

    z = (x - x.mean(0)) / np.maximum(x.std(0), 1e-9)
    k = min(5, x.shape[1] - 1)
    svd = TruncatedSVD(n_components=max(k, 1), random_state=0)
    svd_res = np.linalg.norm(x - svd.inverse_transform(svd.fit_transform(x)), axis=1)
    adj = {i: list(G.neighbors(i)) for i in range(n)}
    nbm = np.array([z[adj[i]].mean(0) if adj[i] else np.zeros(z.shape[1])
                    for i in range(n)])
    featneigh = np.linalg.norm(z - nbm, axis=1)

    def pct(s):
        return np.argsort(np.argsort(s)) / max(len(s) - 1, 1)

    score = (pct(-deg) + pct(svd_res) + pct(featneigh) + pct(clus_a)) / 4.0
    return _normalize(score)


# ---------------------------------------------------------------------------
# 14. Attribute inflation: per-dimension high-rank aggregation (deterministic)
# ---------------------------------------------------------------------------

def attribute_inflation_detector(graph: GraphStore, **kw) -> np.ndarray:
    """Manipulation inflates count-like metadata across MANY dimensions at once.

    Score = mean percentile rank of a node across all feature dimensions.
    A node uniformly high across dimensions (inflated review counts, ratings,
    activity metrics) ranks top; genuine nodes are high in some dimensions
    and ordinary in others. Deterministic, no dimension selection.
    """
    x = graph.node_features().astype(np.float64)
    n = x.shape[0]
    ranks = []
    for d in range(x.shape[1]):
        col = x[:, d]
        if np.std(col) > 1e-12:
            ranks.append(np.argsort(np.argsort(col)) / max(n - 1, 1))
    if not ranks:
        return np.zeros(n)
    return _normalize(np.mean(ranks, axis=0))


# ---------------------------------------------------------------------------
# 15. Clique density: injected/organized dense-subgraph membership
# ---------------------------------------------------------------------------

def clique_density_detector(graph: GraphStore, **kw) -> np.ndarray:
    """Members of abnormally dense subgraphs (injected cliques, collusion
    rings) have extreme triangle participation weighted by local closure.
    Score = triangles(v) * clustering(v). Deterministic. Complements
    feature-based detectors: pairing via per-node rank-max (AOM) covers the
    union of contextual and structural anomaly populations.
    """
    import networkx as nx
    n = graph.num_nodes
    ei = graph.edge_index.cpu().numpy()
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(ei.T.tolist())
    tri = nx.triangles(G)
    clus = nx.clustering(G)
    score = np.array([tri[i] * clus[i] for i in range(n)], dtype=float)
    return _normalize(score)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _normalize(scores: np.ndarray) -> np.ndarray:
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-10:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


# ---------------------------------------------------------------------------
# Subgraph-contrastive paradigm (CoLA / SL-GAD family)
# ---------------------------------------------------------------------------

def _rwr_subgraphs(A: sparse.csr_matrix, size: int, rng: np.random.RandomState) -> list[list[int]]:
    """For every node, a local subgraph of `size` node ids with the target last.
    Restart-biased sampling of the immediate neighbourhood: `size-1` distinct
    1-hop neighbours, widened to 2-hop for low-degree nodes, self-filled if
    isolated. This is the local context CoLA and SL-GAD contrast a node against.
    """
    indptr, indices = A.indptr, A.indices
    reduced = size - 1
    out = []
    for i in range(A.shape[0]):
        nb = indices[indptr[i]:indptr[i + 1]]
        nb = nb[nb != i]
        if len(nb) >= reduced:
            sub = rng.choice(nb, size=reduced, replace=False).tolist()
        else:
            pool = set(int(x) for x in nb)
            for v in nb:
                pool.update(int(x) for x in indices[indptr[v]:indptr[v + 1]])
            pool.discard(i)
            pool = list(pool)
            if len(pool) >= reduced:
                sub = rng.choice(pool, size=reduced, replace=False).tolist()
            elif pool:
                sub = (pool * reduced)[:reduced]
            else:
                sub = [i] * reduced
        sub = [int(x) for x in sub]
        sub.append(int(i))
        out.append(sub)
    return out


def subgraph_contrast_detector(graph: GraphStore, hidden_dim: int = 64, epochs: int = 100,
                               subgraph_size: int = 4, test_rounds: int | None = None,
                               alpha: float = 1.0, beta: float = 0.6, seed: int = 1,
                               **kw) -> np.ndarray:
    """Local subgraph-contrastive detector (CoLA~\\cite{liu2021cola},
    SL-GAD~\\cite{zheng2021slgad} family). Each target node is contrasted against
    a readout of an RWR-sampled local subgraph through a bilinear discriminator,
    with an auxiliary generative branch that reconstructs the target's features
    from the subgraph. The score combines the contrastive margin (a node that
    does not fit its local context) and the generative residual. Effective on
    injected and contextual anomalies that break the local neighbourhood pattern
    but leave global statistics intact.
    """
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import MinMaxScaler

    n = graph.num_nodes
    # The batched GCN forward on high-dimensional features is the bottleneck and is
    # ideal for the GPU, which otherwise sits idle. Use it when available and the
    # dense path fits in memory; very large graphs stay on the original device with
    # the sparse fallback. (Set ARCADE_FORCE_CPU=1 to pin to CPU.)
    if torch.cuda.is_available() and n <= 25000 and os.environ.get("ARCADE_FORCE_CPU") != "1":
        device = torch.device("cuda")
    else:
        device = graph.x.device
    # Fixed test-round budget for EVERY graph — the detector must run at the same
    # fidelity regardless of size, exactly like the other trained paradigms (fixed
    # epochs). Do not scale this by n; a larger graph is simply slower, not cheaper.
    # 256 matches the reference SL-GAD evaluation protocol (GPU makes it cheap).
    if test_rounds is None:
        test_rounds = 256
    ei = graph.edge_index.cpu().numpy()
    A = sparse.csr_matrix((np.ones(ei.shape[1], np.float32), (ei[0], ei[1])), shape=(n, n))
    A = A + A.T
    A.data[:] = 1.0
    A = A.tocsr(); A.setdiag(0); A.eliminate_zeros()

    raw = graph.x.float().cpu().numpy().astype(np.float64)
    rowsum = raw.sum(1, keepdims=True)
    feat = raw / np.maximum(rowsum, 1e-8)                       # row-normalised input
    # symmetric-normalised adjacency with self loop
    deg = np.array(A.sum(1)).ravel()
    dinv = np.power(deg, -0.5, where=deg > 0); dinv[deg == 0] = 0.0
    D = sparse.diags(dinv)
    An = (D @ A @ D + sparse.eye(n)).tocsr()   # symmetric-normalised adjacency with self-loop
    # Subgraph adjacencies are gathered per batch. On graphs that fit in memory we
    # materialise the dense normalised adjacency once and gather every submatrix in a
    # single vectorised index op (no Python/scipy per-node loop) — this is what lets
    # the paradigm run at reference speed. On very large graphs we fall back to
    # on-the-fly sparse slicing to stay within memory. The two paths are numerically
    # identical (submatrix values match exactly).
    An_dense = None
    if n <= 25000:
        An_dense = torch.tensor(An.toarray(), dtype=torch.float32, device=device)

    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.RandomState(seed)
    feat_t = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
    raw_t = torch.tensor(raw, dtype=torch.float32, device=device).unsqueeze(0)
    feat2 = feat_t[0]; raw2 = raw_t[0]   # (n, ft) views for vectorised gather
    ft = feat.shape[1]

    class GCN(nn.Module):
        def __init__(self, i, o):
            super().__init__(); self.fc = nn.Linear(i, o, bias=False); self.act = nn.PReLU()
            nn.init.xavier_uniform_(self.fc.weight.data)
            self.bias = nn.Parameter(torch.zeros(o))
        def forward(self, s, a):
            return self.act(torch.bmm(a, self.fc(s)) + self.bias)

    class Disc(nn.Module):
        """Bilinear discriminator with in-batch negative sampling by circular
        shift of the context (one negative per node)."""
        def __init__(self, h):
            super().__init__(); self.f = nn.Bilinear(h, h, 1)
            nn.init.xavier_uniform_(self.f.weight.data); self.f.bias.data.fill_(0.0)
        def forward(self, c, h_pl):
            pos = self.f(h_pl, c)
            neg = self.f(h_pl, torch.cat((c[-1:], c[:-1]), 0))
            return torch.cat((pos, neg))                      # (2*bs, 1)

    class Net(nn.Module):
        """Two-view subgraph contrast: two independent RWR views per node,
        cross-discriminated, with a generative branch reconstructing the target
        (SL-GAD~\\cite{zheng2021slgad})."""
        def __init__(self):
            super().__init__()
            self.enc = GCN(ft, hidden_dim); self.dec = GCN(hidden_dim, ft)
            self.d1 = Disc(hidden_dim); self.d2 = Disc(hidden_dim)
            self.pdist = nn.PairwiseDistance(p=2)
        def _ctx(self, h):
            return h[:, -1, :], h[:, :-1, :].mean(1)           # target readout, context readout
        def forward(self, bf1, bf2, rbf1, rbf2, ba1, ba2):
            h1 = self.enc(bf1, ba1); h2 = self.enc(bf2, ba2)
            f1 = self.dec(self.enc(rbf1, ba1), ba1); f2 = self.dec(self.enc(rbf2, ba2), ba2)
            m1, c1 = self._ctx(h1); m2, c2 = self._ctx(h2)
            r = torch.cat((self.d1(c1, m2), self.d2(c2, m1)), -1).mean(-1)   # cross-view, (2*bs,)
            return r, f1, f2
        def infer(self, bf1, bf2, rbf1, rbf2, ba1, ba2):
            h1 = self.enc(bf1, ba1); h2 = self.enc(bf2, ba2)
            f1 = self.dec(self.enc(rbf1, ba1), ba1); f2 = self.dec(self.enc(rbf2, ba2), ba2)
            m1, c1 = self._ctx(h1); m2, c2 = self._ctx(h2)
            r = torch.cat((self.d1(c1, m2), self.d2(c2, m1)), -1).mean(-1)
            dist = 0.5 * (self.pdist(f1[:, -2, :], rbf1[:, -1, :])
                          + self.pdist(f2[:, -2, :], rbf2[:, -1, :]))
            return r, dist

    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    bs = min(300, n)   # small batches keep the within-batch contrastive negatives hard
    nb = n // bs + 1

    def assemble(idx, subs):
        # Vectorised: gather all subgraphs at once via a single index op instead of a
        # per-node loop. S holds the batch's subgraph node ids (cb x subgraph_size).
        S = np.asarray([subs[i] for i in idx], dtype=np.int64)
        St = torch.from_numpy(S).to(device)
        cb = S.shape[0]
        zrow = torch.zeros((cb, 1, ft), device=device)
        bf = feat2[St]                                   # (cb, size, ft)
        rbf = raw2[St]
        bf = torch.cat((bf[:, :-1, :], zrow, bf[:, -1:, :]), 1)
        rbf = torch.cat((rbf[:, :-1, :], zrow, rbf[:, -1:, :]), 1)
        if An_dense is not None:                         # one gather, no loop
            ba = An_dense[St[:, :, None], St[:, None, :]]  # (cb, size, size)
        else:                                            # memory-safe fallback for huge graphs
            ba = torch.empty((cb, subgraph_size, subgraph_size), device=device)
            for k in range(cb):
                s = S[k]
                ba[k] = torch.from_numpy(np.asarray(An[s][:, s].todense(), dtype=np.float32)).to(device)
        azr = torch.zeros((cb, 1, subgraph_size), device=device)
        azc = torch.zeros((cb, subgraph_size + 1, 1), device=device); azc[:, -1, :] = 1.
        ba = torch.cat((ba, azr), 1); ba = torch.cat((ba, azc), 2)
        return bf, rbf, ba

    model.train()
    best_loss, best_state = float("inf"), None
    for _ in range(epochs):
        idx_all = list(range(n)); rng.shuffle(idx_all)
        subs1 = _rwr_subgraphs(A, subgraph_size, rng)
        subs2 = _rwr_subgraphs(A, subgraph_size, rng)
        ep_loss, seen = 0.0, 0
        for b in range(nb):
            idx = idx_all[b * bs:(b + 1) * bs] if b < nb - 1 else idx_all[b * bs:]
            if not idx:
                continue
            opt.zero_grad()
            bf1, rbf1, ba1 = assemble(idx, subs1)
            bf2, rbf2, ba2 = assemble(idx, subs2)
            logits, f1, f2 = model(bf1, bf2, rbf1, rbf2, ba1, ba2)
            lbl = torch.cat((torch.ones(len(idx), device=device),
                             torch.zeros(len(idx), device=device)))
            loss = alpha * bce(logits, lbl) \
                + beta * 0.5 * (mse(f1[:, -2, :], rbf1[:, -1, :]) + mse(f2[:, -2, :], rbf2[:, -1, :]))
            loss.backward(); opt.step()
            ep_loss += float(loss) * len(idx); seen += len(idx)
        ep_loss /= max(seen, 1)
        if ep_loss < best_loss:                       # test the lowest-loss model, not the last
            best_loss = ep_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    acc = np.zeros(n)
    for _ in range(test_rounds):
        # Shuffle nodes into batches every round: the contrastive negative is the
        # previous node's context (circular shift), so batches must be randomised
        # or the negative for node i is its ID-neighbour i-1 — often the same
        # injected clique/community — which collapses the contrastive signal. The
        # generative branch uses no negatives and is unaffected. (Matches SL-GAD.)
        idx_all = list(range(n)); rng.shuffle(idx_all)
        subs1 = _rwr_subgraphs(A, subgraph_size, rng)
        subs2 = _rwr_subgraphs(A, subgraph_size, rng)
        for b in range(nb):
            idx = idx_all[b * bs:(b + 1) * bs] if b < nb - 1 else idx_all[b * bs:]
            if not idx:
                continue
            bf1, rbf1, ba1 = assemble(idx, subs1)
            bf2, rbf2, ba2 = assemble(idx, subs2)
            with torch.no_grad():
                logits, dist = model.infer(bf1, bf2, rbf1, rbf2, ba1, ba2)
                logits = torch.sigmoid(logits)
            cb = len(idx)
            contr = -(logits[:cb] - logits[cb:]).cpu().numpy()
            c_n = MinMaxScaler().fit_transform(contr.reshape(-1, 1)).reshape(-1)
            g_n = MinMaxScaler().fit_transform(dist.cpu().numpy().reshape(-1, 1)).reshape(-1)
            acc[idx] += alpha * c_n + beta * g_n
    return _normalize(acc / test_rounds)
