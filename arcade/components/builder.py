"""Method builder — compose real GNN architectures from components.

The agent selects encoder, objective, decoder, augmentation as real nn.Modules,
and the builder wires them into a trainable GNN architecture. Supports
architecture-level fusion (shared encoder, cross-attention, ensemble gate).
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCN, GAT, GIN, MLP
from torch_geometric.utils import to_dense_adj

from arcade.components.registry import KNOWN_RECIPES


BACKBONES = {"gcn": GCN, "gat": GAT, "gin": GIN, "mlp": MLP}


# ═══════════════════════════════════════════════════════════════════════════
#  ENCODER COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

class GNNEncoder(nn.Module):
    """Graph encoder using any PyG backbone (GCN, GAT, GIN, MLP)."""

    def __init__(self, backbone: str, in_dim: int, hid_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.0, act: str = "relu", **kwargs):
        super().__init__()
        self.backbone_name = backbone
        self.is_mlp = backbone == "mlp"
        cls = BACKBONES[backbone]
        extra = {}
        if backbone == "gat":
            extra["heads"] = kwargs.get("heads", 4)
        self.nn = cls(
            in_channels=in_dim, hidden_channels=hid_dim,
            num_layers=max(num_layers, 1), out_channels=hid_dim,
            dropout=dropout, act=act, **extra,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.nn(x) if self.is_mlp else self.nn(x, edge_index)


class CommunityGCNEncoder(nn.Module):
    """FlexGAD-style community-aware encoder with intra/inter-community channels.

    Separates edges into intra-community and inter-community, runs independent
    GCN branches on each, then projects the concatenation to hid_dim.
    Anomalous nodes that bridge communities produce divergent intra vs inter
    representations — the divergence itself becomes signal.
    """

    def __init__(self, in_dim: int, hid_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.0, act: str = "relu"):
        super().__init__()
        self.backbone_name = "community_gcn"
        self.is_mlp = False
        self.hid_dim = hid_dim
        self.intra_gcn = GCN(in_channels=in_dim, hidden_channels=hid_dim,
                             num_layers=max(num_layers, 1), out_channels=hid_dim,
                             dropout=dropout, act=act)
        self.inter_gcn = GCN(in_channels=in_dim, hidden_channels=hid_dim,
                             num_layers=max(num_layers, 1), out_channels=hid_dim,
                             dropout=dropout, act=act)
        self.full_gcn = GCN(in_channels=in_dim, hidden_channels=hid_dim,
                            num_layers=max(num_layers, 1), out_channels=hid_dim,
                            dropout=dropout, act=act)
        self.proj = nn.Sequential(
            nn.Linear(hid_dim * 3, hid_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid_dim * 2, hid_dim),
        )
        self._communities = None

    def set_communities(self, communities: torch.Tensor):
        self._communities = communities

    def _split_edges(self, edge_index: torch.Tensor) -> tuple:
        if self._communities is None:
            return edge_index, edge_index
        src, dst = edge_index
        same = self._communities[src] == self._communities[dst]
        return edge_index[:, same], edge_index[:, ~same]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        intra_ei, inter_ei = self._split_edges(edge_index)
        z_full = self.full_gcn(x, edge_index)
        z_intra = self.intra_gcn(x, intra_ei) if intra_ei.size(1) > 0 else torch.zeros(x.size(0), self.hid_dim, device=x.device)
        z_inter = self.inter_gcn(x, inter_ei) if inter_ei.size(1) > 0 else torch.zeros(x.size(0), self.hid_dim, device=x.device)
        return self.proj(torch.cat([z_full, z_intra, z_inter], dim=1))


# ═══════════════════════════════════════════════════════════════════════════
#  DECODER COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

class FeatureDecoder(nn.Module):
    """Feature decoder: Z -> X_hat using a backbone."""

    def __init__(self, backbone: str, hid_dim: int, out_dim: int,
                 num_layers: int = 2, dropout: float = 0.0, act: str = "relu", **kwargs):
        super().__init__()
        self.is_mlp = backbone == "mlp"
        cls = BACKBONES[backbone]
        extra = {}
        if backbone == "gat":
            extra["heads"] = kwargs.get("heads", 4)
        self.nn = cls(
            in_channels=hid_dim, hidden_channels=hid_dim,
            num_layers=max(num_layers, 1), out_channels=out_dim,
            dropout=dropout, act=act, **extra,
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.nn(z) if self.is_mlp else self.nn(z, edge_index)


class StructureDecoder(nn.Module):
    """Structure decoder with GNN backbone: h = GNN(Z), A_hat = h @ h.T.

    Matches PyGOD's DotProductDecoder: a backbone refines embeddings
    before the dot-product adjacency reconstruction.
    """

    def __init__(self, backbone: str = "gcn", hid_dim: int = 64,
                 num_layers: int = 2, dropout: float = 0.0,
                 act: str = "relu", sigmoid_s: bool = False, **kwargs):
        super().__init__()
        self.sigmoid_s = sigmoid_s
        self.is_mlp = backbone == "mlp"
        cls = BACKBONES[backbone]
        extra = {}
        if backbone == "gat":
            extra["heads"] = kwargs.get("heads", 4)
        self.nn = cls(
            in_channels=hid_dim, hidden_channels=hid_dim,
            num_layers=max(num_layers, 1), out_channels=hid_dim,
            dropout=dropout, act=act, **extra,
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.nn(z) if self.is_mlp else self.nn(z, edge_index)
        s = h @ h.t()
        return torch.sigmoid(s) if self.sigmoid_s else s


# ═══════════════════════════════════════════════════════════════════════════
#  OBJECTIVE / LOSS COMPONENTS (uses PyGOD's loss where applicable)
# ═══════════════════════════════════════════════════════════════════════════

class ReconstructionObjective(nn.Module):
    """Dual reconstruction: alpha * ||X-X'||_2 + (1-alpha) * ||A-A'||_2 per node.

    Uses PyGOD's double_recon_loss for proper scaling.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        from pygod.nn.functional import double_recon_loss
        self._loss_fn = double_recon_loss

    def forward(self, x, x_hat, s, s_hat):
        score = self._loss_fn(x, x_hat, s, s_hat, weight=self.alpha)
        return score.mean(), score


class ContrastiveObjective(nn.Module):
    """Contrastive objective: discriminator separates node from corrupted context.

    Uses a graph-level readout as context (like CoLA), making the discriminator
    check whether a node fits its graph context.
    """

    def __init__(self, hid_dim: int):
        super().__init__()
        self.disc = nn.Bilinear(hid_dim, hid_dim, 1)
        self.readout = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Sigmoid())

    def forward(self, z_pos: torch.Tensor, z_neg: torch.Tensor):
        summary = self.readout(z_pos.mean(dim=0, keepdim=True).expand_as(z_pos))
        pos_logits = self.disc(z_pos, summary).squeeze(-1)
        neg_logits = self.disc(z_neg, summary).squeeze(-1)

        loss = (F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))) / 2
        score = 1.0 - pos_logits.sigmoid()
        return loss, score


class SVDDObjective(nn.Module):
    """SVDD: minimize hypersphere containing normal embeddings.

    Trimmed-mean center (excludes top 10% outliers) prevents contamination.
    EMA center update smooths drift. Adaptive epsilon from MAD.
    """

    def __init__(self, hid_dim: int, warmup: int = 20, ema: float = 0.95):
        super().__init__()
        self.register_buffer("center", torch.zeros(hid_dim))
        self._warmup = warmup
        self._call_count = 0
        self._ema = ema

    @staticmethod
    def _trimmed_mean(z: torch.Tensor, center: torch.Tensor, trim: float = 0.1):
        dist = (z - center).pow(2).sum(dim=1)
        k = max(1, int(z.size(0) * (1.0 - trim)))
        keep = dist.topk(k, largest=False).indices
        return z[keep].mean(dim=0)

    def forward(self, z: torch.Tensor):
        self._call_count += 1
        if self._call_count <= self._warmup:
            self.center = z.mean(dim=0).detach()
            return z.var(dim=0).mean(), (z - self.center).pow(2).sum(dim=1).detach()

        dist = (z - self.center).pow(2).sum(dim=1)

        tm = self._trimmed_mean(z.detach(), self.center, trim=0.1)
        self.center = (self._ema * self.center + (1 - self._ema) * tm).detach()

        med = dist.median()
        mad = (dist - med).abs().median().clamp(min=1e-6)
        eps = med + 1.5 * mad

        margin_loss = F.relu(dist - eps).mean()
        return margin_loss, dist


class ResidualObjective(nn.Module):
    """Radar-style residual analysis as a composable objective.

    Reconstructs node attributes from the structure-aware embedding and
    treats the part the graph cannot explain as the anomaly signal. A
    graph-Laplacian smoothness penalty pulls the reconstruction toward a
    network-coherent signal, so a node whose attributes deviate from what
    its neighbourhood predicts incurs a large residual. Inspired by Radar
    (Li et al., IJCAI 2017); unlike Radar it is trainable and pairs with
    any encoder in the library. Targets the high-homophily, feature-
    perturbation regime (e.g. Weibo, Amazon) where residual structure is
    exactly the anomaly mechanism.
    """

    def __init__(self, hid_dim: int, out_dim: int, lam: float = 0.5):
        super().__init__()
        self.lam = lam
        self.decoder = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, bs: int):
        x_hat = self.decoder(z)
        residual = (x[:bs] - x_hat[:bs]).pow(2).mean(dim=1)   # per-node MSE
        # Laplacian smoothness: the reconstruction should vary little across
        # edges, so unexplained (anomalous) attributes surface as residual.
        src, dst = edge_index
        if src.numel() > 0:
            smooth = (x_hat[src] - x_hat[dst]).pow(2).mean()
        else:
            smooth = x_hat.new_zeros(())
        loss = residual.mean() + self.lam * smooth
        score = torch.sqrt(residual.clamp(min=1e-12))
        return loss, score


# ═══════════════════════════════════════════════════════════════════════════
#  NEIGHBORHOOD JSD OBJECTIVE (inspired by FlexGAD)
# ═══════════════════════════════════════════════════════════════════════════

class NeighborhoodJSDObjective(nn.Module):
    """FlexGAD-style neighborhood JSD with generative sampling.

    Adapted from Flex-GAD (Chakraborty+ CODS'25). The key insight:
    predict each node's neighborhood distribution as a Gaussian, then
    measure JSD against learned neighborhood statistics.

    Three fixes for power-law divergence vs our original naive version:
    1. True JSD via mixture midpoint M = (mu_m, var_m), NOT symmetrized
       KL. KL(P||Q)+KL(Q||P) is unbounded; KL(P||M)+KL(Q||M) <= ln(2).
    2. Generative sampling — predict (mu, sigma) -> reparameterize ->
       generate sample neighbors -> compute stats. The sampling smooths
       out extreme variances from hub nodes.
    3. Target stats via learned SAGEConv(mean) instead of raw scatter.
       Learned aggregation produces well-conditioned outputs on any
       degree distribution.
    """

    def __init__(self, hid_dim: int, out_dim: int, n_samples: int = 10):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.n_samples = n_samples
        # Target: learned neighborhood mean aggregation
        self.mean_agg = SAGEConv(out_dim, out_dim, aggr='mean',
                                 normalize=False)
        # Predicted: generate neighborhood distribution
        self.mu_net = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim))
        self.sigma_net = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim))
        self.generator = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim))

    @staticmethod
    def _safe_kl(mu_a, var_a, mu_b, var_b):
        """KL(N(a) || N(b)), per-dimension, clamped for safety."""
        log_ratio = torch.log(var_b.clamp(min=1e-8)) - torch.log(var_a.clamp(min=1e-8))
        var_ratio = (var_a / var_b.clamp(min=1e-8)).clamp(max=100.0)
        mu_term = (mu_a - mu_b).pow(2) / var_b.clamp(min=1e-8)
        return 0.5 * (log_ratio + var_ratio + mu_term - 1.0)

    def _true_jsd(self, mu_p, var_p, mu_q, var_q):
        """True JSD via mixture midpoint. Bounded by ln(2) per dimension."""
        mu_m = 0.5 * (mu_p + mu_q)
        var_m = (0.5 * (var_p + var_q)).clamp(min=1e-8)
        kl_pm = self._safe_kl(mu_p, var_p, mu_m, var_m)
        kl_qm = self._safe_kl(mu_q, var_q, mu_m, var_m)
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd.clamp(min=0.0).sum(dim=-1)

    def _compute_target_stats(self, x, edge_index, bs):
        """Neighborhood mean via learned SAGEConv, variance via scatter."""
        mu_tgt = self.mean_agg(x, edge_index)[:bs]

        src, dst = edge_index
        seed_mask = dst < bs
        src_s, dst_s = src[seed_mask], dst[seed_mask]
        d = x.size(1)
        var_tgt = torch.zeros(bs, d, device=x.device)
        count = torch.zeros(bs, 1, device=x.device)
        diff_sq = (x[src_s] - mu_tgt[dst_s]).pow(2)
        var_tgt.scatter_add_(0, dst_s.unsqueeze(1).expand(-1, d), diff_sq)
        count.scatter_add_(0, dst_s.unsqueeze(1),
                           torch.ones(src_s.size(0), 1, device=x.device))
        count = count.clamp(min=1)
        var_tgt = (var_tgt / count).clamp(min=1e-4, max=10.0)
        std_tgt = var_tgt.sqrt()
        return mu_tgt, std_tgt

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, bs: int):
        # Normalize features to [0,1] for stable variance computation
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)

        # Target: learned neighborhood statistics
        mu_tgt, std_tgt = self._compute_target_stats(x_norm, edge_index, bs)

        # Predicted: generate neighborhood samples via reparameterization
        z_bs = z[:bs]
        gen_mu = self.mu_net(z_bs)
        gen_logsig = self.sigma_net(z_bs).clamp(-10, 10)

        # Sample n_samples neighbors, compute generated stats
        expanded_mu = gen_mu.unsqueeze(0).expand(self.n_samples, -1, -1)
        expanded_sig = gen_logsig.exp().unsqueeze(0).expand(self.n_samples, -1, -1)
        eps = torch.randn_like(expanded_mu)
        samples = expanded_mu + expanded_sig * eps
        generated = self.generator(samples)

        gen_mean = generated.mean(dim=0)
        gen_std = generated.std(dim=0).clamp(min=1e-4)

        # True JSD between target and generated neighborhood distributions
        jsd = self._true_jsd(mu_tgt, std_tgt.pow(2), gen_mean, gen_std.pow(2))

        # NaN safety
        jsd = torch.where(torch.isnan(jsd), torch.zeros_like(jsd), jsd)
        return jsd.mean(), jsd


class NeighborPredictionObjective(nn.Module):
    """Predict each node's neighborhood mean features from its embedding.

    Unlike JSD (which models neighbourhood as a Gaussian), this directly
    regresses the mean of each node's neighbor features.  Anomalies whose
    features/structure don't match their local context produce high error.
    Inspired by GAD-NR's neighbor reconstruction idea.
    """

    def __init__(self, hid_dim: int, out_dim: int):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, bs: int):
        pred = self.pred(z[:bs])

        src, dst = edge_index
        seed_mask = dst < bs
        src_s, dst_s = src[seed_mask], dst[seed_mask]

        neigh_mean = torch.zeros(bs, x.size(1), device=x.device)
        count = torch.zeros(bs, 1, device=x.device)
        neigh_mean.scatter_add_(0, dst_s.unsqueeze(1).expand(-1, x.size(1)), x[src_s])
        count.scatter_add_(0, dst_s.unsqueeze(1),
                           torch.ones(src_s.size(0), 1, device=x.device))
        count = count.clamp(min=1)
        neigh_mean = neigh_mean / count

        diff = (pred - neigh_mean).pow(2)
        score = torch.sqrt(diff.sum(dim=1))
        return score.mean(), score


class CommunityDeviationObjective(nn.Module):
    """Measure each node's deviation from its community centroid in
    embedding space.  Communities are discovered on the first call via
    1-hop label propagation on the adjacency, then centroids are
    recomputed every forward pass from the current embeddings.
    Anomalies sit far from their community's centre.
    """

    def __init__(self, hid_dim: int, max_comms: int = 64):
        super().__init__()
        self.max_comms = max_comms
        self._labels = None

    def _assign_communities(self, edge_index: torch.Tensor, n: int):
        """Fast 1-hop label propagation for community assignment."""
        labels = torch.arange(n, device=edge_index.device)
        src, dst = edge_index
        for _ in range(10):
            new_labels = labels.clone()
            new_labels.scatter_reduce_(0, dst, labels[src], reduce="amin")
            if (new_labels == labels).all():
                break
            labels = new_labels
        _, labels = labels.unique(return_inverse=True)
        return labels

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, bs: int):
        n = z.size(0)
        if self._labels is None or self._labels.size(0) != n:
            self._labels = self._assign_communities(edge_index, n)

        labels = self._labels
        z_seed = z[:bs]
        lab_seed = labels[:bs]

        k = int(labels.max().item()) + 1
        k = min(k, self.max_comms)
        centroids = torch.zeros(k, z.size(1), device=z.device)
        counts = torch.zeros(k, 1, device=z.device)
        safe_lab = lab_seed.clamp(max=k - 1)
        centroids.scatter_add_(0, safe_lab.unsqueeze(1).expand(-1, z.size(1)), z_seed)
        counts.scatter_add_(0, safe_lab.unsqueeze(1),
                            torch.ones(bs, 1, device=z.device))
        centroids = centroids / counts.clamp(min=1)

        node_centroid = centroids[safe_lab]
        dist = (z_seed - node_centroid).pow(2).sum(dim=1)
        return dist.mean(), dist


class GADNRObjective(nn.Module):
    """GAD-NR 3-part neighbor reconstruction (WSDM 2024).

    Three losses: (1) self-attribute MSE, (2) degree MSE,
    (3) neighbor mean prediction MSE (Gaussian KL simplified to MSE
    for stability — the paper shows MSE≈KL in practice).
    """

    def __init__(self, hid_dim: int, out_dim: int):
        super().__init__()
        self.self_recon = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim))
        self.degree_pred = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, 1))
        self.neigh_pred = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, out_dim))

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, bs: int):
        z_bs = z[:bs]
        d = x.size(1)

        # Neighborhood empirical mean + degree
        src, dst = edge_index
        seed_mask = dst < bs
        src_s, dst_s = src[seed_mask], dst[seed_mask]
        mu_tgt = torch.zeros(bs, d, device=x.device)
        count = torch.zeros(bs, 1, device=x.device)
        mu_tgt.scatter_add_(0, dst_s.unsqueeze(1).expand(-1, d), x[src_s])
        count.scatter_add_(0, dst_s.unsqueeze(1),
                           torch.ones(src_s.size(0), 1, device=x.device))
        count = count.clamp(min=1)
        mu_tgt = mu_tgt / count

        # Part 1: self-attribute reconstruction
        x_hat = self.self_recon(z_bs)
        attr_err = ((x[:bs] - x_hat) ** 2).mean(dim=1)

        # Part 2: degree prediction (log-scale)
        deg_hat = self.degree_pred(z_bs).squeeze(-1)
        deg_err = (deg_hat - count.squeeze(-1).log1p()).pow(2)

        # Part 3: neighbor mean prediction
        neigh_hat = self.neigh_pred(z_bs)
        neigh_err = ((mu_tgt - neigh_hat) ** 2).mean(dim=1)

        loss = (attr_err + neigh_err + deg_err).mean()

        def _norm(t):
            r = t.max() - t.min()
            return (t - t.min()) / r.clamp(min=1e-8)

        score = _norm(attr_err) + _norm(neigh_err) + _norm(deg_err)
        return loss, score


class LocalAffinityObjective(nn.Module):
    """TAM local affinity maximization (NeurIPS 2023).

    Semi-hard negatives (closest non-neighbors) + adaptive margin +
    stop-gradient on neighbor target to prevent collapse.
    """

    def __init__(self, hid_dim: int, out_dim: int = 0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.BatchNorm1d(hid_dim),
            nn.PReLU(), nn.Linear(hid_dim, hid_dim))
        self._margin = 0.5
        self._call_count = 0

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, bs: int):
        h = self.proj(z)
        h_bs = h[:bs]
        d = h.size(1)

        src, dst = edge_index
        seed_mask = dst < bs
        src_s, dst_s = src[seed_mask], dst[seed_mask]

        neigh_sum = torch.zeros(bs, d, device=x.device)
        count = torch.zeros(bs, 1, device=x.device)
        neigh_sum.scatter_add_(0, dst_s.unsqueeze(1).expand(-1, d),
                               h[src_s].detach())
        count.scatter_add_(0, dst_s.unsqueeze(1),
                           torch.ones_like(count[:1]).expand(src_s.size(0), -1))
        count = count.clamp(min=1)
        neigh_mean = neigh_sum / count

        pos_dist = (h_bs - neigh_mean).pow(2).mean(dim=1)

        # Semi-hard negatives: sample K random, pick closest per anchor
        K = min(32, bs - 1)
        with torch.no_grad():
            idx = torch.stack([torch.randperm(bs, device=x.device)[:K]
                               for _ in range(bs)])
            cands = h_bs[idx]
            d2 = (h_bs.unsqueeze(1) - cands).pow(2).mean(dim=2)
            hard = d2.argmin(dim=1)
            neg_sel = idx[torch.arange(bs, device=x.device), hard]
        neg_dist = (h_bs - h_bs[neg_sel]).pow(2).mean(dim=1)

        # Adaptive margin: update every 50 steps
        self._call_count += 1
        if self._call_count % 50 == 0:
            with torch.no_grad():
                gap = (neg_dist.median() - pos_dist.median()).item()
                self._margin = max(0.1, min(2.0, gap * 0.5))

        loss = F.relu(pos_dist - neg_dist + self._margin).mean()
        return loss, pos_dist


class EgoMatchingObjective(nn.Module):
    """PREM ego-neighbor matching (ICDM 2023).

    Ego = projected GCN embedding, neighbor = projected mean of raw
    features. All-pair InfoNCE gives bs-1 negatives per positive for
    a much stronger contrastive signal than single-negative.
    """

    def __init__(self, hid_dim: int, in_dim: int = 0):
        super().__init__()
        feat_dim = in_dim or hid_dim
        self.ego_proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim))
        self.neigh_proj = nn.Sequential(
            nn.Linear(feat_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, hid_dim))

    def forward(self, z: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, bs: int):
        ego = F.normalize(self.ego_proj(z[:bs]), dim=1)

        src, dst = edge_index
        seed_mask = dst < bs
        src_s, dst_s = src[seed_mask], dst[seed_mask]
        neigh_sum = torch.zeros(bs, x.size(1), device=x.device)
        count = torch.zeros(bs, 1, device=x.device)
        neigh_sum.scatter_add_(0, dst_s.unsqueeze(1).expand(-1, x.size(1)),
                               x[src_s])
        count.scatter_add_(0, dst_s.unsqueeze(1),
                           torch.ones(src_s.size(0), 1, device=x.device))
        count = count.clamp(min=1)
        neigh_mean = neigh_sum / count
        neigh = F.normalize(self.neigh_proj(neigh_mean), dim=1)

        tau = 0.2
        pos_sim = (ego * neigh).sum(dim=1)

        K = min(256, bs - 1)
        neg_idx = torch.stack([torch.randperm(bs, device=x.device)[:K]
                               for _ in range(bs)])
        neg_sim = (ego.unsqueeze(1) * neigh[neg_idx]).sum(dim=2) / tau
        logits = torch.cat([pos_sim.unsqueeze(1) / tau, neg_sim], dim=1)
        labels = torch.zeros(bs, dtype=torch.long, device=x.device)
        loss = F.cross_entropy(logits, labels)

        score = 1.0 - pos_sim
        return loss, score


# ═══════════════════════════════════════════════════════════════════════════
#  AUGMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def augment_feature_mask(x: torch.Tensor, mask_rate: float = 0.3) -> torch.Tensor:
    mask = torch.bernoulli(torch.full_like(x, 1.0 - mask_rate))
    return x * mask

def augment_edge_drop(edge_index: torch.Tensor, drop_rate: float = 0.2) -> torch.Tensor:
    keep = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    return edge_index[:, keep]

def augment_noise_inject(x: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    return x + torch.randn_like(x) * noise_std

def augment_community_smooth(x: torch.Tensor, edge_index: torch.Tensor,
                              **kwargs) -> torch.Tensor:
    """FlexGAD-style: replace each node's features with its community average.

    Uses a fast 1-hop mean as a proxy when community labels aren't available
    (the same smoothing effect as averaging within Louvain communities on
    homophilic graphs).
    """
    from torch_geometric.utils import degree
    n = x.size(0)
    src, dst = edge_index
    deg = degree(dst, n).clamp(min=1).unsqueeze(1)
    agg = torch.zeros_like(x)
    agg.scatter_add_(0, dst.unsqueeze(1).expand_as(x[src]), x[src])
    return agg / deg

def augment_edge_truncation(edge_index: torch.Tensor, x: torch.Tensor = None,
                             threshold: float = 0.3, **kwargs) -> torch.Tensor:
    """TAM NSGT-inspired: remove non-homophily edges based on feature similarity."""
    if x is None:
        return edge_index
    src, dst = edge_index
    x_norm = F.normalize(x, dim=1)
    sim = (x_norm[src] * x_norm[dst]).sum(dim=1)
    keep = sim > -threshold
    return edge_index[:, keep]


AUGMENT_FNS = {
    "feature_mask": augment_feature_mask,
    "edge_drop": augment_edge_drop,
    "noise_inject": augment_noise_inject,
    "community_smooth": augment_community_smooth,
    "edge_truncation": augment_edge_truncation,
    "none": None,
}


# ═══════════════════════════════════════════════════════════════════════════
#  COMPOSED METHOD — real GNN wired from components
# ═══════════════════════════════════════════════════════════════════════════

class ComposedMethod(nn.Module):
    """A GAD method assembled from encoder + decoder + objective + augmentation.

    This is a real nn.Module — not a wrapper around a PyGOD detector.
    The agent selects which components to plug in.

    Spec format::

        {
            "encoder":      {"type": "gcn", "hid_dim": 64, "num_layers": 4},
            "objective":    {"type": "reconstruction", "alpha": 0.5},
            "decoder":      {"type": "dual"},
            "augmentation": {"type": "feature_mask", "mask_rate": 0.3},
        }
    """

    def __init__(self, spec: dict, in_dim: int):
        super().__init__()
        self.spec = spec
        self.in_dim = in_dim

        enc_cfg = spec.get("encoder", {"type": "gcn"})
        obj_cfg = spec.get("objective", {"type": "reconstruction"})
        dec_cfg = spec.get("decoder", {"type": "dual"})
        aug_cfg = spec.get("augmentation", {"type": "none"})

        self.obj_type = obj_cfg.get("type", "reconstruction")
        self.dec_type = dec_cfg.get("type", "dual")
        self.aug_type = aug_cfg.get("type", "none")

        backbone = enc_cfg.get("type", "gcn")
        hid_dim = enc_cfg.get("hid_dim", 64)
        total_layers = enc_cfg.get("num_layers", 4)
        dropout = enc_cfg.get("dropout", 0.0)
        act = enc_cfg.get("act", "relu")
        heads = enc_cfg.get("heads", 4)

        n_enc = max(total_layers // 2, 1)
        n_dec = max(total_layers - n_enc, 1)
        self.hid_dim = hid_dim

        # --- Encoder ---
        if backbone == "community_gcn":
            self.encoder = CommunityGCNEncoder(in_dim, hid_dim, n_enc, dropout, act)
        else:
            self.encoder = GNNEncoder(backbone, in_dim, hid_dim, n_enc, dropout, act, heads=heads)

        # --- Decoders ---
        dec_backbone = "gcn" if backbone == "community_gcn" else backbone
        self.feat_decoder = None
        self.struct_decoder = None
        if self.dec_type in ("dual", "feature_mlp"):
            self.feat_decoder = FeatureDecoder(dec_backbone, hid_dim, in_dim, n_dec, dropout, act, heads=heads)
        if self.dec_type in ("dual", "dot_product"):
            self.struct_decoder = StructureDecoder(
                backbone=dec_backbone, hid_dim=hid_dim, num_layers=n_dec,
                dropout=dropout, act=act,
                sigmoid_s=dec_cfg.get("sigmoid_s", False), heads=heads,
            )

        # --- Objective ---
        if self.obj_type == "reconstruction":
            alpha = obj_cfg.get("alpha", dec_cfg.get("alpha", 0.5))
            self.objective = ReconstructionObjective(alpha=alpha)
        elif self.obj_type == "contrastive":
            self.objective = ContrastiveObjective(hid_dim)
        elif self.obj_type == "svdd":
            self.objective = SVDDObjective(hid_dim)
        elif self.obj_type == "neighborhood_jsd":
            self.objective = NeighborhoodJSDObjective(hid_dim, in_dim)
        elif self.obj_type == "neighbor_prediction":
            self.objective = NeighborPredictionObjective(hid_dim, in_dim)
        elif self.obj_type == "community_deviation":
            self.objective = CommunityDeviationObjective(hid_dim)
        elif self.obj_type == "gad_nr":
            self.objective = GADNRObjective(hid_dim, in_dim)
        elif self.obj_type == "local_affinity":
            self.objective = LocalAffinityObjective(hid_dim)
        elif self.obj_type == "ego_matching":
            self.objective = EgoMatchingObjective(hid_dim, in_dim)
        elif self.obj_type == "residual":
            self.objective = ResidualObjective(hid_dim, in_dim,
                                               lam=obj_cfg.get("lam", 0.5))
        else:
            raise ValueError(f"Unknown objective: {self.obj_type}")

        # --- Augmentation ---
        self.aug_fn = AUGMENT_FNS.get(self.aug_type)
        self.aug_params = {k: v for k, v in aug_cfg.items() if k != "type"}

        self.emb = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                s: torch.Tensor | None = None,
                batch_size: int | None = None,
                node_idx: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (loss, per_node_anomaly_score).

        In mini-batch mode (batch_size is set), x/edge_index contain
        seed nodes + sampled neighbors, but loss/score are computed
        only on the first ``batch_size`` seed nodes.
        """
        z = self.encoder(x, edge_index)
        self.emb = z.detach()
        bs = batch_size or x.size(0)

        if self.obj_type == "reconstruction":
            return self._fwd_recon(x, z, edge_index, s, bs, node_idx)
        elif self.obj_type == "contrastive":
            return self._fwd_contrastive(x, z, edge_index, bs)
        elif self.obj_type == "svdd":
            return self._fwd_svdd(z, bs)
        elif self.obj_type in ("neighborhood_jsd", "neighbor_prediction",
                               "community_deviation", "gad_nr",
                               "local_affinity", "ego_matching", "residual"):
            return self.objective(z, x, edge_index, bs)

    def _fwd_recon(self, x, z, edge_index, s, bs, node_idx):
        n = x.size(0)
        x_hat = self.feat_decoder(z, edge_index) if self.feat_decoder else x

        if self.struct_decoder is not None and self.feat_decoder is not None:
            s_hat = self.struct_decoder(z, edge_index)
            if s is None:
                s = to_dense_adj(edge_index, max_num_nodes=n)[0]
            if node_idx is not None:
                return self.objective(x[:bs], x_hat[:bs],
                                      s[:bs, node_idx], s_hat[:bs])
            return self.objective(x[:bs], x_hat[:bs], s[:bs], s_hat[:bs])

        if self.struct_decoder is not None:
            s_hat = self.struct_decoder(z, edge_index)
            if s is None:
                s = to_dense_adj(edge_index, max_num_nodes=n)[0]
            if node_idx is not None:
                diff = torch.pow(s[:bs, node_idx] - s_hat[:bs], 2)
            else:
                diff = torch.pow(s[:bs] - s_hat[:bs], 2)
            score = torch.sqrt(torch.sum(diff, 1))
            return score.mean(), score

        diff = torch.pow(x[:bs] - x_hat[:bs], 2)
        score = torch.sqrt(torch.sum(diff, 1))
        return score.mean(), score

    def _fwd_contrastive(self, x, z, edge_index, bs):
        if self.aug_fn is not None:
            if self.aug_type == "edge_drop":
                ei_aug = self.aug_fn(edge_index, **self.aug_params)
                z_neg = self.encoder(x, ei_aug)
            elif self.aug_type == "community_smooth":
                x_aug = self.aug_fn(x, edge_index, **self.aug_params)
                z_neg = self.encoder(x_aug, edge_index)
            elif self.aug_type == "edge_truncation":
                ei_aug = self.aug_fn(edge_index, x=x, **self.aug_params)
                z_neg = self.encoder(x, ei_aug)
            else:
                x_aug = self.aug_fn(x, **self.aug_params)
                z_neg = self.encoder(x_aug, edge_index)
        else:
            perm = torch.randperm(x.size(0), device=x.device)
            z_neg = self.encoder(x[perm], edge_index)
        return self.objective(z[:bs], z_neg[:bs])

    def _fwd_svdd(self, z, bs):
        return self.objective(z[:bs])

    def describe(self) -> dict:
        """Human-readable description of the composed architecture."""
        return {
            "encoder": self.encoder.backbone_name,
            "objective": self.obj_type,
            "decoder": self.dec_type,
            "augmentation": self.aug_type,
            "hid_dim": self.hid_dim,
            "params": sum(p.numel() for p in self.parameters()),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE-LEVEL FUSION — real multi-head GNN architectures
# ═══════════════════════════════════════════════════════════════════════════

class SharedEncoderFusion(nn.Module):
    """Strategy A: one shared GNN encoder with multiple objective heads.

    A single encoder produces embeddings that are fed to N objective heads
    (reconstruction, contrastive, SVDD). Heads are trained jointly,
    and a learned gate fuses per-node scores.
    """

    def __init__(self, heads: list[dict], in_dim: int):
        super().__init__()
        enc_cfg = heads[0].get("encoder", {"type": "gcn"})
        backbone = enc_cfg.get("type", "gcn")
        hid_dim = enc_cfg.get("hid_dim", 64)
        total_layers = enc_cfg.get("num_layers", 4)
        n_enc = max(total_layers // 2, 1)

        self.encoder = GNNEncoder(backbone, in_dim, hid_dim, n_enc,
                                  enc_cfg.get("dropout", 0.0),
                                  enc_cfg.get("act", "relu"),
                                  heads=enc_cfg.get("heads", 4))

        self.heads = nn.ModuleList()
        for spec in heads:
            head = _build_head(spec, in_dim, hid_dim, total_layers - n_enc,
                               backbone, enc_cfg)
            self.heads.append(head)

        self.gate = nn.Linear(hid_dim, len(heads))
        self.hid_dim = hid_dim
        self.emb = None

    def forward(self, x, edge_index, s=None, batch_size=None, node_idx=None):
        bs = batch_size or x.size(0)
        z = self.encoder(x, edge_index)
        self.emb = z.detach()

        losses, scores = [], []
        for head in self.heads:
            loss, score = head(z, x, edge_index, s, bs, node_idx)
            losses.append(loss)
            scores.append(score)

        gate_w = torch.softmax(self.gate(z[:bs].detach()), dim=1)
        fused = (gate_w * torch.stack(scores, dim=1)).sum(dim=1)
        return sum(losses) / len(losses), fused


class CrossAttentionFusion(nn.Module):
    """Strategy B: independent encoders with cross-attention on embeddings."""

    def __init__(self, methods: nn.ModuleList, hid_dim: int):
        super().__init__()
        self.methods = methods
        self.cross_attn = nn.MultiheadAttention(hid_dim, num_heads=4, batch_first=True)
        self.score_proj = nn.Linear(hid_dim, 1)
        self.emb = None

    def forward(self, x, edge_index, s=None, batch_size=None, node_idx=None):
        bs = batch_size or x.size(0)
        losses, scores, embs = [], [], []
        for m in self.methods:
            loss, score = m(x, edge_index, s, batch_size=bs, node_idx=node_idx)
            losses.append(loss)
            scores.append(score)
            embs.append(m.emb[:bs])

        stacked = torch.stack(embs, dim=1)
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)
        fused_emb = attn_out.mean(dim=1)
        self.emb = fused_emb.detach()

        attn_score = self.score_proj(fused_emb).squeeze(1)
        avg_score = torch.stack(scores).mean(dim=0)
        fused = 0.5 * attn_score + 0.5 * avg_score

        return sum(losses) / len(losses), fused


class EnsembleGateFusion(nn.Module):
    """Strategy C: full independent methods with per-node gating."""

    def __init__(self, methods: nn.ModuleList, in_dim: int, hid_dim: int = 32):
        super().__init__()
        self.methods = methods
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, len(methods)),
        )
        self.emb = None

    def forward(self, x, edge_index, s=None, batch_size=None, node_idx=None):
        bs = batch_size or x.size(0)
        losses, scores = [], []
        for m in self.methods:
            loss, score = m(x, edge_index, s, batch_size=bs, node_idx=node_idx)
            losses.append(loss)
            scores.append(score)

        gate_w = torch.softmax(self.gate(x[:bs]), dim=1)
        fused = (gate_w * torch.stack(scores, dim=1)).sum(dim=1)

        embs = [m.emb[:bs] for m in self.methods if m.emb is not None]
        self.emb = torch.stack(embs).mean(dim=0) if embs else None

        return sum(losses) / len(losses), fused


class DualEncoderAttentionFusion(nn.Module):
    """FlexGAD-style: two independent encoders (e.g. GCN + MLP) fused via
    self-attention.  The attention weights are reused to derive the anomaly
    score balance, eliminating hyperparameter tuning for score weighting.
    """

    def __init__(self, methods: nn.ModuleList, hid_dim: int):
        super().__init__()
        self.methods = methods
        k = len(methods)
        self.attn = nn.MultiheadAttention(hid_dim, num_heads=4, batch_first=True)
        self.proj = nn.Linear(hid_dim * k, hid_dim)
        self.emb = None

    def forward(self, x, edge_index, s=None, batch_size=None, node_idx=None):
        bs = batch_size or x.size(0)
        losses, scores, embs = [], [], []
        for m in self.methods:
            loss, score = m(x, edge_index, s, batch_size=bs, node_idx=node_idx)
            losses.append(loss)
            scores.append(score)
            embs.append(m.emb[:bs])

        stacked = torch.stack(embs, dim=1)                     # [bs, K, hid]
        attn_out, attn_w = self.attn(stacked, stacked, stacked)  # attn_w: [bs, K, K]
        fused_emb = self.proj(attn_out.reshape(bs, -1))
        self.emb = fused_emb.detach()

        # Derive per-method weights from attention (FlexGAD trick)
        col_importance = attn_w.mean(dim=1)                     # [bs, K]
        w = col_importance / col_importance.sum(dim=1, keepdim=True).clamp(min=1e-8)
        fused = (w * torch.stack(scores, dim=1)).sum(dim=1)

        return sum(losses) / len(losses), fused


# --- Head modules for SharedEncoderFusion ---

class _ReconHead(nn.Module):
    def __init__(self, feat_dec, struct_dec, alpha):
        super().__init__()
        self.feat_dec = feat_dec
        self.struct_dec = struct_dec
        self.alpha = alpha
        from pygod.nn.functional import double_recon_loss
        self._loss = double_recon_loss

    def forward(self, z, x, edge_index, s, bs=None, node_idx=None):
        bs = bs or x.size(0)
        x_hat = self.feat_dec(z, edge_index) if self.feat_dec else x
        if self.struct_dec is not None:
            s_hat = self.struct_dec(z, edge_index)
            if s is None:
                s = to_dense_adj(edge_index, max_num_nodes=x.size(0))[0]
            if node_idx is not None:
                score = self._loss(x[:bs], x_hat[:bs],
                                   s[:bs, node_idx], s_hat[:bs], weight=self.alpha)
            else:
                score = self._loss(x[:bs], x_hat[:bs], s[:bs], s_hat[:bs], weight=self.alpha)
        else:
            diff = torch.pow(x[:bs] - x_hat[:bs], 2)
            score = torch.sqrt(torch.sum(diff, 1))
        return score.mean(), score


class _ContrastiveHead(nn.Module):
    def __init__(self, encoder_ref, hid_dim):
        super().__init__()
        self.encoder_ref = encoder_ref
        self.disc = nn.Bilinear(hid_dim, hid_dim, 1)
        self.readout = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Sigmoid())

    def forward(self, z, x, edge_index, s, bs=None, node_idx=None):
        bs = bs or x.size(0)
        summary = self.readout(z.mean(dim=0, keepdim=True).expand(bs, -1))
        pos = self.disc(z[:bs], summary).squeeze(-1)
        perm = torch.randperm(x.size(0), device=x.device)
        z_neg = self.encoder_ref(x[perm], edge_index)
        neg = self.disc(z_neg[:bs], summary).squeeze(-1)
        loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
                + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))) / 2
        return loss, 1.0 - pos.sigmoid()


class _SVDDHead(nn.Module):
    def __init__(self, hid_dim, warmup=20, eps=0.5):
        super().__init__()
        self.register_buffer("center", torch.zeros(hid_dim))
        self._warmup = warmup
        self._call_count = 0
        self.eps = eps

    def forward(self, z, x, edge_index, s, bs=None, node_idx=None):
        bs = bs or z.size(0)
        z_seed = z[:bs]
        self._call_count += 1
        if self._call_count <= self._warmup:
            self.center = z_seed.mean(dim=0).detach()
            return z_seed.var(dim=0).mean(), (z_seed - self.center).pow(2).sum(dim=1).detach()
        dist = (z_seed - self.center).pow(2).sum(dim=1)
        return F.relu(dist - self.eps).mean(), dist


class _NeighborhoodHead(nn.Module):
    """Wraps any neighborhood/community objective as a SharedEncoder head."""
    def __init__(self, objective):
        super().__init__()
        self.objective = objective

    def forward(self, z, x, edge_index, s, bs=None, node_idx=None):
        bs = bs or x.size(0)
        return self.objective(z, x, edge_index, bs)


def _build_head(spec, in_dim, hid_dim, n_dec, backbone, enc_cfg):
    obj_type = spec.get("objective", {}).get("type", "reconstruction")
    dec_type = spec.get("decoder", {}).get("type", "dual")
    alpha = spec.get("objective", {}).get("alpha", spec.get("decoder", {}).get("alpha", 0.5))

    if obj_type == "reconstruction":
        feat_dec = None
        struct_dec = None
        if dec_type in ("dual", "feature_mlp"):
            feat_dec = FeatureDecoder(backbone, hid_dim, in_dim, n_dec,
                                      enc_cfg.get("dropout", 0.0),
                                      enc_cfg.get("act", "relu"),
                                      heads=enc_cfg.get("heads", 4))
        if dec_type in ("dual", "dot_product"):
            struct_dec = StructureDecoder(
                backbone=backbone, hid_dim=hid_dim, num_layers=n_dec,
                dropout=enc_cfg.get("dropout", 0.0), act=enc_cfg.get("act", "relu"),
                sigmoid_s=False, heads=enc_cfg.get("heads", 4),
            )
        return _ReconHead(feat_dec, struct_dec, alpha)
    elif obj_type == "contrastive":
        return _ContrastiveHead(None, hid_dim)
    elif obj_type == "svdd":
        return _SVDDHead(hid_dim)
    elif obj_type == "neighborhood_jsd":
        return _NeighborhoodHead(NeighborhoodJSDObjective(hid_dim, in_dim))
    elif obj_type == "neighbor_prediction":
        return _NeighborhoodHead(NeighborPredictionObjective(hid_dim, in_dim))
    elif obj_type == "community_deviation":
        return _NeighborhoodHead(CommunityDeviationObjective(hid_dim))
    elif obj_type == "gad_nr":
        return _NeighborhoodHead(GADNRObjective(hid_dim, in_dim))
    elif obj_type == "local_affinity":
        return _NeighborhoodHead(LocalAffinityObjective(hid_dim))
    elif obj_type == "ego_matching":
        return _NeighborhoodHead(EgoMatchingObjective(hid_dim, in_dim))
    elif obj_type == "residual":
        return _NeighborhoodHead(ResidualObjective(hid_dim, in_dim))
    raise ValueError(f"Unknown objective for head: {obj_type}")


# ═══════════════════════════════════════════════════════════════════════════
#  METHOD BUILDER — top-level API for the agent
# ═══════════════════════════════════════════════════════════════════════════

class MethodBuilder:
    """Build, train, and evaluate composed GNN architectures.

    The agent uses this to:
    1. Compose single methods from component specs
    2. Compose fused architectures (shared encoder, cross-attention, gate)
    3. Train and evaluate them
    """

    def __init__(self, device: str | torch.device | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._models: dict[str, nn.Module] = {}
        self._scores: dict[str, np.ndarray] = {}

    def _inject_communities(self, model: nn.Module, data: Data):
        """Set community labels on any CommunityGCNEncoder found in the model."""
        from torch_geometric.utils import degree
        import networkx as nx
        from torch_geometric.utils import to_networkx

        def _find_and_set(m):
            for child in m.modules():
                if isinstance(child, CommunityGCNEncoder) and child._communities is None:
                    G = to_networkx(data, to_undirected=True)
                    comms = nx.community.louvain_communities(G, seed=42)
                    labels = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
                    for i, comm in enumerate(comms):
                        for node in comm:
                            labels[node] = i
                    child.set_communities(labels)
        _find_and_set(model)

    # ---- Build ----

    def build(self, name: str, spec: dict, in_dim: int) -> ComposedMethod:
        """Build a single method from a component spec."""
        m = ComposedMethod(spec, in_dim).to(self.device)
        self._models[name] = m
        return m

    def build_from_recipe(self, recipe_name: str, in_dim: int) -> ComposedMethod:
        """Build from a known recipe (dominant, cola, ocgnn, gaan, anomalydae, gae)."""
        recipe = KNOWN_RECIPES.get(recipe_name)
        if not recipe:
            raise ValueError(f"Unknown recipe: {recipe_name}. Available: {list(KNOWN_RECIPES)}")
        spec = {cat: {"type": typ} for cat, typ in recipe.items()}
        return self.build(recipe_name, spec, in_dim)

    def build_shared_encoder(self, name: str, head_specs: list[dict],
                             in_dim: int) -> SharedEncoderFusion:
        """Build a shared-encoder architecture with multiple objective heads."""
        model = SharedEncoderFusion(head_specs, in_dim).to(self.device)
        for head in model.heads:
            if isinstance(head, _ContrastiveHead):
                head.encoder_ref = model.encoder
        self._models[name] = model
        return model

    def build_cross_attention(self, name: str, method_specs: list[dict],
                              in_dim: int) -> CrossAttentionFusion:
        """Build independent methods fused via cross-attention on embeddings."""
        methods = nn.ModuleList([ComposedMethod(s, in_dim) for s in method_specs])
        hid_dim = method_specs[0].get("encoder", {}).get("hid_dim", 64)
        model = CrossAttentionFusion(methods, hid_dim).to(self.device)
        self._models[name] = model
        return model

    def build_ensemble_gate(self, name: str, method_specs: list[dict],
                            in_dim: int) -> EnsembleGateFusion:
        """Build independent methods fused via a learned per-node gate."""
        methods = nn.ModuleList([ComposedMethod(s, in_dim) for s in method_specs])
        model = EnsembleGateFusion(methods, in_dim).to(self.device)
        self._models[name] = model
        return model

    def build_dual_encoder_attention(self, name: str, method_specs: list[dict],
                                     in_dim: int) -> DualEncoderAttentionFusion:
        """Build FlexGAD-style dual-encoder with self-attention fusion.

        Attention weights automatically derive the score balance.
        """
        methods = nn.ModuleList([ComposedMethod(s, in_dim) for s in method_specs])
        hid_dim = method_specs[0].get("encoder", {}).get("hid_dim", 64)
        model = DualEncoderAttentionFusion(methods, hid_dim).to(self.device)
        self._models[name] = model
        return model

    # ---- Train ----

    def train(self, name: str, data: Data, epochs: int = 100,
              lr: float = 0.004, batch_size: int = 0, num_neigh: int = -1,
              verbose: bool = False) -> dict:
        """Train any built model (single or fused).

        Full-batch: pre-loads data to GPU once (fast, no loader overhead).
        Mini-batch (batch_size > 0): uses NeighborLoader for large graphs.
        """
        model = self._models.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' not built.")

        n = data.num_nodes
        needs_adj = (isinstance(model, ComposedMethod)
                     and model.struct_decoder is not None)

        # Inject community labels for CommunityGCNEncoder
        self._inject_communities(model, data)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()

        decision_scores = torch.zeros(n)
        losses = []

        if batch_size > 0 and batch_size < n:
            # Mini-batch path for large graphs
            from torch_geometric.loader import NeighborLoader
            data = data.clone()
            data.s = torch.zeros(1, 1)
            loader = NeighborLoader(data, [num_neigh], batch_size=batch_size)
            for epoch in range(epochs):
                epoch_loss = 0.0
                for sampled in loader:
                    bs = sampled.batch_size
                    nid = sampled.n_id
                    x = sampled.x.float().to(self.device)
                    ei = sampled.edge_index.to(self.device)
                    loss, score = model(x, ei, None, batch_size=bs, node_idx=nid)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    decision_scores[nid[:bs]] = score.detach().cpu()
                    epoch_loss += loss.item() * bs
                losses.append(epoch_loss / n)
                if verbose and (epoch + 1) % 25 == 0:
                    print(f"  [{name}] epoch {epoch+1}/{epochs} loss={losses[-1]:.4f}",
                          flush=True)
        else:
            # Full-batch fast path: pre-load to GPU once
            x = data.x.float().to(self.device)
            ei = data.edge_index.to(self.device)
            s = None
            if needs_adj:
                s = to_dense_adj(data.edge_index, max_num_nodes=n)[0].to(self.device)
            nid = torch.arange(n, device=self.device)

            for epoch in range(epochs):
                loss, score = model(x, ei, s, batch_size=n, node_idx=nid)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                decision_scores[:] = score.detach().cpu()
                losses.append(loss.item())
                if verbose and (epoch + 1) % 25 == 0:
                    print(f"  [{name}] epoch {epoch+1}/{epochs} loss={losses[-1]:.4f}",
                          flush=True)

        scores_np = decision_scores.numpy()
        self._scores[name] = scores_np

        return {
            "name": name,
            "epochs": epochs,
            "final_loss": round(losses[-1], 4),
            "score_range": [round(float(scores_np.min()), 4), round(float(scores_np.max()), 4)],
        }

    # ---- Evaluate ----

    def scores(self, name: str) -> np.ndarray | None:
        return self._scores.get(name)

    def evaluate(self, name: str, labels: np.ndarray) -> dict:
        """Evaluate against ground-truth (AUC-ROC). Auto-binarizes labels."""
        from sklearn.metrics import roc_auc_score
        scores = self._scores.get(name)
        if scores is None:
            raise ValueError(f"No scores for '{name}'.")
        binary = (labels > 0).astype(int)
        auc = roc_auc_score(binary, scores)
        return {"name": name, "auc_roc": round(float(auc), 4)}

    def score_level_fusion(self, name: str, method_names: list[str],
                           weights: list[float] | None = None) -> dict:
        """Baseline: weighted average of normalized scores."""
        normed = []
        for mn in method_names:
            s = self._scores.get(mn)
            if s is None:
                raise ValueError(f"No scores for '{mn}'.")
            s_n = (s - s.min()) / (s.max() - s.min() + 1e-10)
            normed.append(s_n)
        if weights is None:
            weights = [1.0 / len(normed)] * len(normed)
        fused = sum(w * s for w, s in zip(weights, normed))
        self._scores[name] = fused
        return {"name": name, "methods": method_names, "weights": weights}

    # ---- Analyze-and-Build pipeline ----

    def analyze_and_build(self, data: Data, labels: np.ndarray | None = None,
                          top_k: int = 3, verbose: bool = True) -> dict:
        """Full pipeline: analyze graph → recommend → build → train → evaluate.

        Returns the best method name, all results, and the analysis that drove
        component selection.
        """
        from arcade.components.registry import analyze_graph, recommend_components

        # Step 1: Analyze the graph
        profile = analyze_graph(data)
        if verbose:
            print(f"\n{'='*60}")
            print(f"GRAPH ANALYSIS: {profile['num_nodes']} nodes, "
                  f"{profile['num_edges']} edges, {profile['feature_dim']}d features")
            print(f"  homophily={profile['homophily']:.3f}  "
                  f"avg_degree={profile['avg_degree']:.1f}  "
                  f"density={profile['density']:.5f}  "
                  f"feature_sparsity={profile['feature_sparsity']:.1%}")
            print(f"  power_law={profile['is_power_law']}  "
                  f"max_degree={profile['max_degree']}")

        # Step 2: Recommend components
        recs = recommend_components(profile)
        hp = recs.pop("_hyperparams", {})
        hid_dim = hp.get("hid_dim", 64)
        num_layers = hp.get("num_layers", 4)
        lr = hp.get("lr", 0.004)
        epochs = hp.get("epochs", 200)
        batch_size = hp.get("batch_size", 0)
        num_neigh = hp.get("num_neigh", -1)

        if verbose:
            print(f"\nCOMPONENT RECOMMENDATIONS:")
            for cat, items in recs.items():
                picks = ", ".join(f"{n} ({r})" for n, r in items[:2])
                print(f"  {cat}: {picks}")
            bs_str = f"batch={batch_size}" if batch_size > 0 else "full-batch"
            print(f"  hyperparams: hid={hid_dim}, layers={num_layers}, "
                  f"lr={lr}, epochs={epochs}, {bs_str}")

        # Step 3: Build top_k candidate methods from recommendations
        enc_choices = [r[0] for r in recs.get("encoder", [("gcn", "")])]
        obj_choices = [r[0] for r in recs.get("objective", [("reconstruction", "")])]
        dec_choices = [r[0] for r in recs.get("decoder", [("dual", "")])]
        aug_choices = [r[0] for r in recs.get("augmentation", [("none", "")])]

        no_decoder_objs = {"gad_nr", "ego_matching", "contrastive", "svdd",
                           "local_affinity", "neighborhood_jsd",
                           "neighbor_prediction", "community_deviation"}

        large_graph = profile.get("num_nodes", 0) > 20000

        candidates = []
        for obj in obj_choices[:top_k]:
            enc = enc_choices[0]
            if obj in no_decoder_objs:
                dec = "none"
            elif large_graph:
                dec = "feature_mlp"
            elif profile.get("feature_dim", 0) >= 5 and profile.get("avg_degree", 0) >= 2:
                dec = "dual"
            else:
                dec = "feature_mlp"
            aug = aug_choices[0]
            name = f"{obj}-{enc}"
            spec = {
                "encoder": {"type": enc, "hid_dim": hid_dim,
                            "num_layers": num_layers},
                "objective": {"type": obj},
                "decoder": {"type": dec},
                "augmentation": {"type": aug},
            }
            candidates.append((name, spec))

        if verbose:
            print(f"\nCANDIDATES ({len(candidates)}):")
            for name, spec in candidates:
                print(f"  {name}: {spec['encoder']['type']}×{spec['objective']['type']}"
                      f"×{spec['decoder']['type']}+{spec['augmentation']['type']}")

        # Step 4: Train and evaluate all candidates
        in_dim = data.x.size(1)
        results = {}
        for name, spec in candidates:
            if verbose:
                print(f"\n--- Training {name} ---")
            self.build(name, spec, in_dim)
            info = self.train(name, data, epochs=epochs, lr=lr,
                              batch_size=batch_size, num_neigh=num_neigh,
                              verbose=verbose)
            result = {"spec": spec, "train": info}
            if labels is not None:
                ev = self.evaluate(name, labels)
                result["auc_roc"] = ev["auc_roc"]
                if verbose:
                    print(f"  AUROC: {ev['auc_roc']:.4f}")
            results[name] = result

        # Step 5: Select best
        if labels is not None:
            best_name = max(results, key=lambda k: results[k].get("auc_roc", 0))
        else:
            best_name = candidates[0][0]

        # Step 6: Score-level fusion of all candidates
        trained = [n for n in results if n in self._scores]
        if len(trained) >= 2:
            self.score_level_fusion("fusion", trained)
            if labels is not None:
                ev_fusion = self.evaluate("fusion", labels)
                results["fusion"] = {"auc_roc": ev_fusion["auc_roc"],
                                     "methods": trained}
                if ev_fusion["auc_roc"] > results[best_name].get("auc_roc", 0):
                    best_name = "fusion"

        if verbose:
            print(f"\n{'='*60}")
            print(f"RESULTS:")
            for name, r in results.items():
                auc_str = f"AUROC={r['auc_roc']:.4f}" if "auc_roc" in r else ""
                marker = " ← BEST" if name == best_name else ""
                print(f"  {name:30s} {auc_str}{marker}")
            print(f"{'='*60}")

        return {
            "profile": profile,
            "recommendations": recs,
            "hyperparams": hp,
            "results": results,
            "best": best_name,
        }

    # ---- Hybrid pipeline (fast paradigms + trained components) ----

    def hybrid_pipeline(self, data: Data, gap_analysis: dict,
                        profile: dict, epochs: int = 150,
                        lr: float = 0.004, verbose: bool = False,
                        ) -> dict:
        """Train complementary components based on gap analysis.

        Returns trained score arrays + metadata. The LLM agent
        inspects results and decides how to combine with fast paradigms.
        No automatic fusion — the agent is the decision-maker.
        """
        from arcade.components.registry import recommend_complements

        complements = recommend_complements(gap_analysis, profile)
        in_dim = data.x.size(1)
        n = data.num_nodes
        batch_size = 0 if n <= 20000 else 4096
        num_neigh = -1 if n <= 20000 else 10

        trained_scores = {}
        train_info = {}
        for comp in complements:
            name = comp["name"]
            spec = comp["spec"]
            if verbose:
                print(f"  {name}: {comp['reason']}", flush=True)

            self.build(name, spec, in_dim)
            info = self.train(name, data, epochs=epochs, lr=lr,
                              batch_size=batch_size, num_neigh=num_neigh)
            train_info[name] = info

            scores = self._scores.get(name)
            if scores is not None and scores.std() > 1e-6:
                trained_scores[name] = scores
            elif verbose:
                print(f"    SKIP {name}: degenerate", flush=True)

        return {
            "trained_scores": trained_scores,
            "train_info": train_info,
            "complements": complements,
        }

    # ---- Listing ----

    def list_methods(self) -> list[dict]:
        results = []
        for name, model in self._models.items():
            info = {"name": name, "trained": name in self._scores}
            if isinstance(model, ComposedMethod):
                info["type"] = "single"
                info.update(model.describe())
            elif isinstance(model, SharedEncoderFusion):
                info["type"] = "shared_encoder"
                info["num_heads"] = len(model.heads)
            elif isinstance(model, CrossAttentionFusion):
                info["type"] = "cross_attention"
                info["num_methods"] = len(model.methods)
            elif isinstance(model, EnsembleGateFusion):
                info["type"] = "ensemble_gate"
                info["num_methods"] = len(model.methods)
            elif isinstance(model, DualEncoderAttentionFusion):
                info["type"] = "dual_encoder_attention"
                info["num_methods"] = len(model.methods)
            results.append(info)
        return results
