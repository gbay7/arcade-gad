"""ARCADE Scoring Engine — SOTA-informed fusion.

Five improvements from ensemble AD literature:
1. SELECT-style pruning (Rayana & Akoglu 2016) — replaces consensus delta
2. AOM fusion (Aggarwal & Sathe 2015) — bias-variance optimal aggregation
3. UED-style quality (Idan+ 2024) — better unsupervised quality proxy
4. SEAD sparsity prior (ICML 2025) — penalize over-flagging detectors
5. Robust Gaussian Scaling (Rochner+ 2024) — better score normalization
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from typing import Any
from dataclasses import dataclass, field

from arcade.models import Weight, GraphProfile, Paradigm


class ScoringEngine:
    """ARCADE scoring: SELECT pruning → AOM/median/weighted fusion."""

    def __init__(self):
        self._score_matrix: dict[str, np.ndarray] = {}
        self._weights: dict[str, Weight] = {}
        self._paradigms: dict[str, Paradigm] = {}
        self._consensus_cache: np.ndarray | None = None
        self._fusion_mode: str = "fuse"
        self._pruning_info: dict[str, dict[str, Any]] = {}
        self._profile: dict[str, Any] | None = None
        self._degrees: np.ndarray | None = None

    def set_profile(self, profile: Any) -> None:
        """Attach the label-free graph profile so the fusion gate can route the
        contextual companion by degree regime (dense vs. sparse)."""
        if profile is None:
            self._profile = None
        elif isinstance(profile, dict):
            self._profile = profile
        else:
            self._profile = {k: getattr(profile, k) for k in dir(profile)
                             if not k.startswith("_") and not callable(getattr(profile, k))}

    def register_paradigm(self, paradigm: Paradigm) -> None:
        self._paradigms[paradigm.name] = paradigm

    def set_scores(self, paradigm_name: str, scores: np.ndarray) -> None:
        self._score_matrix[paradigm_name] = scores
        self._consensus_cache = None

    def get_scores(self, paradigm_name: str) -> np.ndarray | None:
        return self._score_matrix.get(paradigm_name)

    @property
    def available_paradigms(self) -> list[str]:
        return list(self._score_matrix.keys())

    @property
    def weights(self) -> dict[str, Weight]:
        return dict(self._weights)

    @property
    def fusion_mode(self) -> str:
        return self._fusion_mode

    # ==================================================================
    # Normalization — Robust Gaussian Scaling (Rochner+ SISAP 2024)
    # ==================================================================

    @staticmethod
    def robust_gaussian_scale(scores: np.ndarray) -> np.ndarray:
        """Map raw scores to [0,1] using robust center (median) and scale (nMAD)."""
        from scipy.special import erf
        med = np.median(scores)
        mad = np.median(np.abs(scores - med))
        nmad = 1.4826 * mad  # normalized MAD ≈ σ for Gaussian
        if nmad < 1e-10:
            return _percentile_rank(scores)
        z = (scores - med) / nmad
        return np.clip(0.5 * (1 + erf(z / np.sqrt(2))), 0.0, 1.0)

    # ==================================================================
    # I1: Quality — UED-style (Idan+ 2024)
    # ==================================================================

    def ued_quality(self, name: str) -> float:
        """UED-inspired quality: agreement with ensemble on tail, disagreement on body.

        High UED = paradigm agrees with others on who the outliers are
        but brings independent information on normal node ordering.
        """
        scores = self._score_matrix.get(name)
        if scores is None:
            return 0.0
        other_names = [o for o in self._score_matrix if o != name]
        if not other_names:
            return 0.5

        num_nodes = len(scores)
        my_ranks = sp_stats.rankdata(scores) / num_nodes
        other_ranks = np.stack([
            sp_stats.rankdata(self._score_matrix[o]) / num_nodes for o in other_names
        ])
        ensemble_ranks = np.median(other_ranks, axis=0)

        p90 = np.percentile(my_ranks, 90)
        tail_mask = my_ranks > p90
        body_mask = ~tail_mask

        tail_agreement = 0.0
        if tail_mask.sum() > 5:
            tail_rho = _spearman(my_ranks[tail_mask], ensemble_ranks[tail_mask])
            tail_agreement = max(0.0, tail_rho)

        body_diversity = 0.0
        if body_mask.sum() > 20:
            body_rho = abs(_spearman(my_ranks[body_mask], ensemble_ranks[body_mask]))
            body_diversity = 1.0 - body_rho

        return 0.6 * tail_agreement + 0.4 * body_diversity

    def compute_ued_all(self) -> dict[str, float]:
        return {name: self.ued_quality(name) for name in self._score_matrix}

    # ==================================================================
    # I1b: Tail Separation (kept for backward compat)
    # ==================================================================

    @staticmethod
    def tail_separation(scores: np.ndarray) -> float:
        if len(scores) < 20:
            return 0.0
        p90 = np.percentile(scores, 90)
        tail = scores[scores > p90]
        body = scores[scores <= p90]
        if len(tail) < 2 or len(body) < 2:
            return 0.0

        body_std = max(float(np.std(body)), 1e-10)
        separation = float(np.mean(tail) - np.mean(body)) / body_std
        ks_stat, _ = sp_stats.ks_2samp(tail, body)
        tail_cv = float(np.std(tail)) / max(float(np.mean(tail)), 1e-10)
        coherence = 1.0 / (1.0 + tail_cv)

        raw = 0.4 * min(separation / 5.0, 1.0) + 0.4 * ks_stat + 0.2 * coherence
        return float(np.clip(raw, 0.0, 1.0))

    def compute_tail_separation_all(self) -> dict[str, float]:
        return {name: self.tail_separation(scores) for name, scores in self._score_matrix.items()}

    # ==================================================================
    # Gap Analysis — unsupervised diagnostics for hybrid pipeline
    # ==================================================================

    def diagnose_gaps(self) -> dict[str, Any]:
        """Analyze fast paradigm scores WITHOUT labels.

        Returns diagnostics the LLM agent uses to decide which trained
        components to add:
        - tail_agreement: pairwise Spearman on P90+ tails
        - bimodality: per-paradigm (skew^2+1)/kurtosis (>0.555 = bimodal)
        - signal_count: how many paradigms have UED > 0.3
        - coverage_clusters: correlation-based grouping (redundancy detection)
        """
        names = list(self._score_matrix.keys())
        if len(names) < 2:
            return {"signal_count": len(names), "tail_agreement": {},
                    "bimodality": {}, "coverage_clusters": [names]}

        n = len(self._score_matrix[names[0]])
        p90_idx = int(0.9 * n)

        # Tail agreement
        ranks = {nm: _percentile_rank(sc) for nm, sc in self._score_matrix.items()}
        tail_agr = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                order = ranks[a].argsort()[::-1][:n - p90_idx]
                rho = _spearman(ranks[a][order], ranks[b][order])
                tail_agr[f"{a}|{b}"] = round(rho, 3)

        # Bimodality coefficient per paradigm
        bimod = {}
        for nm, sc in self._score_matrix.items():
            kurt = float(sp_stats.kurtosis(sc, fisher=False))
            skew = float(sp_stats.skew(sc))
            bc = (skew ** 2 + 1) / max(kurt, 1e-8)
            bimod[nm] = round(bc, 4)

        # Signal count (UED > 0.3)
        ued = self.compute_ued_all()
        sig_count = sum(1 for v in ued.values() if v > 0.3)

        # Coverage clusters via correlation
        if len(names) >= 3:
            corr = np.zeros((len(names), len(names)))
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    corr[i, j] = abs(_spearman(ranks[a], ranks[b]))
            clusters = []
            assigned = set()
            for i in range(len(names)):
                if i in assigned:
                    continue
                cluster = [names[i]]
                assigned.add(i)
                for j in range(i + 1, len(names)):
                    if j not in assigned and corr[i, j] > 0.6:
                        cluster.append(names[j])
                        assigned.add(j)
                clusters.append(cluster)
        else:
            clusters = [names]

        return {
            "signal_count": sig_count,
            "ued_scores": {k: round(v, 3) for k, v in ued.items()},
            "tail_agreement": tail_agr,
            "bimodality": bimod,
            "coverage_clusters": clusters,
            "num_paradigms": len(names),
        }

    # ==================================================================
    # I2: SELECT-style Pruning (Rayana & Akoglu 2016)
    # ==================================================================

    def select_prune(self) -> dict[str, dict[str, Any]]:
        """DROP anti-signal detectors using SELECT's pseudo-GT approach.

        1. Compute pseudo ground truth = median of all rank-normalized scores
        2. Correlate each detector against pseudo GT
        3. Drop detectors with ρ < 0 (anti-signals)
        4. Also apply SEAD sparsity prior
        """
        names = list(self._score_matrix.keys())
        if len(names) < 3:
            self._pruning_info = {
                n: {"rho_pgt": 1.0, "sparsity": 1.0, "delta": 1.0, "status": "keep"}
                for n in names
            }
            return self._pruning_info

        n = len(next(iter(self._score_matrix.values())))
        rank_matrix = np.stack([
            sp_stats.rankdata(self._score_matrix[name]) / n for name in names
        ])
        pseudo_gt = np.median(rank_matrix, axis=0)

        results = {}
        for i, name in enumerate(names):
            scores = self._score_matrix[name]

            if np.std(scores) < 1e-10:
                rho_pgt = 0.0
            else:
                rho_pgt, _ = sp_stats.spearmanr(rank_matrix[i], pseudo_gt)
                rho_pgt = 0.0 if np.isnan(rho_pgt) else float(rho_pgt)

            # SEAD sparsity prior: fraction of nodes above P90
            p90 = np.percentile(scores, 90)
            tail_frac = float((scores > p90).sum()) / n
            # Expected: 10%. Penalize if much more.
            sparsity = np.exp(-3.0 * max(0.0, tail_frac - 0.10))

            # UED quality
            ued = self.ued_quality(name)

            # Agent-created paradigms bypass SELECT pruning — they're
            # specifically chosen to complement fast paradigms, not agree
            paradigm = self._paradigms.get(name)
            is_trained = paradigm and getattr(paradigm, 'agent_created', False)

            if is_trained:
                delta = 1.0
                status = "keep_trained"
            elif rho_pgt < -0.05:
                delta = 0.0
                status = "exclude_anti_signal"
            elif rho_pgt < 0.05 and ued < 0.2:
                delta = 0.0
                status = "exclude_low_quality"
            else:
                delta = 1.0
                status = "keep"

            results[name] = {
                "rho_pgt": round(rho_pgt, 4),
                "sparsity": round(sparsity, 4),
                "ued_quality": round(ued, 4),
                "delta": delta,
                "status": status,
            }

        self._pruning_info = results
        return results

    # Backward-compatible alias
    def compute_delta(self, tau: float = 0.25) -> dict[str, dict[str, Any]]:
        return self.select_prune()

    # ==================================================================
    # I3: Mechanism vs. Artifact Test — m_k (unchanged)
    # ==================================================================

    def mechanism_test(
        self,
        degrees: np.ndarray,
        consistency_threshold: float = 0.6,
        residual_drop_threshold: float = 0.25,
    ) -> dict[str, Any]:
        names = list(self._score_matrix.keys())
        if not names:
            return {"decision": "artifact", "m_k": 1, "reason": "no scores"}

        deg_corrs = {}
        for name in names:
            s = self._score_matrix[name]
            if np.std(s) < 1e-10:
                deg_corrs[name] = 0.0
            else:
                rho, _ = sp_stats.spearmanr(s, degrees)
                deg_corrs[name] = 0.0 if np.isnan(rho) else float(rho)

        signs = [1 if r > 0.1 else (-1 if r < -0.1 else 0) for r in deg_corrs.values()]
        nonzero = [s for s in signs if s != 0]
        if not nonzero:
            return {
                "decision": "artifact",
                "m_k": 1,
                "reason": "no paradigm shows degree correlation",
                "per_paradigm_deg_corr": deg_corrs,
            }

        from collections import Counter
        sign_counts = Counter(nonzero)
        dominant_sign = sign_counts.most_common(1)[0]
        consistency = dominant_sign[1] / len(nonzero)

        stage1 = "mechanism" if consistency >= consistency_threshold else "artifact"

        residual_agreement = None
        if stage1 == "mechanism":
            from sklearn.linear_model import LinearRegression
            consensus = self._compute_consensus()
            raw_rho, _ = sp_stats.spearmanr(
                np.column_stack([self._score_matrix[n] for n in names]).mean(axis=1),
                consensus
            )
            deg_reshaped = degrees.reshape(-1, 1)
            residuals = {}
            for name in names:
                lr = LinearRegression().fit(deg_reshaped, self._score_matrix[name])
                residuals[name] = self._score_matrix[name] - lr.predict(deg_reshaped)

            resid_ranks = np.zeros((len(names), len(degrees)))
            for i, name in enumerate(names):
                resid_ranks[i] = sp_stats.rankdata(residuals[name]) / len(degrees)
            resid_consensus = np.median(resid_ranks, axis=0)
            resid_rho, _ = sp_stats.spearmanr(
                np.column_stack([residuals[n] for n in names]).mean(axis=1),
                resid_consensus
            )
            residual_agreement = float(resid_rho)
            relative_drop = 1.0 - residual_agreement / max(float(raw_rho), 1e-8)

            if relative_drop > residual_drop_threshold:
                stage2 = "mechanism"
            else:
                stage2 = "artifact"
        else:
            stage2 = "artifact"

        final = "mechanism" if stage1 == "mechanism" and stage2 == "mechanism" else "artifact"
        m_k = 0 if final == "mechanism" else 1

        return {
            "decision": final,
            "m_k": m_k,
            "stage1_consistency": round(consistency, 3),
            "stage1_verdict": stage1,
            "stage2_residual_agreement": round(residual_agreement, 3) if residual_agreement is not None else None,
            "stage2_verdict": stage2,
            "per_paradigm_deg_corr": {k: round(v, 4) for k, v in deg_corrs.items()},
            "reason": f"stage1={stage1} (consistency={consistency:.2f}), stage2={stage2}",
        }

    # ==================================================================
    # I4: Adaptive Fusion Gate (v3: uses UED spread)
    # ==================================================================

    _CLOSED_FORM = ("conformity", "linear_residual", "peripheral_mismatch",
                    "attribute_inflation", "clique_density")

    def compute_fusion_gate(self) -> dict[str, Any]:
        names = list(self._score_matrix.keys())
        n_p = len(names)
        if n_p <= 1:
            self._fusion_mode = "select"
            return {"mode": "select", "reason": "single paradigm", "D": 0, "Q": 1}

        # Trigger-trust: a closed-form detector runs only when its label-free
        # mechanism trigger fired, so if one is present we trust it rather than let
        # the UED gate average a dominant-but-uncorrelated signal away. Prune to the
        # triggered detector(s), adding a contextual companion when clique_density
        # fired (injected graphs carry two populations: structural + contextual).
        # The companion is chosen by the DEGREE REGIME, not hardcoded: on sparse
        # citation graphs local affinity separates the contextual population, but on
        # dense community graphs the low-affinity prior inverts (every node sits in a
        # coherent neighbourhood), so the subgraph-contrastive view is used instead.
        cf = [x for x in names if x in self._CLOSED_FORM]
        degraded_reason = None
        if cf:
            trusted = set(cf)
            companion_failed = False
            if "clique_density" in cf:
                # The structural view needs a contextual companion (two anomaly
                # populations can coexist), chosen by degree regime — but the
                # candidate must pass a label-free sanity check: a companion whose
                # score is essentially an echo of node degree carries no contextual
                # signal (it was the failure mode on dense community graphs).
                avg_deg = float((self._profile or {}).get("avg_degree", 0.0) or 0.0)
                dense = avg_deg >= 8.0
                order = (["subgraph_contrast", "contrastive", "local_affinity"] if dense
                         else ["local_affinity", "subgraph_contrast", "contrastive"])
                companion = None
                for cand in order:
                    if cand not in names:
                        continue
                    if self._degrees is not None:
                        dc = abs(_spearman(self._score_matrix[cand], self._degrees))
                        if dc >= 0.6:      # degree artifact — not a contextual view
                            continue
                    # A rank-max union amplifies whatever the companion puts in
                    # its top ranks; a companion with NO separated tail of its
                    # own contributes no retrievable population — only noise.
                    if self.tail_separation(self._score_matrix[cand]) < 0.3:
                        continue
                    companion = cand
                    break
                if companion is not None:
                    trusted.add(companion)
                else:
                    # A companion was needed but none passed: the trusted-pair
                    # premise fails. Degrade to the standard label-free gate over
                    # everything rather than trusting a blind or artifact pair.
                    companion_failed = True
                    degraded_reason = ("clique_density fired but no companion passed the "
                                       "degree-artifact check — standard gate over all detectors")
            if not companion_failed:
                for x in names:
                    w = self._weights.get(x)
                    if w is None:
                        w = Weight(alpha=1.0, beta=1.0, gamma=1.0)
                        self._weights[x] = w
                    w.delta = 1.0 if x in trusted else 0.0
                # A trusted structural+contextual PAIR is fused by per-node rank
                # maximum: the two views retrieve different anomaly populations,
                # and an average dilutes each (a union retrieves both).
                if len(trusted) == 1:
                    self._fusion_mode = "select"
                elif len(trusted) == 2 and "clique_density" in trusted:
                    self._fusion_mode = "pairmax"
                else:
                    self._fusion_mode = "aom"
                return {"mode": self._fusion_mode, "D": None, "Q": None,
                        "trusted": sorted(trusted),
                        "reason": f"closed-form trigger(s) fired ({', '.join(cf)}) — trust the mechanism detector(s)"}

        # Diversity: mean pairwise |ρ|
        corr_sum = 0.0
        pairs = 0
        for i in range(n_p):
            for j in range(i + 1, n_p):
                rho = abs(_spearman(self._score_matrix[names[i]], self._score_matrix[names[j]]))
                corr_sum += rho
                pairs += 1
        D = 1.0 - corr_sum / max(pairs, 1)

        # Quality spread: UED-based
        ueds = self.compute_ued_all()
        ued_vals = sorted(ueds.values())
        max_ued = max(ued_vals) if ued_vals else 0
        # true median (the old index formula returned the MAX for 2-detector
        # matrices, making the spread statistic blind exactly when the
        # fuse-vs-select choice matters most)
        median_ued = float(np.median(ued_vals)) if ued_vals else 1e-8
        Q_ued = max_ued / max(median_ued, 0.01)

        # Also compute tail_separation spread for backward compat
        ems = {name: self.tail_separation(self._score_matrix[name]) for name in names}

        # Dominant-select: when the SAME detector tops both label-free quality
        # diagnostics (UED and tail separation) and its UED is at least twice the
        # runner-up's, averaging can only dilute it — select it. (The observed
        # failure: a strong deep view fused with a weak fast one loses the
        # separation the deep view had alone.)
        if len(names) >= 2:
            ued_rank = sorted(names, key=lambda k: ueds.get(k, 0.0), reverse=True)
            top, second = ued_rank[0], ued_rank[1]
            top_ued, second_ued = ueds.get(top, 0.0), ueds.get(second, 0.0)
            if (top_ued >= 2.0 * max(second_ued, 1e-8)
                    and ems.get(top, 0.0) >= max(ems.get(n2, 0.0) for n2 in names if n2 != top)):
                for x in names:
                    w = self._weights.get(x)
                    if w is None:
                        w = Weight(alpha=1.0, beta=1.0, gamma=1.0)
                        self._weights[x] = w
                    w.delta = 1.0 if x == top else 0.0
                self._fusion_mode = "select"
                reason = (f"'{top}' dominates both quality diagnostics "
                          f"(UED {top_ued:.2f} vs {second_ued:.2f}) — select it")
                if degraded_reason:
                    reason = degraded_reason + "; " + reason
                return {"mode": "select", "D": round(D, 3), "Q_ued": round(top_ued / max(second_ued, 0.01), 2),
                        "trusted": [top], "reason": reason}

        # Check if trained components dominate tail separation
        trained_names = [n for n in names
                         if getattr(self._paradigms.get(n), 'agent_created', False)]
        fast_names_here = [n for n in names if n not in trained_names]
        trained_ts = [ems[n] for n in trained_names] if trained_names else []
        fast_ts = [ems[n] for n in fast_names_here] if fast_names_here else []
        trained_dominates = (trained_ts and fast_ts and
                             max(trained_ts) > 1.5 * max(fast_ts))

        # Fusion gate decision
        if trained_dominates:
            mode = "fuse"
            reason = (f"trained components dominate tail separation "
                      f"({max(trained_ts):.2f} vs {max(fast_ts):.2f})")
        elif Q_ued > 2.5:
            mode = "select"
            reason = f"dominant paradigm (Q_ued={Q_ued:.2f})"
        elif D > 0.4 and Q_ued < 2:
            # Diverse detectors with no dominant one: AOM (average of group
            # maxima) is the robust ensemble here — weight-averaged fusion in
            # this regime lets correlated mediocre members drown the signal.
            mode = "aom"
            reason = f"diverse (D={D:.2f}), no dominant paradigm (Q_ued={Q_ued:.2f}) — AOM"
        elif D < 0.3:
            mode = "select"
            reason = f"low diversity (D={D:.2f}) — paradigms redundant"
        else:
            mode = "hybrid"
            reason = f"moderate diversity (D={D:.2f}), moderate spread (Q_ued={Q_ued:.1f})"

        self._fusion_mode = mode
        return {
            "mode": mode,
            "D": round(D, 3),
            "Q_ued": round(Q_ued, 2),
            "reason": reason,
            "ued_quality": {k: round(v, 4) for k, v in ueds.items()},
            "tail_separation": {k: round(v, 4) for k, v in ems.items()},
        }

    # ==================================================================
    # Alpha: relevance
    # ==================================================================

    def compute_alpha(self, profile: GraphProfile) -> dict[str, float]:
        alphas = {}
        for name, paradigm in self._paradigms.items():
            alphas[name] = 1.0 if paradigm.is_applicable(profile) else 0.0
        return alphas

    # ==================================================================
    # Beta v3: UED * sparsity * (1-|ρ_deg|)^m_k * (1-|ρ_com|)
    # ==================================================================

    def compute_beta(
        self,
        degrees: np.ndarray,
        communities: np.ndarray,
        m_k: int = 1,
    ) -> dict[str, float]:
        betas = {}
        ueds = self.compute_ued_all()
        for name, scores in self._score_matrix.items():
            rho_deg = abs(_spearman(scores, degrees))
            comm_sizes = np.bincount(communities)
            node_comm_sizes = comm_sizes[communities].astype(float)
            rho_com = abs(_spearman(scores, node_comm_sizes))

            ued = ueds.get(name, 0.0)
            sparsity = self._pruning_info.get(name, {}).get("sparsity", 1.0)

            paradigm = self._paradigms.get(name)
            is_trained = paradigm and getattr(paradigm, 'agent_created', False)

            beta = ued * sparsity * ((1 - rho_deg) ** m_k) * (1 - rho_com)
            if is_trained:
                ts = self.tail_separation(scores)
                beta = max(beta, 0.3 * ts)
            betas[name] = max(0.0, min(1.0, beta))
        return betas

    # ==================================================================
    # Gamma: independence
    # ==================================================================

    def compute_gamma(self) -> dict[str, float]:
        names = list(self._score_matrix.keys())
        n = len(names)
        if n <= 1:
            return {name: 1.0 for name in names}

        corr_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                corr_matrix[i, j] = abs(_spearman(
                    self._score_matrix[names[i]],
                    self._score_matrix[names[j]],
                ))

        gammas = {}
        for i, name in enumerate(names):
            row_sum = corr_matrix[i].sum()
            gammas[name] = 1.0 / max(row_sum, 1e-8)
        return gammas

    def paradigm_correlation_matrix(self) -> dict[str, Any]:
        names = list(self._score_matrix.keys())
        n = len(names)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = _spearman(
                    self._score_matrix[names[i]],
                    self._score_matrix[names[j]],
                )
        return {"paradigms": names, "correlation_matrix": matrix.tolist()}

    # ==================================================================
    # Consensus (for mechanism test compatibility)
    # ==================================================================

    def _compute_consensus(self) -> np.ndarray:
        if self._consensus_cache is not None:
            return self._consensus_cache
        names = list(self._score_matrix.keys())
        if not names:
            return np.array([])
        n = len(next(iter(self._score_matrix.values())))
        rank_matrix = np.zeros((len(names), n))
        for i, name in enumerate(names):
            rank_matrix[i] = sp_stats.rankdata(self._score_matrix[name]) / n
        self._consensus_cache = np.median(rank_matrix, axis=0)
        return self._consensus_cache

    def _consensus_reliability(self) -> float:
        names = list(self._score_matrix.keys())
        if len(names) < 2:
            return 0.0
        total = 0.0
        pairs = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                total += abs(_spearman(self._score_matrix[names[i]], self._score_matrix[names[j]]))
                pairs += 1
        return total / max(pairs, 1)

    # ==================================================================
    # Compute all v3 weights
    # ==================================================================

    def compute_weights(
        self,
        profile: GraphProfile,
        degrees: np.ndarray,
        communities: np.ndarray,
        tau: float = 0.25,
    ) -> dict[str, Weight]:
        alphas = self.compute_alpha(profile)

        # kept for the fusion gate's label-free companion sanity checks
        self._degrees = np.asarray(degrees, dtype=float)

        mech = self.mechanism_test(degrees)
        m_k = mech["m_k"]

        # SELECT pruning (replaces consensus delta)
        pruning = self.select_prune()

        betas = self.compute_beta(degrees, communities, m_k=m_k)
        gammas = self.compute_gamma()
        gate = self.compute_fusion_gate()
        ems = self.compute_tail_separation_all()

        self._weights = {}
        for name in self._score_matrix:
            prune_info = pruning.get(name, {"delta": 1.0})
            delta_val = prune_info["delta"]
            self._weights[name] = Weight(
                alpha=alphas.get(name, 1.0),
                beta=betas.get(name, 1.0),
                gamma=gammas.get(name, 1.0),
                delta=delta_val,
                excess_mass=ems.get(name, 0.0),
                mechanism_flag=m_k,
                fusion_mode=gate["mode"],
            )
        return dict(self._weights)

    # ==================================================================
    # Override
    # ==================================================================

    def override_weight(self, paradigm: str, factor: str, value: float, reason: str) -> dict[str, Any]:
        if paradigm not in self._weights:
            return {"error": f"No weights computed for paradigm '{paradigm}'"}

        w = self._weights[paradigm]
        valid_factors = ("alpha", "beta", "gamma", "delta")
        if factor not in valid_factors:
            return {"error": f"Unknown factor '{factor}'. Use: {valid_factors}"}

        old_value = getattr(w, factor)
        setattr(w, factor, value)
        w.overrides.append({
            "factor": factor,
            "old_value": old_value,
            "new_value": value,
            "reason": reason,
        })
        return {
            "paradigm": paradigm,
            "factor": factor,
            "old": old_value,
            "new": value,
            "reason": reason,
            "new_w": w.w,
        }

    # ==================================================================
    # Fusion — v3: AOM, median, weighted, select, hybrid
    # ==================================================================

    def _get_active(self, selected_only: bool = True) -> dict[str, tuple[np.ndarray, float]]:
        """Collect paradigms that survived pruning with their weights."""
        active = {}
        for name, scores in self._score_matrix.items():
            w = self._weights.get(name)
            if w is None:
                active[name] = (scores, 1.0)
                continue
            if selected_only and w.alpha == 0:
                continue
            if w.delta == 0:
                continue
            active[name] = (scores, abs(w.w))
        return active

    def fuse(self, selected_only: bool = True) -> np.ndarray:
        if not self._score_matrix:
            raise ValueError("No scores computed yet")

        ref = next(iter(self._score_matrix.values()))
        n = len(ref)

        # AOM: Average of Maximums (Aggarwal & Sathe 2015)
        if self._fusion_mode == "aom":
            return self._fuse_aom(n, selected_only)

        active = self._get_active(selected_only)
        if not active:
            return np.zeros(n)

        if self._fusion_mode == "select":
            best_name = max(active, key=lambda k: active[k][1])
            return _percentile_rank(active[best_name][0])

        if self._fusion_mode == "pairmax":
            # Per-node rank maximum over the trusted pair: the structural and
            # contextual views retrieve DIFFERENT anomaly populations; the max
            # unions them where a mean would dilute each.
            rank_list = [_percentile_rank(scores) for scores, _ in active.values()]
            return np.max(np.stack(rank_list), axis=0)

        if self._fusion_mode == "hybrid":
            top3 = sorted(active, key=lambda k: active[k][1], reverse=True)[:3]
            rank_list = [_percentile_rank(active[name][0]) for name in top3]
            return np.median(np.stack(rank_list), axis=0)

        # Fuse mode: weighted fusion
        weighted_sum = np.zeros(n)
        weight_sum = 0.0
        for name, (scores, wk) in active.items():
            ranks = _percentile_rank(scores)
            weighted_sum += wk * ranks
            weight_sum += wk

        if weight_sum < 1e-8:
            return np.zeros(n)
        return weighted_sum / weight_sum

    def _fuse_aom(self, n: int, selected_only: bool = True, n_partitions: int = 10) -> np.ndarray:
        """Average of Maximums: partition → max per group → average across groups.

        Randomly partitions active paradigms into groups.
        Within each group: take the per-node maximum rank (reduces bias).
        Across groups: average the maxima (reduces variance).
        """
        active = self._get_active(selected_only)
        if not active:
            return np.zeros(n)

        # canonical order: the seeded partition must not depend on score-matrix
        # insertion order, or identical selections fuse differently per caller
        names = sorted(active.keys())
        rank_matrix = {name: _percentile_rank(active[name][0]) for name in names}

        if len(names) <= 2:
            ranks = np.stack(list(rank_matrix.values()))
            return np.median(ranks, axis=0)

        rng = np.random.RandomState(42)
        group_maxima = []
        for _ in range(n_partitions):
            rng.shuffle(names)
            group_size = max(2, len(names) // 3)
            for start in range(0, len(names), group_size):
                group = names[start:start + group_size]
                if not group:
                    continue
                group_ranks = np.stack([rank_matrix[name] for name in group])
                group_maxima.append(np.max(group_ranks, axis=0))

        return np.mean(np.stack(group_maxima), axis=0)

    def top_k(self, k: int = 50) -> list[dict[str, Any]]:
        fused = self.fuse()
        indices = np.argsort(fused)[::-1][:k]
        results = []
        for idx in indices:
            per_paradigm = {}
            for name, scores in self._score_matrix.items():
                per_paradigm[name] = float(scores[idx])
            results.append({
                "node_id": int(idx),
                "fused_score": float(fused[idx]),
                "per_paradigm": per_paradigm,
            })
        return results

    def score_summary(self) -> dict[str, Any]:
        fused = self.fuse()
        return {
            "num_paradigms": len(self._score_matrix),
            "paradigms": list(self._score_matrix.keys()),
            "fusion_mode": self._fusion_mode,
            "weights": {
                name: {
                    "alpha": w.alpha, "beta": round(w.beta, 3),
                    "gamma": round(w.gamma, 3), "delta": w.delta,
                    "excess_mass": round(w.excess_mass, 3),
                    "mechanism_flag": w.mechanism_flag,
                    "w": round(w.w, 3),
                }
                for name, w in self._weights.items()
            },
            "pruning": {
                name: info.get("status", "unknown")
                for name, info in self._pruning_info.items()
            },
            "fused_stats": {
                "mean": float(np.mean(fused)),
                "std": float(np.std(fused)),
                "p90": float(np.percentile(fused, 90)),
                "p95": float(np.percentile(fused, 95)),
                "p99": float(np.percentile(fused, 99)),
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    rho, _ = sp_stats.spearmanr(a, b)
    return 0.0 if np.isnan(rho) else float(rho)


def _percentile_rank(scores: np.ndarray) -> np.ndarray:
    ranks = sp_stats.rankdata(scores, method="average")
    return ranks / len(ranks)
