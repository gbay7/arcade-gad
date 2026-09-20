"""Statistical-significance analysis of the 10-benchmark comparison
(Demsar, JMLR 2006), run on BOTH metrics: AUROC (Table II) and AP
(ap_table.json, same runs and protocol).

Compares ARCADE against the unsupervised baselines across the 10 benchmarks with
the standard protocol for "multiple methods over multiple datasets":

  1. Friedman test (with the Iman-Davenport F correction) on the per-dataset
     rank matrix: is there ANY significant difference among the methods?
  2. Nemenyi post-hoc (all pairs): two methods differ if their average ranks
     differ by more than the critical difference CD. Rendered as a CD diagram.
  3. Bonferroni-Dunn (ARCADE as the control): the more powerful test when one
     method is compared against all others.

AUROC means come straight from the shipped result artifacts (same numbers as
Table II). AutoGAD is excluded: its search selects by measured AUROC, so it is
label-guided, not an unsupervised baseline.

Missing cells ("/" in Table II: a runtime failure or an out-of-memory on our
12 GB GPU) are handled two ways, and we report both:
  - primary: all seven unsupervised baselines + ARCADE (k=8), a failed method
    taking the worst rank on the graph it could not run (it produced no usable
    output on our hardware);
  - robustness: complete-case on the six methods that ran on every graph (k=6),
    which imputes nothing.

Writes significance.json and cd_diagram.png next to this file.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2, f as f_dist, friedmanchisquare, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["Enron", "Reddit", "Books", "Disney", "Weibo",
            "Cora", "Amazon", "BlogCatalog", "ACM", "Flickr"]
BASELINES = ["DOMINANT", "AnomalyDAE", "DONE", "CoLA", "GAAN", "OCGNN", "Radar", "SL-GAD"]
METHODS = BASELINES + ["ARCADE"]

# Studentized-range critical values q_alpha (divided by sqrt(2)) for the Nemenyi
# test, and the two-tailed Bonferroni-Dunn critical values, both at alpha=0.05,
# indexed by the number of methods k (Demsar 2006, Tables 5 and 6).
Q_NEMENYI_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219}
Q_BONF_DUNN_05 = {2: 1.960, 3: 2.241, 4: 2.394, 5: 2.498, 6: 2.576,
                  7: 2.638, 8: 2.690, 9: 2.724, 10: 2.773, 11: 2.807}


def _fail(v):
    return v is None or (isinstance(v, dict) and "error" in v)


def load_matrix():
    """Assemble the 10x8 AUROC matrix from the result artifacts (np.nan = '/')."""
    pg = json.load(open(f"{HERE}/rebench_pygod.json"))
    suite = json.load(open(f"{HERE}/rebench_social_suite.json"))
    social = json.load(open(f"{HERE}/rebench_social.json"))
    slgad = json.load(open(f"{HERE}/rebench_slgad.json"))
    cov = json.load(open(f"{HERE}/coverage_bench.json"))

    social_key = {"BlogCatalog": "blogcatalog", "ACM": "acm", "Flickr": "cola_flickr"}
    cov_key = {"Enron": "enron", "Reddit": "reddit", "Books": "books",
               "Disney": "disney", "Weibo": "weibo", "Cora": "inj_cora",
               "Amazon": "inj_amazon", "BlogCatalog": "blogcatalog",
               "ACM": "acm", "Flickr": "cola_flickr"}

    radar = json.load(open(f"{HERE}/rebench_radar.json")) if os.path.exists(f"{HERE}/rebench_radar.json") else {}

    def cell(ds, m):
        if m == "ARCADE":
            return cov[cov_key[ds]]["auroc_mean"]
        if m == "Radar":                                # added after the first benchmark pass
            v = radar.get(ds)
            return np.nan if _fail(v) else v["mean"]
        if ds in pg:                                   # 7 standard datasets
            if m == "SL-GAD":
                return slgad[ds]["mean"]
            v = pg[ds].get(m)
            return np.nan if _fail(v) else v["mean"]
        sk = social_key[ds]                            # 3 social datasets
        if m == "SL-GAD":
            return social[f"SL-GAD:{sk}"]["mean"]
        if m == "CoLA":
            return social[f"CoLA:{sk}"]["mean"]
        v = suite.get(f"{sk}|{m}")
        return np.nan if _fail(v) else v["mean"]

    M = np.array([[cell(ds, m) for m in METHODS] for ds in DATASETS], dtype=float)
    return M


def load_matrix_ap():
    """Same 10x8 matrix for average precision, from the consolidated AP artifact
    (built from rebench_ap.json + rebench_social_suite.json + coverage_bench.json,
    identical runs and protocol as the AUROC artifacts)."""
    tab = json.load(open(f"{HERE}/ap_table.json"))
    radar = json.load(open(f"{HERE}/rebench_radar.json")) if os.path.exists(f"{HERE}/rebench_radar.json") else {}
    def ap_cell(ds, m):
        if m == "Radar":
            v = radar.get(ds); return np.nan if _fail(v) else v["ap_mean"]
        return np.nan if tab[ds].get(m) is None else tab[ds][m]
    M = np.array([[ap_cell(ds, m) for m in METHODS] for ds in DATASETS], dtype=float)
    return M


def rank_matrix(M, worst_rank_for_nan=True):
    """Per-dataset ranks (1 = best AUROC). NaN takes the worst average rank."""
    N, k = M.shape
    R = np.zeros((N, k))
    for i in range(N):
        row = M[i]
        ok = ~np.isnan(row)
        r = np.full(k, np.nan)
        # rank present methods 1..m by AUROC descending, average ties
        r[ok] = rankdata(-row[ok], method="average")
        if worst_rank_for_nan and (~ok).any():
            f = int((~ok).sum())
            # the f failures share ranks (k-f+1)..k
            r[~ok] = k - (f - 1) / 2.0
        R[i] = r
    return R


def friedman(R):
    N, k = R.shape
    r = R.mean(axis=0)
    chi2_F = (12 * N) / (k * (k + 1)) * (np.sum(r ** 2) - k * (k + 1) ** 2 / 4.0)
    df1 = k - 1
    p_chi2 = chi2.sf(chi2_F, df1)
    # Iman-Davenport F correction
    denom = N * (k - 1) - chi2_F
    F = (N - 1) * chi2_F / denom if denom > 0 else np.inf
    df2 = (k - 1) * (N - 1)
    p_F = f_dist.sf(F, df1, df2)
    return dict(avg_ranks=r.tolist(), chi2_F=chi2_F, p_chi2=p_chi2,
                iman_davenport_F=F, df=(df1, df2), p_F=p_F)


def cds(k, N):
    nem = Q_NEMENYI_05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
    bd = Q_BONF_DUNN_05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
    return nem, bd


def analyze(methods, M, tag):
    R = rank_matrix(M)
    fr = friedman(R)
    N, k = M.shape
    nem_cd, bd_cd = cds(k, N)
    ranks = dict(zip(methods, fr["avg_ranks"]))
    ctrl = ranks["ARCADE"]
    bonf = {m: {"rank_gap": round(ranks[m] - ctrl, 3),
                "significant_vs_ARCADE": bool(ranks[m] - ctrl > bd_cd)}
            for m in methods if m != "ARCADE"}
    # cross-check Friedman against scipy on complete columns
    return dict(tag=tag, methods=methods, N=N, k=k,
                avg_ranks={m: round(v, 3) for m, v in ranks.items()},
                friedman={kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in fr.items() if kk != "avg_ranks"},
                nemenyi_CD=round(nem_cd, 3), bonferroni_dunn_CD=round(bd_cd, 3),
                bonferroni_dunn_vs_control=bonf)


def cd_diagram(methods, avg_ranks, cd, path, title):
    """Classic Demsar critical-difference diagram (rank 1 = best, on the right)."""
    order = sorted(range(len(methods)), key=lambda i: avg_ranks[i])
    names = [methods[i] for i in order]
    ranks = [avg_ranks[i] for i in order]
    k = len(methods)
    lo, hi = 1, k
    n_right = (k + 1) // 2                       # best half branches right
    n_left = k - n_right
    row_gap = 0.7
    y_axis = 0.0
    y_first = -0.95                              # first (topmost) method label row
    y_lowest = y_first - (max(n_right, n_left) - 1) * row_gap
    y_cd = 1.15                                  # CD bar sits above the axis

    fig, ax = plt.subplots(figsize=(8.5, 1.9 + 0.5 * k))
    ax.set_xlim(hi + 0.7, lo - 0.7)             # reversed: best (low rank) on the right
    ax.set_ylim(y_lowest - 0.6, y_cd + 0.9)
    ax.axis("off")

    # rank axis with ticks
    ax.plot([lo, hi], [y_axis, y_axis], "k-", lw=1.4)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y_axis, y_axis + 0.11], "k-", lw=1.1)
        ax.text(t, y_axis + 0.26, str(t), ha="center", va="bottom", fontsize=10)

    # CD reference bar
    ax.plot([lo, lo + cd], [y_cd, y_cd], "k-", lw=2.5)
    for x in (lo, lo + cd):
        ax.plot([x, x], [y_cd - 0.07, y_cd + 0.07], "k-", lw=1.1)
    ax.text(lo + cd / 2, y_cd + 0.13, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=10)

    # method stems: best half to the right, worst half to the left
    for idx, (nm, rk) in enumerate(zip(names, ranks)):
        if idx < n_right:
            y = y_first - idx * row_gap
            ax.plot([rk, rk], [y_axis, y], "k-", lw=1.0)
            ax.plot([rk, lo - 0.5], [y, y], "k-", lw=1.0)
            ax.text(lo - 0.58, y, f"{nm} ({rk:.2f})", ha="right", va="center", fontsize=10)
        else:
            y = y_first - (idx - n_right) * row_gap
            ax.plot([rk, rk], [y_axis, y], "k-", lw=1.0)
            ax.plot([rk, hi + 0.5], [y, y], "k-", lw=1.0)
            ax.text(hi + 0.58, y, f"({rk:.2f}) {nm}", ha="left", va="center", fontsize=10)

    # cliques: maximal groups of methods all within CD of each other
    sr = sorted(ranks)
    groups = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and sr[j + 1] - sr[i] <= cd:
            j += 1
        if j > i:
            groups.append((sr[i], sr[j]))
        i += 1
    maximal = [g for g in groups
               if not any(g2 != g and g2[0] <= g[0] and g[1] <= g2[1] for g2 in groups)]
    for gi, (a, b) in enumerate(maximal):
        yb = y_axis - 0.16 - gi * 0.17
        ax.plot([a - 0.05, b + 0.05], [yb, yb], "k-", lw=4.0, solid_capstyle="round")

    fig.suptitle(title, fontsize=11, y=0.99)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_metric(metric, M, diagram_path):
    print(f"\n===== {metric.upper()} =====")
    print("           " + "  ".join(f"{m:>10s}" for m in METHODS))
    for ds, row in zip(DATASETS, M):
        print(f"{ds:11s} " + "  ".join(f"{('/' if np.isnan(v) else f'{v:.1f}'):>10s}" for v in row))

    primary = analyze(METHODS, M, f"primary_8methods_failworst_{metric}")

    # robustness: complete-case, drop methods with any '/'
    keep = [j for j, m in enumerate(METHODS) if not np.isnan(M[:, j]).any()]
    cc_methods = [METHODS[j] for j in keep]
    cc_M = M[:, keep]
    robust = analyze(cc_methods, cc_M, f"robustness_completecase_{metric}")

    # scipy cross-check of Friedman on the complete-case columns
    sp = friedmanchisquare(*[cc_M[:, j] for j in range(cc_M.shape[1])])
    robust["friedman"]["scipy_chi2"] = round(float(sp.statistic), 4)
    robust["friedman"]["scipy_p"] = float(sp.pvalue)

    cd_diagram(primary["methods"], list(primary["avg_ranks"].values()),
               primary["nemenyi_CD"], diagram_path,
               f"Nemenyi critical-difference diagram, {metric.upper()} "
               f"(10 benchmarks, alpha=0.05)")

    print(f"PRIMARY (k={primary['k']}, N={primary['N']}): "
          f"Friedman chi2_F={primary['friedman']['chi2_F']}, "
          f"Iman-Davenport F={primary['friedman']['iman_davenport_F']}, "
          f"p={primary['friedman']['p_F']:.2e}")
    print("avg ranks:", primary["avg_ranks"])
    print(f"Nemenyi CD={primary['nemenyi_CD']}, Bonferroni-Dunn CD={primary['bonferroni_dunn_CD']}")
    for m, d in primary["bonferroni_dunn_vs_control"].items():
        print(f"  ARCADE vs {m:11s} gap={d['rank_gap']:+.2f}  "
              f"{'SIGNIFICANT' if d['significant_vs_ARCADE'] else 'n.s.'}")
    print(f"ROBUSTNESS (k={robust['k']}): scipy Friedman p={robust['friedman']['scipy_p']:.2e}; "
          f"ARCADE rank={robust['avg_ranks']['ARCADE']}")
    return {"primary": primary, "robustness": robust}


def main():
    out = {"note": "Demsar 2006 protocol; AutoGAD excluded (label-guided). "
                   "AUROC means from the shipped artifacts (= Table II); "
                   "AP means from ap_table.json (same runs)."}
    out["auroc"] = run_metric("auroc", load_matrix(), f"{HERE}/cd_diagram.png")
    out["ap"] = run_metric("ap", load_matrix_ap(), f"{HERE}/cd_diagram_ap.png")
    json.dump(out, open(f"{HERE}/significance.json", "w"), indent=2)
    print("SIGNIFICANCE_DONE")


if __name__ == "__main__":
    main()
