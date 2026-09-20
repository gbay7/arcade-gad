"""Fusion test bench: replay every stored analyst decision through the shipped
gate and through candidate label-free fusion rules, on every graph at once.

A rule sees only the selected paradigms' cached score vectors (and, for one
variant, the analyst's stated order). Labels are used once per row, to measure,
and for two diagnostics that are reported but never given to a rule: the best
selected view (the ceiling any fusion could reach without changing the
selection) and whether the selection contains a below-chance view.

    python experiments/fusion_rules.py            # all graphs
    python experiments/fusion_rules.py tolokers   # one graph
"""
import os, sys, json, glob, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.llm_ablation.cache import DatasetReplayer
from arcade.engine.scoring import ScoringEngine

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(HERE, "llm_ablation")
HELDOUT = {"tolokers", "amazon_fraud", "questions"}
FILE_ARM = {"results_heldout_probeall.jsonl": "probeall", "results_heldout_domain.jsonl": "domain",
            "results_heldout_rich.jsonl": "rich", "results_heldout_rich_domain.jsonl": "rich+domain",
            "results_study_rich.jsonl": "rich", "results_study_rich_v2.jsonl": "rich",
            "results_study_rich_keepdesc.jsonl": "rich+desc", "results_traced_rich.jsonl": "rich"}

def decisions():
    seen, out = set(), []
    for p in sorted(glob.glob(os.path.join(ABL, "results*.jsonl"))):
        f = os.path.basename(p)
        if f in ("results_packv2.jsonl", "results_packv3.jsonl", "results_checklist.jsonl",
                 "results_thinking.jsonl"):
            continue                      # superseded pack versions / other ablations
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "llm" or not r.get("selected"):
                continue
            t = r.get("temperature")
            if not (t in (0, 0.0) or (t is None and r["model"].startswith(("claude:", "opencode:")))):
                continue
            arm = r.get("arm") or FILE_ARM.get(f, "study")
            key = (arm, r["model"], r["dataset"], r.get("repeat", 0))
            if key in seen:
                continue
            seen.add(key)
            out.append({"arm": arm, "model": r["model"], "dataset": r["dataset"],
                        "repeat": r.get("repeat", 0), "selected": r["selected"]})
    return out

def ranks(v):
    """rank-normalised to [0,1]; fusion on ranks makes views comparable."""
    order = np.argsort(np.argsort(np.asarray(v, dtype=float)))
    return order / max(len(order) - 1, 1)

def spearman(a, b):
    a, b = ranks(a), ranks(b)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

# ---------------------------------------------------------------- rules
def rule_gate(rep, sel, _order):
    res = rep.replay(sel)
    return None if res is None else np.asarray(res.fused, dtype=float)

def rule_consensus_filter(rep, sel, _order):
    """Drop views that disagree with the aggregate of the others, then hand the
    survivors back to the shipped gate. Drops rather than flips: a wrong guess
    costs one view instead of inverting the ranking."""
    if len(sel) < 3:
        return rule_gate(rep, sel, _order)
    R = {n: ranks(rep.scores[n]) for n in sel}
    keep = []
    for n in sel:
        others = np.mean([R[m] for m in sel if m != n], axis=0)
        if spearman(R[n], others) >= 0:
            keep.append(n)
    return rule_gate(rep, keep or sel, _order)

def rule_agreement_weight(rep, sel, _order):
    """Weight each view by its median agreement with the others, clipped at 0."""
    R = {n: ranks(rep.scores[n]) for n in sel}
    if len(sel) == 1:
        return R[sel[0]]
    w = {}
    for n in sel:
        cs = [spearman(R[n], R[m]) for m in sel if m != n]
        w[n] = max(float(np.median(cs)), 0.0)
    if sum(w.values()) <= 0:
        return rule_gate(rep, sel, _order)
    return np.sum([w[n] * R[n] for n in sel], axis=0) / sum(w.values())

def rule_order_decay(rep, sel, order):
    """The analyst's stated order as a soft prior (1, 1/2, 1/4, ...) instead of
    the hard lead-view rule that was tested and rejected."""
    R = [ranks(rep.scores[n]) for n in sel]
    w = np.array([0.5 ** i for i in range(len(sel))], dtype=float)
    return np.sum([wi * r for wi, r in zip(w, R)], axis=0) / w.sum()

def _ued(rep, sel):
    from arcade.engine.scoring import ScoringEngine
    from arcade.paradigms.registry import BUILTIN_PARADIGMS as P
    eng = ScoringEngine()
    for p in P:
        eng.register_paradigm(p)
    for n in sel:
        eng.set_scores(n, rep.scores[n])
    u = eng.compute_ued_all()
    return {n: float(u.get(n, 0.0)) for n in sel}

def rule_lead_only(rep, sel, order):
    """The analyst's first-named view alone. Rejected as a global rule; here it
    is confined to the branch where the gate has no dominant view anyway."""
    return ranks(rep.scores[sel[0]])

def rule_ued_top1(rep, sel, _order):
    """Dominant selection with the gate's 2x margin requirement removed: run the
    highest-UED view alone whether or not it leads by a margin."""
    u = _ued(rep, sel)
    return ranks(rep.scores[max(u, key=u.get)])

def rule_drop_lowest_ued(rep, sel, _order):
    """Keep the ensemble but discard the single worst view by UED tail quality."""
    if len(sel) < 3:
        return rule_gate(rep, sel, _order)
    u = _ued(rep, sel)
    keep = [n for n in sel if n != min(u, key=u.get)]
    return rule_gate(rep, keep, _order)

_ADJ = {}
def _adjudications(noq=False):
    """Cached answers from experiments/fusion_adjudicator.py, keyed by graph and
    selection. Absent answers leave the gate untouched."""
    key = "noq" if noq else "q"
    if key not in _ADJ:
        _ADJ[key] = {}
        f = os.path.join(HERE, "fusion_adjudicator_noq.jsonl" if noq else "fusion_adjudicator.jsonl")
        if os.path.exists(f):
            for line in open(f):
                d = json.loads(line)
                _ADJ[key][(d["dataset"], d["selection"])] = d
    return _ADJ[key]

def _adjudicate(rep, sel, order, noq):
    a = _adjudications(noq).get((rep_ds[0], ",".join(order)))
    if not a or a.get("mode") != "single":
        return rule_gate(rep, sel, order)
    v = (a.get("view") or "").strip()
    return ranks(rep.scores[v]) if v in rep.scores else rule_gate(rep, sel, order)

def rule_adjudicator(rep, sel, order):
    """Candidate 5: the analyst is asked a second, narrower question (combine
    these views, or use exactly one) with the pairwise agreement matrix and each
    view's shape in front of it."""
    return _adjudicate(rep, sel, order, noq=False)

def rule_adjudicator_noq(rep, sel, order):
    """The same question with the two quality-looking statistics withheld: 12 of
    the first 15 amazon_fraud adjudications ranked by tail_separation or UED,
    neither of which predicts which view to keep."""
    return _adjudicate(rep, sel, order, noq=True)

rep_ds = [None]                      # dataset currently loaded, set in main()

RULES = {"gate (shipped)": rule_gate, "consensus_filter": rule_consensus_filter,
         "agreement_weight": rule_agreement_weight, "order_decay": rule_order_decay,
         "lead_only": rule_lead_only, "ued_top1": rule_ued_top1,
         "drop_lowest_ued": rule_drop_lowest_ued, "adjudicator": rule_adjudicator,
         "adjudicator_noq": rule_adjudicator_noq}

def main():
    only = set(sys.argv[1:])
    rows = [d for d in decisions() if not only or d["dataset"] in only]
    by_ds = collections.defaultdict(list)
    for d in rows:
        by_ds[d["dataset"]].append(d)
    OUTF = os.path.join(HERE, "fusion_rules.json")
    prev = json.load(open(OUTF)) if os.path.exists(OUTF) else []
    # keep rows for graphs this run does not touch, so partial runs accumulate
    out = [r for r in prev if r["dataset"] not in by_ds]
    for ds in sorted(by_ds):
        try:
            rep = DatasetReplayer(ds)
        except Exception as e:
            print(f"[skip] {ds}: {type(e).__name__}", flush=True); continue
        rep_ds[0] = ds
        y = (np.asarray(rep.labels) > 0).astype(int)
        single = {n: roc_auc_score(y, rep.scores[n]) * 100 for n in rep.available}
        for d in by_ds[ds]:
            sel = [v for v in d["selected"] if v in rep.scores]
            if not sel:
                continue
            shipped = rep.replay(sel)
            if shipped is None:
                continue
            mode = shipped.fusion_mode
            rec = {**d, "selected": sel, "mode": mode,
                   "oracle_best_selected": round(max(single[n] for n in sel), 1),
                   "worst_selected": round(min(single[n] for n in sel), 1),
                   "has_anti_signal": bool(min(single[n] for n in sel) < 50),
                   "held_out": ds in HELDOUT}
            # Label-free diagnostics of the selection itself. None of these is
            # given to a rule here; the question they answer is whether the two
            # regimes (ensemble helps / ensemble dilutes) are separable at all
            # without labels, which is what a conditional rule would need.
            R = {n: ranks(rep.scores[n]) for n in sel}
            pair = [spearman(R[a], R[b]) for i, a in enumerate(sel) for b in sel[i + 1:]]
            u = _ued(rep, sel)
            ts = {n: float(ScoringEngine.tail_separation(rep.scores[n])) for n in sel}
            rec.update({
                "agree_mean": round(float(np.mean(pair)), 3) if pair else None,
                "agree_min": round(float(np.min(pair)), 3) if pair else None,
                "ued_max": round(max(u.values()), 3), "ued_min": round(min(u.values()), 3),
                "ued_ratio": round(max(u.values()) / max(sorted(u.values())[-2], 1e-9), 2)
                             if len(u) > 1 else None,
                "tailsep_max": round(max(ts.values()), 3),
                "n_views": len(sel),
            })
            base = np.asarray(shipped.fused, dtype=float)
            for name, fn in RULES.items():
                # A candidate rule replaces ONLY the diverse-ensemble branch.
                # Trigger trust, the pairmax union and dominant selection are
                # separate decisions with their own evidence, and a rule that
                # overrides them is being measured on the wrong question.
                if name == "gate (shipped)" or mode != "aom":
                    fused = base
                else:
                    fused = fn(rep, sel, d["selected"])
                    if fused is None:
                        fused = base
                rec[name] = round(roc_auc_score(y, fused) * 100, 1)
            out.append(rec)
        print(f"[fusion] {ds}: {len(by_ds[ds])} decisions", flush=True)
        del rep
        json.dump(out, open(os.path.join(HERE, "fusion_rules.json"), "w"), indent=1)

    def block(title, rs):
        if not rs:
            return
        print(f"\n=== {title} ({len(rs)} decisions, "
              f"{sum(r['has_anti_signal'] for r in rs)} contain a below-chance view) ===")
        cols = list(RULES) + ["oracle_best_selected"]
        hdr = f"  {'graph':14s} {'n':>4s}" + "".join(f"{c[:17]:>18s}" for c in cols)
        print(hdr)
        for ds in sorted({r['dataset'] for r in rs}):
            d = [r for r in rs if r['dataset'] == ds]
            line = f"  {ds:14s} {len(d):4d}"
            for c in cols:
                v = [r[c] for r in d if r.get(c) is not None]
                line += f"{np.mean(v):18.1f}" if v else f"{'-':>18s}"
            print(line)
        line = f"  {'MEAN':14s} {len(rs):4d}"
        base = None
        for c in cols:
            v = [r[c] for r in rs if r.get(c) is not None]
            m = float(np.mean(v)) if v else float('nan')
            if c == "gate (shipped)":
                base = m
            line += f"{m:18.1f}"
        print(line)
        print(f"  {'vs gate':14s} {'':4s}" + "".join(
            f"{(np.mean([r[c] for r in rs if r.get(c) is not None]) - base):+18.1f}" for c in cols))

    block("benchmark graphs", [r for r in out if not r["held_out"]])
    block("held-out graphs", [r for r in out if r["held_out"]])
    block("benchmark, ensemble branch only",
          [r for r in out if not r["held_out"] and r["mode"] == "aom"])
    block("held-out, ensemble branch only",
          [r for r in out if r["held_out"] and r["mode"] == "aom"])
    print("\nFUSION_BENCH_DONE", flush=True)

main()
