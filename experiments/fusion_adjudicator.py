"""Candidate 5 of the fusion plan: ask the analyst to adjudicate the fusion.

Every statistic tried so far can say that the ensemble is diluting but not which
view to keep (see experiments/fusion_rules.json). This asks the model the
question directly, in a second round that sees what the first round could not:
how the selected views agree with each other, what shape each one's scores have,
and the gate's own two diagnostics. No labels, no dataset name beyond the sourced
description the domain arm already used.

Adjudication depends only on (graph, selected views), so it is computed once per
unique pair and applied to every decision that produced that pair.

    python experiments/fusion_adjudicator.py [--model claude:haiku] [graphs...]
"""
import os, sys, json, collections, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import roc_auc_score
from experiments.llm_ablation.cache import DatasetReplayer
from arcade.engine.scoring import ScoringEngine
from arcade.paradigms.registry import BUILTIN_PARADIGMS as PARADIGMS

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(HERE, "llm_ablation")
HELDOUT = {"tolokers", "amazon_fraud", "questions"}
CACHE = os.path.join(HERE, "fusion_adjudicator.jsonl")
CACHE_NOQ = os.path.join(HERE, "fusion_adjudicator_noq.jsonl")

SCHEMA = {"type": "object", "properties": {
    "mode": {"type": "string", "enum": ["single", "combine"],
             "description": "single: one view should carry this graph; combine: fuse them."},
    "view": {"type": "string", "description": "If single, the paradigm to use alone; else empty."},
    "rationale": {"type": "string"}},
    "required": ["mode", "view", "rationale"]}

SYSTEM = (
    "You are the analyst of an unsupervised graph anomaly detection framework, at the "
    "fusion step. A set of detection paradigms has already been selected for this graph "
    "and each has produced a score over the nodes. You have no labels. Your question is "
    "narrower than the selection question: should these views be combined into one "
    "ranking, or does one of them describe the anomaly mechanism well enough that "
    "combining it with the others would dilute it? Averaging helps when the views "
    "measure the same thing from different angles and hurts when they measure "
    "different things or point in opposite directions. Reply with JSON only.")

def ranks(v):
    o = np.argsort(np.argsort(np.asarray(v, dtype=float)))
    return o / max(len(o) - 1, 1)

def spearman(a, b):
    a, b = ranks(a) - ranks(a).mean(), ranks(b) - ranks(b).mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

def build_prompt(ds, sel, rep, catalog, domain, quality=True):
    prof = rep._store.profile()
    prof = prof if isinstance(prof, dict) else prof.__dict__
    eng = ScoringEngine()
    for p in PARADIGMS:
        eng.register_paradigm(p)
    for n in sel:
        eng.set_scores(n, rep.scores[n])
    ued = eng.compute_ued_all()
    views = {}
    for n in sel:
        s = np.asarray(rep.scores[n], dtype=float)
        q = np.quantile(s, [0.5, 0.9, 0.99])
        views[n] = {"what_it_computes": (catalog.get(n) or {}).get("computes", ""),
                    "reference_frame": (catalog.get(n) or {}).get("reference_frame", ""),
                    "degenerates_when": (catalog.get(n) or {}).get("degenerates_when", ""),
                    "score_p50": round(float(q[0]), 4), "score_p90": round(float(q[1]), 4),
                    "score_p99": round(float(q[2]), 4)}
        # The two quality-looking statistics are optional because handing them
        # over reproduces the failure this project keeps hitting: 12 of the first
        # 15 amazon_fraud adjudications cited tail_separation or ued_tail_quality,
        # and the bench shows neither predicts which view to keep (UED ratio
        # correlates +0.04 with the dilution gap). --no-quality withholds them and
        # leaves the mechanism text, the shapes and the agreement matrix.
        if quality:
            views[n]["tail_separation"] = round(float(ScoringEngine.tail_separation(s)), 3)
            views[n]["ued_tail_quality"] = round(float(ued.get(n, 0.0)), 3)
    corr = {f"{a} vs {b}": round(spearman(rep.scores[a], rep.scores[b]), 3)
            for i, a in enumerate(sel) for b in sel[i + 1:]}
    parts = []
    if domain:
        parts += ["DATASET DESCRIPTION (what this graph is and what the practitioner is "
                  "looking for; no labels):", domain, ""]
    parts += ["GRAPH PROFILE (label-free):", json.dumps(prof, indent=1, default=str), "",
              "THE SELECTED VIEWS, in the order the selection named them:",
              json.dumps(views, indent=1), "",
              "HOW THE VIEWS AGREE (Spearman rank correlation between their score vectors; "
              "near zero means they rank different nodes, negative means they contradict "
              "each other):", json.dumps(corr, indent=1), "",
              "Decide: combine these views into one ranking, or use exactly one of them "
              "alone. If one alone, name it."]
    return "\n".join(parts)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--model=")), "claude:haiku")
    quality = "--no-quality" not in sys.argv
    cache_path = CACHE if quality else CACHE_NOQ
    rows = [r for r in json.load(open(os.path.join(HERE, "fusion_rules.json")))
            if r.get("mode") == "aom" and (not args or r["dataset"] in args)]
    catalog = json.load(open(os.path.join(ABL, "catalog_rich.json")))
    domains = json.load(open(os.path.join(ABL, "domain_context.json")))
    done = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            d = json.loads(line)
            done[(d["model"], d["dataset"], d["selection"])] = d

    uniq = collections.OrderedDict()
    for r in rows:
        uniq.setdefault((r["dataset"], ",".join(r["selected"])), r["selected"])
    print(f"{len(rows)} decisions -> {len(uniq)} unique (graph, selection) pairs, model {model}", flush=True)

    from experiments.llm_ablation.claude_client import ClaudeClient
    by_ds = collections.defaultdict(list)
    for (ds, key), sel in uniq.items():
        by_ds[ds].append((key, sel))
    for ds in sorted(by_ds):
        todo = [(k, s) for k, s in by_ds[ds] if (model, ds, k) not in done]
        if not todo:
            continue
        rep = DatasetReplayer(ds)
        dom = domains.get(ds)
        dom = dom["text"] if isinstance(dom, dict) else dom
        for key, sel in todo:
            prompt = build_prompt(ds, sel, rep, catalog, dom, quality=quality)
            cl = ClaudeClient(model.split(":", 1)[1],
                              transcript_path=os.path.join(ABL, "transcripts",
                                                           f"adjudicator_{model.replace(':', '_')}.jsonl"))
            cl.context = {"dataset": ds, "selection": key, "step": "fusion_adjudication",
                          "quality_stats_shown": quality}
            parsed, res = cl.decide([{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": prompt}], SCHEMA)
            rec = {"model": model, "dataset": ds, "selection": key, "selected": sel,
                   "mode": (parsed or {}).get("mode"), "view": (parsed or {}).get("view"),
                   "rationale": (parsed or {}).get("rationale", ""),
                   "latency_s": res.latency_s}
            done[(model, ds, key)] = rec
            with open(cache_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  [adj] {ds:13s} {rec['mode']:8s} {rec['view'] or '':22s} {sel}", flush=True)
        del rep
    print("ADJUDICATOR_DONE", flush=True)

main()
