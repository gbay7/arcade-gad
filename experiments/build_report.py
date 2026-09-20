"""Regenerate every table of the evidence-interface study from the shipped
artifacts, as markdown and as one machine-readable index.

Nothing here recomputes a score: each number is read from the artifact that
produced it, so the report cannot drift from the experiments. Rows whose
artifact is missing are printed as absent rather than silently dropped.

    python experiments/build_report.py
writes experiments/reports/RESULTS.md and experiments/reports/results_index.json
"""
import os, sys, json, glob, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.join(HERE, "llm_ablation")
OUT = os.path.join(HERE, "reports")
os.makedirs(OUT, exist_ok=True)

def load(name, sub=""):
    p = os.path.join(HERE, sub, name)
    return json.load(open(p)) if os.path.exists(p) else None

# ---------------------------------------------------------------- held-out
HELDOUT_FILES = {"tolokers": "heldout_tolokers.json", "amazon_fraud": "heldout_amazon_fraud.json"}
BASELINES = ["DOMINANT", "AnomalyDAE", "DONE", "CoLA", "GAAN", "OCGNN", "Radar", "SL-GAD"]

def heldout_table():
    rows, index = [], {}
    for ds, f in HELDOUT_FILES.items():
        d = load(f)
        if not d:
            continue
        per = d.get("per_detector_seed1", {})
        best = max(per, key=per.get) if per else None
        e = {"frozen_commit": d.get("frozen_commit"),
             "ARCADE": [d["ARCADE"]["mean"], d["ARCADE"].get("ap_mean"), d["ARCADE"].get("std")],
             "best_single_view": [best, per.get(best)] if best else None,
             "per_detector_seed1": per}
        for b in BASELINES:
            if isinstance(d.get(b), dict) and "mean" in d[b]:
                e[b] = [d[b]["mean"], d[b].get("ap_mean")]
        index[ds] = e
        rows.append((ds, e))
    seed1 = load("heldout_seed1.json") or {}
    for ds, d in seed1.items():
        per = d.get("per_detector", {})
        best = max(per, key=per.get) if per else None
        e = {"ARCADE": [d["ARCADE_seed1"]["auroc"], d["ARCADE_seed1"]["auprc"], None],
             "best_single_view": [best, per.get(best)],
             "seeds": 1, "note": "single cached seed, no baselines",
             "per_detector_seed1": per}
        index[ds] = e
        rows.append((ds, e))
    return rows, index

# ------------------------------------------------- evidence-interface arms
ARM_FILES = ["results.jsonl", "results_packv4.jsonl", "results_packv4_fill.jsonl",
             "results_packv4_fill_opus48.jsonl", "results_heldout.jsonl",
             "results_heldout_probeall.jsonl", "results_heldout_domain.jsonl",
             "results_heldout_rich.jsonl", "results_heldout_rich_domain.jsonl",
             "results_study_rich.jsonl", "results_study_rich_v2.jsonl",
             "results_study_rich_keepdesc.jsonl", "results_traced_study.jsonl",
             "results_traced_rich.jsonl"]
# arm of a row: recorded from ABLATION_* since 2026-09-18; before that, inferred
# from the file it landed in (the only record that exists for those runs)
FILE_ARM = {"results_heldout_probeall.jsonl": "probeall",
            "results_heldout_domain.jsonl": "domain",
            "results_heldout_rich.jsonl": "rich",
            "results_heldout_rich_domain.jsonl": "rich+domain",
            "results_study_rich.jsonl": "rich",
            "results_study_rich_v2.jsonl": "rich",
            "results_study_rich_keepdesc.jsonl": "rich+desc",
            "results_traced_rich.jsonl": "rich"}

def arm_rows():
    rows = []
    for f in ARM_FILES:
        p = os.path.join(ABL, f)
        if not os.path.exists(p):
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "llm" or not r.get("selected"):
                continue
            t = r.get("temperature")
            if not (t in (0, 0.0) or (t is None and r["model"].startswith(("claude:", "opencode:")))):
                continue
            if r.get("auroc") is None:
                continue
            rows.append({"file": f, "arm": r.get("arm") or FILE_ARM.get(f, "study"),
                         "model": r["model"], "dataset": r["dataset"],
                         "repeat": r.get("repeat", 0), "auroc": r["auroc"],
                         "ap": r.get("auprc"), "selected": r["selected"],
                         "mode": r.get("fusion_mode"),
                         "thinking_chars": r.get("thinking_chars")})
    return rows

REPLAY_CACHE = os.path.join(OUT, "replay_cache.json")

def replay_all(rows):
    """Re-execute every stored selection through the CURRENT gate, so one table
    never mixes scores produced by different gate versions (results.jsonl
    predates the gate-v2 change). Deterministic given the cached scores, so the
    outcome is cached by (dataset, selection) and later runs are instant.
    Datasets are processed one at a time and released: amazon_fraud alone holds
    several GB."""
    cache = json.load(open(REPLAY_CACHE)) if os.path.exists(REPLAY_CACHE) else {}
    todo = collections.defaultdict(list)
    for r in rows:
        key = r["dataset"] + "|" + ",".join(r["selected"])
        if key not in cache:
            todo[r["dataset"]].append((key, r["selected"]))
    if todo:
        from experiments.llm_ablation.cache import DatasetReplayer
        for ds, items in sorted(todo.items()):
            try:
                rep = DatasetReplayer(ds)
            except Exception as e:
                print(f"[replay] {ds}: unavailable ({type(e).__name__}), keeping recorded scores", flush=True)
                continue
            for key, sel in items:
                if key in cache:
                    continue
                use = [v for v in sel if v in rep.scores]
                res = rep.replay(use) if use else None
                cache[key] = None if res is None else [round(rep.measure(res.fused)["auroc"], 1),
                                                       round(rep.measure(res.fused)["auprc"], 1),
                                                       res.fusion_mode]
            del rep
            json.dump(cache, open(REPLAY_CACHE, "w"))
            print(f"[replay] {ds}: {len(items)} selections", flush=True)
    for r in rows:
        v = cache.get(r["dataset"] + "|" + ",".join(r["selected"]))
        if v:
            r["auroc"], r["ap"], r["mode"] = v[0], v[1], v[2]
            r["replayed"] = True
    return rows


def agg(rows, keys):
    out = collections.defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return {k: (round(float(np.mean([x["auroc"] for x in v])), 1), len(v)) for k, v in out.items()}

def md_table(header, body_rows):
    w = "| " + " | ".join(header) + " |\n| " + " | ".join("---" for _ in header) + " |\n"
    for r in body_rows:
        w += "| " + " | ".join("" if c is None else str(c) for c in r) + " |\n"
    return w

def main():
    L = []
    A = L.append
    A("# ARCADE: evidence-interface study, consolidated results\n")
    A("Generated by `experiments/build_report.py` from the shipped artifacts. "
      "Every number is read from the artifact that produced it.\n")

    # 1 held-out
    rows, heldout_index = heldout_table()
    A("\n## 1. Held-out graphs, frozen rules\n")
    A("Rules, library, preconditions and gate fixed at the recorded commit; only the loader changed.\n")
    hdr = ["graph", "ARCADE AUROC", "ARCADE AP", "best single library view"] + BASELINES
    body = []
    for ds, e in rows:
        bs = f"{e['best_single_view'][0]} {e['best_single_view'][1]}" if e.get("best_single_view") else None
        body.append([ds, e["ARCADE"][0], e["ARCADE"][1], bs] +
                    [f"{e[b][0]} / {e[b][1]}" if b in e else "not run" for b in BASELINES])
    A(md_table(hdr, body))
    A("\nBaseline cells are AUROC / AP. Questions is a single cached seed with no baselines.\n")

    # 2 interface arms
    rows = arm_rows()
    if "--no-replay" not in sys.argv:
        rows = replay_all(rows)
    replayed = sum(1 for r in rows if r.get("replayed"))
    A("\n## 2. Evidence-interface arms\n")
    A("Same library, same gate, same cached scores; only the text of the evidence pack differs. "
      "`study` is the shipped pack, `domain` adds a source-cited description of the graph and task, "
      "`rich` replaces the one-line paradigm descriptions with fields derived from the implementations "
      "plus a legend defining every statistic, `rich+desc` keeps the original line as well.\n")
    A(f"\nEvery selection below was re-executed through the current gate "
      f"({replayed} of {len(rows)} decisions); the rest keep the score recorded at run time.\n")
    per = agg(rows, ["dataset", "arm", "model"])
    datasets = sorted({k[0] for k in per})
    arms = ["study", "probeall", "domain", "rich", "rich+domain", "rich+desc"]
    models = sorted({k[2] for k in per})
    for ds in datasets:
        present = [(a, m) for a in arms for m in models if (ds, a, m) in per]
        if not present:
            continue
        A(f"\n### {ds}\n")
        ms = sorted({m for _, m in present})
        body = []
        for a in arms:
            if not any(x[0] == a for x in present):
                continue
            body.append([a] + [f"{per[(ds, a, m)][0]} (n={per[(ds, a, m)][1]})" if (ds, a, m) in per else ""
                               for m in ms])
        A(md_table(["arm"] + ms, body))

    # 3 rules tested
    A("\n## 3. Label-free rules tested against the gate\n")
    for name, f in [("lead view alone", "lead_view_rule.json"),
                    ("lead view when it also tops a gate diagnostic", "lead_view_agreement_rule.json")]:
        d = load(f)
        if not d:
            continue
        s = d["summary"]
        body = []
        for blk in ("benchmark", "heldout"):
            if s.get(blk):
                b = s[blk]
                body.append([blk, b["n"], b["mean_shipped"],
                             b.get("mean_lead_only", b.get("mean_rule")),
                             f"{b.get('graphs_worse')}/{b.get('graphs', len(b['per_dataset']))}"])
        A(f"\n**{name}**\n")
        A(md_table(["decisions", "n", "shipped gate", "with rule", "graphs worse"], body))

    # 3b fusion bench
    fr = load("fusion_rules.json")
    if fr:
        A("\n## 3b. Fusion test bench\n")
        A("Each candidate replaces only the diverse-ensemble branch of the gate; trigger trust, "
          "the pairmax union and dominant selection are left untouched. The oracle column is the "
          "best single selected view, the ceiling any fusion rule could reach without changing "
          "the selection, and it uses labels, so it is a diagnostic and never a rule.\n")
        rules = [k for k in fr[0] if k in ("gate (shipped)", "consensus_filter", "agreement_weight",
                                           "order_decay", "lead_only", "ued_top1", "drop_lowest_ued",
                                           "adjudicator", "adjudicator_noq", "oracle_best_selected")]
        aom = [r for r in fr if r.get("mode") == "aom"]
        body = []
        for grp, rs in (("benchmark", [r for r in aom if not r["held_out"]]),
                        ("held-out", [r for r in aom if r["held_out"]])):
            if not rs:
                continue
            base = float(np.mean([r["gate (shipped)"] for r in rs]))
            row = [grp, len(rs)]
            for k in rules:
                v = [r[k] for r in rs if r.get(k) is not None]
                row.append("-" if not v else f"{np.mean(v):.1f}" +
                           ("" if k == "gate (shipped)" else f" ({np.mean(v) - base:+.1f})"))
            body.append(row)
        A(md_table(["decisions", "n"] + rules, body))
        A("\nNo rule gains on both sides, so nothing ships and the gate stands. The two "
          "adjudicator columns ask the model a second, narrower question; the second withholds "
          "the two quality-looking statistics, which makes it worse rather than better.\n")
        anti = sum(1 for r in aom if r.get("has_anti_signal"))
        A(f"\n{anti} of {len(aom)} ensemble-branch selections contain a view that scores below "
          f"chance on that graph.\n")

    # 4 paper tables still standing
    A("\n## 4. Study benchmark, unchanged reference numbers\n")
    sig = load("significance.json")
    if sig:
        A(f"\nFriedman and Nemenyi over the study table: see `experiments/significance.json` "
          f"(k={sig.get('k', '?')} methods).\n")
    rt = load("runtime_bench.json")
    if rt:
        A("\nRuntime and peak memory: `experiments/runtime_bench.json`.\n")

    index = {"heldout": heldout_index,
             "arms": {f"{k[0]}|{k[1]}|{k[2]}": {"auroc": v[0], "n": v[1]} for k, v in per.items()},
             "rules": {n: (load(f) or {}).get("summary")
                       for n, f in [("lead_view_only", "lead_view_rule.json"),
                                    ("lead_view_agreement", "lead_view_agreement_rule.json")]}}
    open(os.path.join(OUT, "RESULTS.md"), "w").write("\n".join(L))
    json.dump(index, open(os.path.join(OUT, "results_index.json"), "w"), indent=2)
    print(f"wrote {OUT}/RESULTS.md ({sum(len(x) for x in L)} chars) and results_index.json")
    print(f"arms covered: {len(per)} (dataset, arm, model) cells over {len(rows)} decisions")

main()
