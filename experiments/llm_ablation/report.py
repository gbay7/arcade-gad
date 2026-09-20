"""Aggregate results.jsonl into the ablation report (markdown).

Per model: decision validity, selection stability across repeats, mean AUROC
over datasets (repeat-averaged), latency — plus the baseline arms and, when
present, per-dataset breakdown.

    python -m experiments.llm_ablation.report
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from statistics import mean

from . import config


def load() -> list[dict]:
    if not os.path.exists(config.RESULTS_PATH):
        return []
    with open(config.RESULTS_PATH) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # A cell may be re-run (e.g. a timeout later filled at a longer budget):
    # keep only the LATEST row per (model, dataset, repeat, temperature).
    latest: dict[tuple, dict] = {}
    for r in rows:
        latest[(r["model"], r["dataset"], r.get("repeat", 0), r.get("temperature"))] = r
    return list(latest.values())


def summarize(rows: list[dict]) -> str:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    def _is_sampled(r: dict) -> bool:
        # rows that measure decision variability: ollama at T>0, or any Claude
        # row (the CLI samples at its default temperature).
        t = r.get("temperature")
        return (t is None and r["model"].startswith("claude:")) or (t or 0) > 0

    lines = ["# LLM ablation report",
             "",
             "Outcome columns use canonical rows (greedy/T=0 for ollama; all rows for "
             "Claude). Stability is measured on sampled rows only — identical re-decisions "
             "under sampling; '–' means no sampled rows yet (run ABLATION_TEMP=0.7).",
             "",
             "| model | kind | datasets | valid-JSON | applicable-only | stability | mean AUROC | mean AUPRC | med latency |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model, rs in sorted(by_model.items(), key=lambda kv: kv[0]):
        kind = rs[0]["kind"]
        datasets = sorted({r["dataset"] for r in rs})
        valid = mean(1.0 if r.get("valid_json") else 0.0 for r in rs)
        appl = mean(1.0 if r.get("all_applicable") else 0.0 for r in rs)
        # stability over SAMPLED rows: 1 if every sampled repeat picked the same set
        sampled = [r for r in rs if _is_sampled(r)]
        stab_vals = []
        for ds in sorted({r["dataset"] for r in sampled}):
            sels = [tuple(sorted(r.get("selected", []))) for r in sampled if r["dataset"] == ds]
            if len(sels) >= 2:
                stab_vals.append(1.0 if len(set(sels)) == 1 else 0.0)
        stability = f"{mean(stab_vals):.0%}" if stab_vals else "–"
        # outcome over CANONICAL rows (all rows for claude models)
        canonical = [r for r in rs if r["model"].startswith("claude:") or not _is_sampled(r)]
        aurocs, auprcs = [], []
        for ds in datasets:
            vals = [r["auroc"] for r in canonical if r["dataset"] == ds and r.get("executed")]
            pvals = [r["auprc"] for r in canonical if r["dataset"] == ds and r.get("executed")]
            if vals:
                aurocs.append(mean(vals)); auprcs.append(mean(pvals))
        lat = sorted(r.get("latency_s", 0.0) for r in rs)
        med_lat = lat[len(lat) // 2] if lat else 0.0
        lines.append(
            f"| {model} | {kind} | {len(datasets)} | {valid:.0%} | {appl:.0%} | "
            f"{stability} | {mean(aurocs):.1f} | {mean(auprcs):.1f} | {med_lat:.0f}s |"
            if aurocs else
            f"| {model} | {kind} | {len(datasets)} | {valid:.0%} | {appl:.0%} | "
            f"{stability} | – | – | {med_lat:.0f}s |")

    # per-dataset AUROC matrix (repeat means)
    datasets = sorted({r["dataset"] for r in rows})
    lines += ["", "## AUROC by dataset (repeat mean)", "",
              "| model | " + " | ".join(datasets) + " |",
              "|---|" + "---|" * len(datasets)]
    for model, rs in sorted(by_model.items(), key=lambda kv: kv[0]):
        cells = []
        for ds in datasets:
            vals = [r["auroc"] for r in rs if r["dataset"] == ds and r.get("executed")]
            cells.append(f"{mean(vals):.1f}" if vals else "–")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = load()
    if not rows:
        print("no results yet"); return
    report = summarize(rows)
    out = os.path.join(os.path.dirname(config.RESULTS_PATH), "REPORT.md")
    with open(out, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n[report] written to {out}")


if __name__ == "__main__":
    main()
