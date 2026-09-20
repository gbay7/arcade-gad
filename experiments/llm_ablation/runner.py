"""Ablation runner: evidence (MCP) -> model decision (ollama) -> genuine replay
(ScoringEngine gate) -> single measurement -> results.jsonl.

Per dataset the evidence pack is built once over MCP (the same context path any
agent uses). Each model answers the same structured question REPEATS times
(temperature 0; the sampling seed varies per repeat to expose instability).
Compositions are replayed against cached paradigm scores through the genuine
label-free gate; labels enter once, at measurement.

    python -m experiments.llm_ablation.runner              # full matrix
    python -m experiments.llm_ablation.runner inj_cora     # one dataset
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from typing import Any

import numpy as np

from . import config
from .cache import DatasetReplayer, build_cache
from .mcp_client import ArcadeMCP
from .ollama_client import OllamaClient, list_local_models
from .protocol import CHECKLIST_SCHEMA, DECISION_SCHEMA, Decision, EvidencePack
from arcade.paradigms.registry import BUILTIN_PARADIGMS as PARADIGMS_ALL

# fast paradigms whose cached score distributions serve as label-free landmarkers
FAST_FOR_STATS = ["local_affinity", "density", "community", "homophily_violation",
                  "spectral", "attr_struct_mismatch", "clique_density", "conformity",
                  "linear_residual", "peripheral_mismatch", "attribute_inflation"]
CLOSED_FORM = ["conformity", "linear_residual", "peripheral_mismatch",
               "attribute_inflation", "clique_density"]


def _dist_stats(v: np.ndarray) -> dict[str, float]:
    # NOTE: a per-paradigm tail-separation score was added here once (pack v3)
    # and REMOVED after measurement: models ranked paradigms by the number
    # instead of reasoning about mechanisms, and the statistic is a raw-
    # distribution property, not detector quality (on the graph where the
    # mechanism evidence mattered most it argued for the wrong paradigms and
    # un-flipped previously correct decisions). Quantified "quality-looking"
    # numbers in an evidence pack act as a false ranking signal.
    q = np.quantile(v, [0.5, 0.9, 0.95, 0.99])
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
            "p50": round(float(q[0]), 4), "p90": round(float(q[1]), 4),
            "p95": round(float(q[2]), 4), "p99": round(float(q[3]), 4)}


async def build_pack(dataset: str, replayer: DatasetReplayer) -> EvidencePack:
    async with ArcadeMCP() as arcade:
        await arcade.call("load_dataset", {"name": dataset})
        prof = await arcade.call("graph_profile")
        appl = await arcade.call("applicable_paradigms")
    fast_stats = {n: _dist_stats(replayer.scores[n])
                  for n in FAST_FOR_STATS if n in replayer.scores}
    # Deep views were evidence-blind: fast paradigms get distribution stats, deep
    # ones got nothing, so no analyst had grounds to select them. Provide a CHEAP
    # label-free probe (few epochs, few test rounds) — a preview, clearly labeled,
    # so the analyst has evidence about the deep view without paying its full cost.
    applicable_names = [p["name"] for p in appl]
    # ABLATION_PROBE_ALL=1 generalizes the probe to every applicable deep paradigm
    # (5 epochs each), the uniform form of the design rule; the study's packs
    # probed only the subgraph-contrastive view.
    # A probe is evidence only if the short run ranks nodes the way the full run
    # does. Measured (experiments/probe_fidelity.py) that is false for three of
    # the five deep views: over 8 graphs the median probe-to-full rank
    # correlation is 0.63 for reconstruction but 0.28 contrastive, 0.20
    # adversarial and 0.04 one-class, and on Tolokers the one-class probe scores
    # 43.1 against a full run of 62.7 with correlation -0.03, so the pack was
    # handing the analyst evidence AGAINST the view that wins that graph.
    # probe_policy.json (label-free calibration, see build_probe_policy.py) says
    # which paradigms may show a probe and at which budget. ABLATION_PROBE_LEGACY=1
    # restores the old behaviour for reproducing the frozen study.
    legacy = os.environ.get("ABLATION_PROBE_LEGACY") == "1"
    probe_all = os.environ.get("ABLATION_PROBE_ALL") == "1"
    if os.environ.get("ABLATION_NO_PROBE") != "1":
        from arcade.paradigms.detectors import run_detector
        byname = {p.name: p for p in PARADIGMS_ALL}
        pol_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_policy.json")
        policy = json.load(open(pol_path)) if (os.path.exists(pol_path) and not legacy) else None
        deep = ["reconstruction", "contrastive", "adversarial", "one_class", "subgraph_contrast"]
        if legacy:
            candidates = ([n for n in deep[:4] if probe_all] + ["subgraph_contrast"])
        else:
            candidates = deep
        withheld = []
        for name in candidates:
            if name not in applicable_names:
                continue
            if policy is not None:
                entry = (policy.get("paradigms") or {}).get(name, {})
                if not entry.get("show_probe"):
                    withheld.append(name)
                    continue
                budget = entry.get("budget") or 5
            else:
                budget = 5
            kw = {"epochs": budget, "seed": 1}
            if name == "subgraph_contrast":
                kw["test_rounds"] = 4 if budget <= 5 else 8
            try:
                sc = run_detector(byname[name].detector_name, replayer._store, **kw)
            except TypeError:
                sc = run_detector(byname[name].detector_name, replayer._store, seed=1)
            except Exception:
                # GPU busy (another arm's model loaded): same probe on CPU.
                os.environ["ARCADE_FORCE_CPU"] = "1"
                try:
                    sc = run_detector(byname[name].detector_name, replayer._store, **kw)
                finally:
                    os.environ.pop("ARCADE_FORCE_CPU", None)
            label = (f"{name} (preview: the same view trained for {budget} epochs instead of 100; "
                     f"its ranking tracks the full run, rank correlation "
                     f"{(policy['paradigms'][name]['budget%d' % budget]['median']) if policy else 'n/a'} "
                     f"median over the calibration graphs)") if policy else \
                    f"{name} (cheap {budget}-epoch probe — full run is much stronger)"
            fast_stats[label] = _dist_stats(np.asarray(sc))
        if withheld:
            # Absence of a preview must not read as evidence against a paradigm:
            # that is how the old pack made one-class invisible.
            fast_stats["_no_preview_for"] = {
                "paradigms": withheld,
                "why": ("A short run of these views does not rank nodes like their own full run "
                        "(calibrated offline without labels), so no preview is shown. This says "
                        "nothing about whether the paradigm suits this graph; judge them from "
                        "the catalog and the profile."),
            }
    # ABLATION_DOMAIN=1: add the expert description of the graph and task (what it
    # is, what nodes/edges/attributes mean, what the practitioner looks for) from
    # domain_context.json; still label-free, still no benchmark name.
    domain = None
    if os.environ.get("ABLATION_DOMAIN") == "1":
        ctx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "domain_context.json")
        entry = json.load(open(ctx_path)).get(dataset)
        domain = entry["text"] if isinstance(entry, dict) else entry
        if domain is None:
            print(f"[warn] no domain description for {dataset}", flush=True)
    # ABLATION_RICH=1: enrich the catalog and add a legend for every profile and
    # landmarker field. The shipped registry is untouched; the text comes from
    # catalog_rich.json and is derived from the implementations (what each view
    # computes, in which reference frame, from which inputs, and the conditions
    # under which its formula degenerates), plus literal definitions of the
    # statistics. It states mechanism facts, never which paradigm to pick.
    legend = None
    if os.environ.get("ABLATION_RICH") == "1":
        rich_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog_rich.json")
        rich = json.load(open(rich_path))
        # ABLATION_LEGEND=v1 restores the first legend, which also told the
        # analyst what to look for in the landmarkers; v2 (default) defines
        # the fields and stops there.
        legend = rich.get("_stats_legend_prescriptive_v1") \
            if os.environ.get("ABLATION_LEGEND") == "v1" else rich.get("_stats_legend")
        missing = [p["name"] for p in appl if p["name"] not in rich]
        if missing:
            print(f"[warn] no enriched entry for {missing}", flush=True)
        for p in appl:
            entry = rich.get(p["name"])
            if entry:
                p.update({k: v for k, v in entry.items() if not k.startswith("_")})
                # The enriched fields REPLACE the shipped one-line description.
                # ABLATION_RICH_KEEP_DESC=1 keeps it as well, to separate the
                # effect of the added text from the effect of the removed text.
                if os.environ.get("ABLATION_RICH_KEEP_DESC") != "1":
                    p.pop("description", None)
    return EvidencePack(dataset_key=dataset, profile=prof.get("data", prof),
                        paradigms=appl, fast_stats=fast_stats, domain=domain, legend=legend)


def arm_name() -> str:
    """Name of the evidence-interface arm, from the environment that built the
    pack. Recorded on every decision and transcript line so arms never have to
    be inferred from which file a row landed in."""
    parts = []
    if os.environ.get("ABLATION_RICH") == "1":
        parts.append("rich" if os.environ.get("ABLATION_RICH_KEEP_DESC") != "1" else "rich+desc")
        if os.environ.get("ABLATION_LEGEND") == "v1":
            parts.append("legend-v1")
    if os.environ.get("ABLATION_DOMAIN") == "1":
        parts.append("domain")
    if os.environ.get("ABLATION_PROBE_ALL") == "1":
        parts.append("probeall")
    if os.environ.get("ABLATION_PROTOCOL") == "checklist":
        parts.append("checklist")
    return "+".join(parts) if parts else "study"


def baseline_decision(arm: str, applicable: list[str], rng: random.Random) -> list[str]:
    if arm == "all_applicable":
        return list(applicable)
    if arm == "random3":
        k = min(3, len(applicable))
        return rng.sample(applicable, k)
    if arm == "closed_form":
        cf = [p for p in applicable if p in CLOSED_FORM]
        return cf or [applicable[0]]
    raise ValueError(arm)


def execute(replayer: DatasetReplayer, selected: list[str]) -> dict[str, Any]:
    rep = replayer.replay(selected)
    if rep is None:
        return {"executed": False}
    metrics = replayer.measure(rep.fused)
    return {"executed": True, "used": rep.used, "fusion_mode": rep.fusion_mode,
            "trusted": rep.trusted, "weights": rep.weights, **metrics}


def append_result(record: dict[str, Any]) -> None:
    # ABLATION_RESULTS redirects output (e.g. a pack-v2 pass measured against a
    # frozen pack-v1 archive for before/after comparisons).
    path = os.environ.get("ABLATION_RESULTS", config.RESULTS_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_dataset(dataset: str, models: list[str], skip_baselines: bool = False) -> None:
    print(f"=== {dataset} ===", flush=True)
    build_cache(dataset)                       # no-op when cached
    replayer = DatasetReplayer(dataset)
    pack = asyncio.run(build_pack(dataset, replayer))
    applicable = [p["name"] for p in pack.paradigms]

    # reference arms (deterministic seeds per dataset)
    for arm in ([] if skip_baselines else config.BASELINE_ARMS):
        rng = random.Random(f"{dataset}:{arm}")
        sel = baseline_decision(arm, applicable, rng)
        rec = {"kind": "baseline", "model": arm, "dataset": dataset, "repeat": 0,
               "selected": sel, "valid_json": True, "all_applicable": True,
               **execute(replayer, sel), "ts": time.time()}
        append_result(rec)
        print(f"[arm ] {arm:16s} {rec.get('auroc')}  sel={sel}", flush=True)

    # ABLATION_TEMP: 0 = canonical greedy decision; >0 = consistency probe
    # (at temperature 0 ollama decoding is greedy, so seed-repeats are trivially
    # identical — real decision stability must be measured at sampling temp).
    temperature = float(os.environ.get("ABLATION_TEMP", config.TEMPERATURE))
    local = list_local_models()
    arm = arm_name()
    for model in models:
        is_claude = model.startswith("claude:")
        is_opencode = model.startswith("opencode:")
        if not (is_claude or is_opencode) and not any(m.startswith(model) for m in local):
            print(f"[skip] {model} not pulled", flush=True)
            continue
        for repeat in range(config.REPEATS):
            transcript = os.path.join(
                config.TRANSCRIPT_DIR, f"{model.replace(':', '_')}__{dataset}.jsonl")
            if is_claude:
                from .claude_client import ClaudeClient
                # separate headless session per decision; temperature/seed are
                # not exposed by the CLI — repeats measure natural variability
                client = ClaudeClient(model.split(":", 1)[1], transcript_path=transcript)
            elif is_opencode:
                from .opencode_client import OpencodeClient
                # free-tier serving is slow on evidence-pack-sized prompts
                client = OpencodeClient(model.split(":", 1)[1], transcript_path=transcript,
                                        timeout_s=float(os.environ.get("ABLATION_TIMEOUT", 900)))
            else:
                client = OllamaClient(
                    model, seed=repeat + 1, temperature=temperature,
                    num_ctx=config.NUM_CTX, transcript_path=transcript,
                    timeout_s=float(os.environ.get("ABLATION_TIMEOUT", 600)))
            checklist = os.environ.get("ABLATION_PROTOCOL") == "checklist"
            schema = CHECKLIST_SCHEMA if checklist else DECISION_SCHEMA
            # join key for the transcript line this call will write
            client.context = {"dataset": dataset, "repeat": repeat, "arm": arm}
            try:
                parsed, res = client.decide(pack.messages(checklist=checklist), schema)
            except Exception as e:
                # A model that cannot serve the protocol is a RESULT (capability
                # failure), not a reason to abort the matrix.
                client.close()
                rec = {"kind": "llm", "model": model, "dataset": dataset,
                       "repeat": repeat, "arm": arm, "selected": [], "valid_json": False,
                       "all_applicable": False, "executed": False,
                       "error": f"{type(e).__name__}: {str(e)[:160]}", "ts": time.time()}
                append_result(rec)
                print(f"[llm ] {model:14s} r{repeat} PROTOCOL-FAIL {rec['error'][:60]}",
                      flush=True)
                continue
            client.close()
            decision = Decision.from_response(parsed, applicable)
            rec = {"kind": "llm", "model": model, "dataset": dataset, "repeat": repeat,
                   "temperature": None if (is_claude or is_opencode) else temperature,
                   "protocol": "checklist" if checklist else "freeform",
                   "selected": decision.selected, "rationale": decision.rationale,
                   "valid_json": decision.valid_json,
                   "all_applicable": decision.all_applicable,
                   "invalid_names": decision.invalid_names,
                   "latency_s": res.latency_s,
                   "tokens": [res.prompt_tokens, res.completion_tokens],
                   "arm": arm,
                   # the model's own reasoning, when the provider returns it
                   # (ollama does; the Claude CLI returns signed-but-empty
                   # thinking blocks, so only its token count is meaningful)
                   "thinking": getattr(res, "thinking", "") or "",
                   "thinking_chars": len(getattr(res, "thinking", "") or ""),
                   "raw_content": res.content,
                   **execute(replayer, decision.selected), "ts": time.time()}
            append_result(rec)
            print(f"[llm ] {model:14s} r{repeat} {rec.get('auroc')}  "
                  f"sel={decision.selected}{' INVALID' if not decision.valid_json else ''}",
                  flush=True)


def main() -> None:
    datasets = sys.argv[1:] or config.DATASETS
    # ABLATION_MODELS=qwen3:8b,glm4:latest overrides the roster (e.g. to fill in
    # rows for one model without duplicating the others).
    override = os.environ.get("ABLATION_MODELS")
    models = override.split(",") if override else config.MODELS
    skip_baselines = bool(override)
    for ds in datasets:
        run_dataset(ds, models, skip_baselines=skip_baselines)
    print("RUNNER_DONE", flush=True)


if __name__ == "__main__":
    main()
