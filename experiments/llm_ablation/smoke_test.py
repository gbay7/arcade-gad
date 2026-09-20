"""Phase-0 per-model gate: can this local model consume ARCADE's context and
return a valid selection decision?

For each model: build a real evidence pack over MCP (profile + applicable
paradigms for one graph), ask for the structured decision, and report validity
+ selection + latency. No labels anywhere.

    python -m experiments.llm_ablation.smoke_test qwen3:8b [more models...]
"""
from __future__ import annotations

import asyncio
import sys

from .mcp_client import ArcadeMCP
from .ollama_client import OllamaClient, list_local_models
from .protocol import DECISION_SCHEMA, Decision, EvidencePack

DATASET = "inj_cora"   # internal key; the model never sees it


async def build_pack() -> EvidencePack:
    async with ArcadeMCP() as arcade:
        await arcade.call("load_dataset", {"name": DATASET})
        prof = await arcade.call("graph_profile")
        appl = await arcade.call("applicable_paradigms")
        profile = prof.get("data", prof)
        return EvidencePack(dataset_key=DATASET, profile=profile,
                            paradigms=appl, fast_stats={})


def main() -> int:
    models = sys.argv[1:] or ["qwen3:8b"]
    local = list_local_models()
    pack = asyncio.run(build_pack())
    applicable = [p["name"] for p in pack.paradigms]
    print(f"[smoke] evidence pack ready: {len(applicable)} applicable paradigms")

    failures = 0
    for model in models:
        if not any(m.startswith(model) for m in local):
            print(f"[smoke] {model:14s} SKIP (not pulled yet)")
            continue
        client = OllamaClient(model, transcript_path=f"experiments/llm_ablation/transcripts/smoke_{model.replace(':','_')}.jsonl")
        parsed, res = client.decide(pack.messages(), DECISION_SCHEMA)
        decision = Decision.from_response(parsed, applicable)
        client.close()
        status = "OK" if decision.valid_json and decision.selected else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[smoke] {model:14s} {status}  valid_json={decision.valid_json} "
              f"selected={decision.selected} invalid={decision.invalid_names} "
              f"latency={res.latency_s}s tokens={res.prompt_tokens}+{res.completion_tokens}")
        if decision.rationale:
            print(f"        rationale: {decision.rationale[:220]}")
    print("SMOKE_DONE")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
