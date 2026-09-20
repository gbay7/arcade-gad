# LLM-ablation: small open models as the ARCADE analyst

Ablation study over the language model behind ARCADE's reasoning step: can a
small local model (2B–14B, via ollama) read the same label-free evidence and
compose detectors as well as a frontier model?

## Design

- **Same context for every model.** Per dataset, one evidence pack is built from
  the ARCADE MCP server (graph profile + applicable paradigm catalog + fast-
  detector score distributions). It is provenance-agnostic: dataset names are
  hidden and nothing reveals whether anomalies are organic or injected.
- **Same question.** Each model returns one structured decision
  (`selected_paradigms`, `rationale`) under schema-constrained decoding,
  temperature 0; three repeats with different sampling seeds measure stability.
- **Same execution.** Selected compositions are replayed through ARCADE's real
  `ScoringEngine` (the server's exact `compute_weights → set_profile →
  fusion gate → fuse` sequence) against cached paradigm scores (seed 1), so the
  only variable is the selection decision. Labels are used once, to measure.
- **Reference arms.** `closed_form` (applicable closed-form detectors),
  `random3`, `all_applicable` run through the identical replay path.

## Layout

| file | role |
|---|---|
| `mcp_client.py` | stdio MCP client; `python -m ….mcp_client` = handshake check |
| `ollama_client.py` | /api/chat wrapper: JSON-schema decoding, seeds, transcripts |
| `protocol.py` | evidence pack + decision schema + validation |
| `cache.py` | one-time per-dataset paradigm scores + genuine gate replay |
| `runner.py` | full matrix: evidence → decisions → replay → `results.jsonl` |
| `report.py` | aggregates `results.jsonl` → `REPORT.md` |
| `smoke_test.py` | per-model go/no-go gate (valid decision from real evidence) |
| `tests/` | pytest unit tests (no network/GPU needed) |

## Reproduce

```bash
ollama serve &                                   # daemon
for m in gemma4:e2b gemma4:12b qwen3:8b llama3.1:8b phi4:14b; do ollama pull $m; done
python -m experiments.llm_ablation.mcp_client    # handshake (HANDSHAKE_OK)
python -m experiments.llm_ablation.smoke_test qwen3:8b   # per-model gate
python -m experiments.llm_ablation.runner        # full matrix (uses/builds cache)
python -m experiments.llm_ablation.report        # REPORT.md
pytest experiments/llm_ablation/tests -q
```

Models and datasets are configured in `config.py`. Transcripts of every
prompt/response are kept under `transcripts/` for audit. The GPU is shared:
`keep_alive=0` unloads each LLM after its decision so detector runs never
contend for memory.

## Providers

Three provider families share the same `decide()` interface:

- **ollama** (`ollama_client.py`): local models (gemma4:e2b/12b, qwen3:4b/8b,
  llama3.1:8b, phi4:14b), schema-constrained decoding, temperature 0, seeds.
- **claude** (`claude_client.py`): `claude:haiku`, `claude:sonnet`,
  `claude:opus` through headless `claude -p` runs, one fresh session per
  decision (never the interactive session driving a study). The CLI exposes no
  temperature or seed, so repeats measure natural decision variability.
- **opencode** (`opencode_client.py`): free hosted models through the opencode
  CLI (`opencode:opencode/deepseek-v4-flash-free`).

## Environment flags

| flag | effect |
|---|---|
| `ABLATION_MODELS=a,b` | override the roster (baseline arms skipped) |
| `ABLATION_RESULTS=path` | redirect the output jsonl (evidence-pack conditions) |
| `ABLATION_TEMP=0.7` | sampling temperature for the consistency probe (default 0) |
| `ABLATION_PROTOCOL=checklist` | verdict-per-paradigm protocol instead of free-form |
| `ABLATION_THINK=on/off` | toggle native thinking for local models that support it |
| `ABLATION_EFFORT=high` | reasoning-effort control for the claude CLI |
| `ABLATION_NO_PROBE=1` | drop the cheap deep-detector probe from the pack |
| `ABLATION_TIMEOUT=900` | per-decision timeout in seconds |
| `ARCADE_CACHE_SEED=s` | build/use the seed-s score cache (multi-seed coverage) |

## Results files (shipped)

| file | paper artifact |
|---|---|
| `results.jsonl` | evidence condition: decisive field hidden (pack v1) + baseline arms |
| `results_packv2.jsonl` | evidence condition: field exposed, no grounding |
| `results_packv3.jsonl` | evidence condition: field + grounding + quality scores |
| `results_packv4.jsonl` | final interface; the analyst-study table |
| `results_checklist.jsonl` | checklist-protocol ablation |
| `results_thinking.jsonl` | thinking/effort-toggle ablation |
| `transcripts/` | every prompt, response, rationale, latency, cost |
| `score_cache/` | per-dataset per-seed paradigm scores (256-round protocol) |
