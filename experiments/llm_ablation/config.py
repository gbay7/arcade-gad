"""Ablation configuration (kept as code: typed, greppable, no yaml dependency)."""
from __future__ import annotations

# Local models under ablation (all fit the RTX A2000 12GB, all tool-capable).
MODELS: list[str] = [
    "gemma4:e2b",
    "qwen3:4b",      # tiny arm (glm4:latest was dropped: completion-only, no chat endpoint)
    "llama3.1:8b",
    "phi4:14b",
]
# Filled in separate passes (ABLATION_MODELS=...):
#   qwen3:8b     — thinking mode exceeds the default 600s timeout on large packs;
#                  run with ABLATION_TIMEOUT=1800 so its (honest) latency is recorded
#   gemma4:12b   — requires the upgraded ollama daemon (restart after matrix)
#   claude:*     — headless Claude Code sessions (see CLAUDE_MODELS)

# 7 paper benchmarks + 3 dense social benchmarks. inj_flickr (89k nodes) is
# excluded: its deep paradigms fall back to the slow CPU path and dominate the
# cache budget without adding a new regime.
DATASETS: list[str] = [
    "enron", "reddit", "books", "disney", "weibo", "inj_cora", "inj_amazon",
    "blogcatalog", "acm", "cola_flickr",
]

REPEATS = 3          # decision stability: same evidence, sampling seed varies
TEMPERATURE = 0.0    # canonical decisions; consistency pass sets ABLATION_TEMP=0.7
NUM_CTX = 16384

# Claude arms (run via `claude -p`, one headless session per decision — kept
# out of MODELS so the ollama matrix and the Claude pass are launched
# independently: ABLATION_MODELS=claude:sonnet,claude:haiku).
CLAUDE_MODELS: list[str] = ["claude:sonnet", "claude:haiku"]

# Non-LLM reference arms, executed through the same replay/gate path.
BASELINE_ARMS = ["closed_form", "random3", "all_applicable"]

RESULTS_PATH = "experiments/llm_ablation/results.jsonl"
TRANSCRIPT_DIR = "experiments/llm_ablation/transcripts"
