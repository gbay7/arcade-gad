"""Claude Code headless provider for the ablation.

Each decision runs `claude -p` as its OWN subprocess/session (never the
interactive session driving the study), with the same evidence pack and the
same JSON decision contract as the ollama models. The CLI does not expose
temperature or seeds, so repeated calls measure Claude's natural decision
variability — which is precisely the consistency question.

Model aliases ("sonnet", "haiku", "opus") resolve to the current models on the
user's plan. Transcripts (prompt, raw reply, cost, latency) are persisted like
the ollama ones.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Any

from .ollama_client import ChatResult

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    text = _FENCE.sub("", text.strip()).strip()
    # With higher reasoning effort the model may wrap the JSON in prose —
    # extract the outermost object.
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


class ClaudeClient:
    """Mirrors OllamaClient's decide() interface for the runner."""

    def __init__(self, model: str, transcript_path: str | None = None,
                 timeout_s: float = 300.0, **_ignored: Any):
        self.model = model                     # "sonnet" | "haiku" | "opus" | full id
        self.transcript_path = transcript_path
        self.timeout_s = timeout_s
        # set by the runner so a transcript line joins to its decision
        self.context: dict[str, Any] = {}

    def decide(self, messages: list[dict[str, Any]], json_schema: dict[str, Any],
               max_retries: int = 1) -> tuple[dict[str, Any] | None, ChatResult]:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        prompt = (f"{system}\n\n{user}\n\n"
                  "Reply with ONLY a JSON object matching this schema (no prose, "
                  "no markdown fences):\n" + json.dumps(json_schema))

        parsed: dict[str, Any] | None = None
        result: ChatResult | None = None
        for _ in range(max_retries + 1):
            result = self._invoke(prompt)
            try:
                parsed = json.loads(_strip_fences(result.content))
                break
            except (json.JSONDecodeError, ValueError):
                prompt += "\n\nYour previous reply was not valid JSON. JSON object only."
        return parsed, result  # type: ignore[return-value]

    def _invoke(self, prompt: str) -> ChatResult:
        t0 = time.time()
        # Headless invocations fail sporadically when fired back-to-back
        # (transient CLI/session contention) — retry with backoff before
        # declaring the row a protocol failure.
        # stream-json + --verbose is the only output format that exposes the
        # assistant's content blocks. NOTE: the CLI returns thinking blocks whose
        # text is empty (signature only), so Claude's chain of thought is NOT
        # recoverable here; what is recorded is the block structure and the
        # thinking-token counts. Local models (ollama) do return the text.
        cmd = ["claude", "-p", prompt, "--model", self.model,
               "--output-format", "stream-json", "--verbose", "--max-turns", "1"]
        # ABLATION_EFFORT=high/low sets the CLI's reasoning-effort control
        # (extended reasoning where the model supports it).
        effort = os.environ.get("ABLATION_EFFORT")
        if effort:
            cmd += ["--effort", effort]
        for attempt in range(3):
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=self.timeout_s,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                break
            time.sleep(10 * (attempt + 1))
        latency = time.time() - t0
        content, usage, cost, model_usage = "", {}, None, {}
        blocks: list[dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if ev.get("type") == "assistant":
                for b in (ev.get("message", {}) or {}).get("content", []) or []:
                    blocks.append({"type": b.get("type"),
                                   "text": b.get("text") or b.get("thinking") or "",
                                   "chars": len(b.get("text") or b.get("thinking") or ""),
                                   "signed": bool(b.get("signature"))})
            elif ev.get("type") == "result":
                content = ev.get("result", "") or ""
                usage = ev.get("usage", {}) or {}
                cost = ev.get("total_cost_usd")
                # per-model accounting: canonical model id and thinking tokens
                model_usage = {k: {"canonical": v.get("canonicalModel"),
                                   "thinking_tokens": v.get("thinkingTokens"),
                                   "output_tokens": v.get("outputTokens")}
                               for k, v in (ev.get("modelUsage") or {}).items()}
        if not content:                      # stream unusable: fall back to the text blocks
            content = "\n".join(b["text"] for b in blocks if b["type"] == "text").strip()
        if not content:
            content = proc.stdout.strip()
        result = ChatResult(
            content=content, tool_calls=[], latency_s=round(latency, 2),
            prompt_tokens=int(usage.get("input_tokens", 0) or 0),
            completion_tokens=int(usage.get("output_tokens", 0) or 0),
            raw={"cost_usd": cost, "stderr": proc.stderr[-400:] if proc.returncode else ""},
        )
        self._log({**self.context,
                   "model": f"claude:{self.model}", "prompt": prompt,
                   "response_content": content,
                   "blocks": blocks,
                   "thinking_chars": sum(b["chars"] for b in blocks if b["type"] == "thinking"),
                   "thinking_text_available": any(b["type"] == "thinking" and b["chars"] for b in blocks),
                   "latency_s": result.latency_s,
                   "cost_usd": cost, "effort": effort, "model_usage": model_usage, "ts": time.time()})
        if proc.returncode != 0 and not content:
            raise RuntimeError(f"claude CLI failed: {proc.stderr[:200]}")
        return result

    def _log(self, record: dict[str, Any]) -> None:
        if not self.transcript_path:
            return
        os.makedirs(os.path.dirname(self.transcript_path), exist_ok=True)
        with open(self.transcript_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def close(self) -> None:  # interface parity with OllamaClient
        pass
