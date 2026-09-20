"""Minimal ollama client for the ablation harness.

Uses the native /api/chat endpoint with:
  - structured outputs (`format` = JSON schema) for Mode B decisions, which
    constrains decoding so even small models emit valid JSON;
  - optional native tool-calling (`tools`) for Mode A;
  - temperature 0 + fixed seed for reproducibility;
  - keep_alive=0 so the model unloads and never contends with the GAD
    detectors for GPU memory;
  - full request/response transcripts persisted as JSONL for the audit.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict[str, Any]]
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    # Native chain of thought, when the model emits one. ollama returns it in
    # message.thinking, separate from the JSON answer that `format` constrains.
    # It is the only place a small model's actual reasoning is visible, so it is
    # captured verbatim rather than summarised.
    thinking: str = field(repr=False, default="")


class OllamaClient:
    def __init__(self, model: str, host: str = DEFAULT_HOST,
                 transcript_path: str | None = None, timeout_s: float = 600.0,
                 seed: int = 1, temperature: float = 0.0, num_ctx: int = 16384):
        self.model = model
        # set by the runner before each call so a transcript line can be joined
        # to the decision it produced (dataset, repeat, arm)
        self.context: dict[str, Any] = {}
        self.host = host.rstrip("/")
        self.transcript_path = transcript_path
        self.seed = seed
        self.temperature = temperature
        self.num_ctx = num_ctx
        self._client = httpx.Client(timeout=timeout_s)

    # ------------------------------------------------------------------
    def chat(self, messages: list[dict[str, Any]],
             json_schema: dict[str, Any] | None = None,
             tools: list[dict[str, Any]] | None = None) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": 0,
            "options": {"temperature": self.temperature, "seed": self.seed,
                        "num_ctx": self.num_ctx},
        }
        # ABLATION_THINK=off/on toggles native thinking for models that support
        # it (qwen3, gemma4) — the deliberation ablation.
        think_env = os.environ.get("ABLATION_THINK")
        if think_env == "off":
            body["think"] = False
        elif think_env == "on":
            body["think"] = True
        if json_schema is not None:
            body["format"] = json_schema
        if tools is not None:
            body["tools"] = tools

        t0 = time.time()
        resp = self._client.post(f"{self.host}/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()
        latency = time.time() - t0

        msg = data.get("message", {})
        result = ChatResult(
            content=msg.get("content", "") or "",
            tool_calls=msg.get("tool_calls", []) or [],
            latency_s=round(latency, 2),
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(data.get("eval_count", 0) or 0),
            raw=data,
            thinking=msg.get("thinking", "") or "",
        )
        self._log({**self.context,
                   "model": self.model, "request": {k: v for k, v in body.items() if k != "options"},
                   "options": body["options"], "response_content": result.content,
                   "thinking": result.thinking,
                   "thinking_chars": len(result.thinking),
                   "tool_calls": result.tool_calls, "latency_s": result.latency_s,
                   "prompt_tokens": result.prompt_tokens,
                   "completion_tokens": result.completion_tokens,
                   "done_reason": data.get("done_reason"),
                   "eval_duration_s": round((data.get("eval_duration") or 0) / 1e9, 2),
                   "ts": time.time()})
        return result

    # ------------------------------------------------------------------
    def decide(self, messages: list[dict[str, Any]], json_schema: dict[str, Any],
               max_retries: int = 1) -> tuple[dict[str, Any] | None, ChatResult]:
        """Structured decision: returns (parsed_json_or_None, last_result).

        One retry with an explicit correction message when the constrained
        output still fails to parse (rare but observed on small models).
        """
        attempt_messages = list(messages)
        last: ChatResult | None = None
        for attempt in range(max_retries + 1):
            self.context = {**self.context, "attempt": attempt}
            last = self.chat(attempt_messages, json_schema=json_schema)
            try:
                return json.loads(last.content), last
            except (json.JSONDecodeError, ValueError):
                attempt_messages = attempt_messages + [
                    {"role": "assistant", "content": last.content},
                    {"role": "user", "content":
                        "Your previous reply was not valid JSON matching the schema. "
                        "Reply with ONLY the JSON object, no prose."},
                ]
        return None, last  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def _log(self, record: dict[str, Any]) -> None:
        if not self.transcript_path:
            return
        os.makedirs(os.path.dirname(self.transcript_path), exist_ok=True)
        with open(self.transcript_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        self._client.close()


def list_local_models(host: str = DEFAULT_HOST) -> list[str]:
    r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]
