"""opencode headless provider (e.g. Kimi K2 via OpenRouter's free tier).

Each decision runs `opencode run` as its own subprocess/session with the same
evidence pack and JSON contract as the other providers. opencode 1.4.x prints
formatted text (no JSON output mode), so the reply is ANSI-stripped and the
JSON object extracted; retries mirror the Claude client.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from typing import Any

from .ollama_client import ChatResult

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json(text: str) -> str:
    text = _FENCE.sub("", _ANSI.sub("", text)).strip()
    # take the outermost JSON object in the reply
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text


class OpencodeClient:
    """Mirrors OllamaClient's decide() interface for the runner."""

    def __init__(self, model: str, transcript_path: str | None = None,
                 timeout_s: float = 300.0, **_ignored: Any):
        self.model = model                     # full "provider/model" id
        self.transcript_path = transcript_path
        self.timeout_s = timeout_s

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
                parsed = json.loads(_extract_json(result.content))
                break
            except (json.JSONDecodeError, ValueError):
                prompt += "\n\nYour previous reply was not valid JSON. JSON object only."
        return parsed, result  # type: ignore[return-value]

    def _invoke(self, prompt: str) -> ChatResult:
        t0 = time.time()
        for attempt in range(3):
            proc = subprocess.run(
                ["opencode", "run", prompt, "--model", self.model],
                capture_output=True, text=True, timeout=self.timeout_s,
            )
            body = _ANSI.sub("", proc.stdout or "")
            if proc.returncode == 0 and "{" in body and "Error:" not in body:
                break
            time.sleep(10 * (attempt + 1))
        latency = time.time() - t0
        content = _ANSI.sub("", proc.stdout or "").strip()
        result = ChatResult(content=content, tool_calls=[], latency_s=round(latency, 2),
                            prompt_tokens=0, completion_tokens=0,
                            raw={"stderr": (proc.stderr or "")[-300:] if proc.returncode else ""})
        self._log({"model": f"opencode:{self.model}", "prompt": prompt[:2000],
                   "response_content": content[:4000], "latency_s": result.latency_s,
                   "ts": time.time()})
        if proc.returncode != 0 and "{" not in content:
            raise RuntimeError(f"opencode failed: {(proc.stderr or content)[:200]}")
        return result

    def _log(self, record: dict[str, Any]) -> None:
        if not self.transcript_path:
            return
        os.makedirs(os.path.dirname(self.transcript_path), exist_ok=True)
        with open(self.transcript_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def close(self) -> None:
        pass
