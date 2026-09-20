"""Tool registry — discovers and dispatches all GUIDE tools."""

from __future__ import annotations

import time
import traceback
from typing import Any, Callable

from arcade.models import ToolResult, FailResult, ToolCategory


class ToolSpec:
    def __init__(
        self,
        name: str,
        category: ToolCategory,
        description: str,
        handler: Callable,
        input_schema: dict[str, Any] | None = None,
        timeout: int = 120,
        cacheable: bool = True,
    ):
        self.name = name
        self.category = category
        self.description = description
        self.handler = handler
        self.input_schema = input_schema or {}
        self.timeout = timeout
        self.cacheable = cacheable


class ToolRegistry:
    """Central registry for all tools — built-in and agent-composed."""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self._cache: dict[str, ToolResult] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self, category: ToolCategory | None = None) -> list[dict[str, Any]]:
        tools = self._tools.values()
        if category:
            tools = [t for t in tools if t.category == category]
        return [
            {
                "name": t.name,
                "category": t.category.value,
                "description": t.description,
            }
            for t in sorted(tools, key=lambda t: (t.category.value, t.name))
        ]

    def execute(self, name: str, **params: Any) -> ToolResult | FailResult:
        spec = self._tools.get(name)
        if spec is None:
            return FailResult(error=f"Unknown tool: {name}", suggestion=f"Available: {list(self._tools.keys())}")

        cache_key = f"{name}:{sorted(params.items())}" if spec.cacheable else ""
        if cache_key and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.metadata["cache_hit"] = True
            return cached

        start = time.time()
        try:
            result = spec.handler(**params)
            elapsed = time.time() - start

            if not isinstance(result, (ToolResult, FailResult)):
                result = ToolResult(
                    data=result if isinstance(result, dict) else {"result": result},
                    summary=str(result)[:500],
                )

            result.metadata["exec_time"] = round(elapsed, 3)
            result.metadata["cache_hit"] = False
            result.provenance = {"tool": name, "params": _safe_params(params)}

            if cache_key:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            elapsed = time.time() - start
            return FailResult(
                error=f"{type(e).__name__}: {e}",
                suggestion=f"Tool '{name}' failed after {elapsed:.1f}s. Check parameters or try with different settings.",
            )

    def clear_cache(self) -> int:
        n = len(self._cache)
        self._cache.clear()
        return n


def _safe_params(params: dict) -> dict:
    """Make params JSON-serializable for provenance."""
    safe = {}
    for k, v in params.items():
        if hasattr(v, 'tolist'):
            safe[k] = f"<array shape={getattr(v, 'shape', '?')}>"
        elif callable(v):
            safe[k] = f"<callable {getattr(v, '__name__', '?')}>"
        else:
            safe[k] = v
    return safe
