"""Thin MCP client for the ARCADE server (stdio).

This is the ONLY path the ablation harness uses to reach ARCADE: the same MCP
protocol any agent (Claude, or a local ollama model through our loop) speaks.
Nothing is imported from the arcade package in-process — the server runs as a
subprocess and every interaction goes through tool calls, so a local model sees
exactly the context (tool names, descriptions, schemas, results) an MCP agent
sees.

Run as a script for the Phase-0 handshake check:
    python -m experiments.llm_ablation.mcp_client
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


class ArcadeMCP:
    """Synchronous-feeling wrapper around an async MCP stdio session."""

    def __init__(self, python_exe: str = sys.executable):
        self._params = StdioServerParameters(
            command=python_exe,
            args=["-m", "arcade.server"],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": REPO_ROOT},
        )
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def __aenter__(self) -> "ArcadeMCP":
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(self._params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._stack is not None
        await self._stack.aclose()

    async def list_tools(self) -> list[ToolInfo]:
        assert self._session is not None
        result = await self._session.list_tools()
        return [ToolInfo(t.name, t.description or "", t.inputSchema) for t in result.tools]

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Call a tool; return parsed JSON when the result is JSON, else raw text."""
        assert self._session is not None
        result = await self._session.call_tool(name, arguments or {})
        texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
        payload = "\n".join(texts)
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return payload


async def _handshake() -> int:
    """Phase-0 check: can a plain MCP client obtain ARCADE's context and drive
    the label-free pipeline? Exercises list_tools + load/profile/applicable."""
    async with ArcadeMCP() as arcade:
        tools = await arcade.list_tools()
        names = [t.name for t in tools]
        print(f"[handshake] tools exposed: {len(tools)}")
        required = ["load_dataset", "graph_profile", "applicable_paradigms",
                    "run_paradigm", "compute_weights", "evaluate_against_labels"]
        missing = [r for r in required if r not in names]
        if missing:
            print(f"[handshake] MISSING required tools: {missing}")
            return 1
        print(f"[handshake] required tools present: {required}")

        info = await arcade.call("load_dataset", {"name": "inj_cora"})
        print(f"[handshake] load_dataset -> n={info.get('num_nodes')} dim={info.get('feature_dim')}")
        prof = await arcade.call("graph_profile")
        pdata = prof.get("data", prof)
        print(f"[handshake] graph_profile -> avg_degree={pdata.get('avg_degree')} "
              f"homophily={pdata.get('homophily')} sparsity={pdata.get('feature_sparsity')}")
        appl = await arcade.call("applicable_paradigms")
        print(f"[handshake] applicable_paradigms -> {[p['name'] for p in appl]}")

        # The context a local model would receive: tool descriptions are non-empty.
        undocumented = [t.name for t in tools if not t.description.strip()]
        print(f"[handshake] tools without description: {undocumented or 'none'}")
    print("HANDSHAKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_handshake()))
