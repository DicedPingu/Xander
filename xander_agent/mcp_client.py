"""MCP to MCP: Xander speaks the protocol in both directions.

He already *serves* MCP (``xander mcp``); this module lets him *call*
other MCP servers — list their tools and invoke them over stdio. Remote
servers are declared once in ``mcp-servers.json`` in the config
directory::

    {"github": {"command": ["npx", "-y", "@modelcontextprotocol/server-github"]}}

Every call is bounded by a timeout and runs an isolated session: connect,
initialize, do one thing, disconnect. No long-lived subprocesses.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any


def _servers_path() -> Path:
    from .paths import config_dir

    return config_dir() / "mcp-servers.json"


def load_servers(path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = path or _servers_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    servers: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for name, spec in raw.items():
            if isinstance(spec, dict) and isinstance(spec.get("command"), list) and spec["command"]:
                servers[str(name)] = {
                    "command": [str(part) for part in spec["command"]],
                    "env": {str(k): str(v) for k, v in spec.get("env", {}).items()},
                }
    return servers


def add_server(name: str, command: list[str], *, env: dict[str, str] | None = None, path: Path | None = None) -> None:
    if not name or not command:
        raise ValueError("a server needs a name and a command")
    path = path or _servers_path()
    servers = load_servers(path)
    servers[name] = {"command": command, "env": env or {}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(servers, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def _with_session(spec: dict[str, Any], operation: Any, timeout: int) -> Any:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    command = spec["command"]
    parameters = StdioServerParameters(command=command[0], args=command[1:], env=spec.get("env") or None)

    async def run() -> Any:
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await operation(session)

    return await asyncio.wait_for(run(), timeout=timeout)


def _resolve(server: str, path: Path | None) -> dict[str, Any]:
    servers = load_servers(path)
    if server not in servers:
        known = ", ".join(sorted(servers)) or "none configured"
        raise ValueError(f"unknown MCP server {server!r}; known: {known}")
    return servers[server]


def list_remote_tools(server: str, *, path: Path | None = None, timeout: int = 30) -> list[dict[str, str]]:
    async def operation(session: Any) -> list[dict[str, str]]:
        result = await session.list_tools()
        return [
            {"name": tool.name, "description": (tool.description or "")[:300]}
            for tool in result.tools
        ]

    return asyncio.run(_with_session(_resolve(server, path), operation, timeout))


def call_remote_tool(
    server: str,
    tool: str,
    arguments: dict[str, Any] | None = None,
    *,
    path: Path | None = None,
    timeout: int = 60,
) -> str:
    async def operation(session: Any) -> str:
        result = await session.call_tool(tool, arguments or {})
        parts = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)[:24_000]

    return asyncio.run(_with_session(_resolve(server, path), operation, timeout))
