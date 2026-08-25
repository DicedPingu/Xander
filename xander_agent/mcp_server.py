"""Stdio MCP sidecar used by Codex and Claude.

All sidecar work is proposal-only: callers receive structured evidence and
retain authority over edits and final judgment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from . import __version__
from .cli import create_engine, doctor_payload, invoke_engine

Caller = Literal["codex", "claude", "human"]


def _failure(exc: Exception) -> dict[str, Any]:
    return {"ok": False, "error": str(exc), "type": type(exc).__name__}


def _call(
    mode: str,
    *,
    workspace: str,
    goal: str,
    caller: Caller,
    variant: str,
    constraints: list[str] | None = None,
    acceptance_checks: list[str] | None = None,
    allowed_paths: list[str] | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    try:
        return invoke_engine(
            mode,
            workspace=Path(workspace),
            goal=goal,
            variant=variant,
            autonomy="proposal-only",
            caller=caller,
            constraints=constraints or (),
            acceptance_checks=acceptance_checks or (),
            allowed_paths=allowed_paths or (),
            timeout=timeout,
        )
    except Exception as exc:
        return _failure(exc)


def create_server() -> FastMCP:
    mcp = FastMCP(
        "Xander",
        instructions=(
            "Use Xander for local inspection, documentation research, planning, patch proposals, "
            "and test triage. Always provide an absolute workspace and acceptance checks. "
            "Xander is proposal-only over MCP; the calling agent owns edits and final judgment."
        ),
    )

    @mcp.tool(structured_output=True)
    def xander_doctor(workspace: str = ".") -> dict[str, Any]:
        """Check Xander's local engine, models, state paths, and interface versions."""

        report = doctor_payload(Path(workspace))
        report["mcp_server_version"] = __version__
        return report

    @mcp.tool(structured_output=True)
    def xander_inspect(
        workspace: str,
        goal: str,
        caller: Caller = "codex",
        variant: str = "default",
        constraints: list[str] | None = None,
        allowed_paths: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Inspect repository state without modifying it."""

        return _call(
            "inspect",
            workspace=workspace,
            goal=goal,
            caller=caller,
            variant=variant,
            constraints=constraints,
            allowed_paths=allowed_paths,
            timeout=timeout,
        )

    @mcp.tool(structured_output=True)
    def xander_research(
        workspace: str,
        goal: str,
        caller: Caller = "codex",
        variant: str = "default",
        constraints: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Research a coding decision using local truth and configured current-doc sources."""

        return _call(
            "research",
            workspace=workspace,
            goal=goal,
            caller=caller,
            variant=variant,
            constraints=constraints,
            timeout=timeout,
        )

    @mcp.tool(structured_output=True)
    def xander_plan(
        workspace: str,
        goal: str,
        acceptance_checks: list[str],
        caller: Caller = "codex",
        variant: str = "default",
        constraints: list[str] | None = None,
        allowed_paths: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Return an evidence-backed plan with explicit acceptance checks."""

        return _call(
            "plan",
            workspace=workspace,
            goal=goal,
            caller=caller,
            variant=variant,
            constraints=constraints,
            acceptance_checks=acceptance_checks,
            allowed_paths=allowed_paths,
            timeout=timeout,
        )

    @mcp.tool(structured_output=True)
    def xander_propose_patch(
        workspace: str,
        goal: str,
        acceptance_checks: list[str],
        caller: Caller = "codex",
        variant: str = "default",
        constraints: list[str] | None = None,
        allowed_paths: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Produce a validated unified-diff proposal without applying it."""

        return _call(
            "implement",
            workspace=workspace,
            goal=goal,
            caller=caller,
            variant=variant,
            constraints=constraints,
            acceptance_checks=acceptance_checks,
            allowed_paths=allowed_paths,
            timeout=timeout,
        )

    @mcp.tool(structured_output=True)
    def xander_triage_tests(
        workspace: str,
        goal: str,
        acceptance_checks: list[str],
        caller: Caller = "codex",
        variant: str = "default",
        constraints: list[str] | None = None,
        allowed_paths: list[str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Run or interpret focused tests and return evidence and likely remedies."""

        return _call(
            "test-triage",
            workspace=workspace,
            goal=goal,
            caller=caller,
            variant=variant,
            constraints=constraints,
            acceptance_checks=acceptance_checks,
            allowed_paths=allowed_paths,
            timeout=timeout,
        )

    @mcp.tool(structured_output=True)
    def xander_tasks_list(
        workspace: str,
        caller: Caller = "codex",
        variant: str = "default",
    ) -> dict[str, Any]:
        """List persisted Xander tasks for a workspace."""

        try:
            engine = create_engine(Path(workspace), variant=variant, autonomy="proposal-only")
            return {"ok": True, "tasks": engine.list_tasks(), "caller": caller}
        except Exception as exc:
            return _failure(exc)

    @mcp.tool(structured_output=True)
    def xander_resume(
        workspace: str,
        task_id: str,
        selected_options: list[str] | None = None,
        caller: Caller = "codex",
        variant: str = "default",
    ) -> dict[str, Any]:
        """Resume a waiting task after selecting a compatible set of option ids."""

        try:
            engine = create_engine(Path(workspace), variant=variant, autonomy="proposal-only")
            result = engine.resume(task_id, selected_options=selected_options or ())
            result["caller"] = caller
            return result
        except Exception as exc:
            return _failure(exc)

    @mcp.tool(structured_output=True)
    def xander_task_show(
        workspace: str,
        task_id: str,
        caller: Caller = "codex",
        variant: str = "default",
    ) -> dict[str, Any]:
        """Read one persisted task, including phases, actions, and evidence."""

        try:
            engine = create_engine(Path(workspace), variant=variant, autonomy="proposal-only")
            return {"ok": True, "task": engine.show_task(task_id), "caller": caller}
        except Exception as exc:
            return _failure(exc)

    @mcp.resource("xander://tasks/{task_id}", mime_type="application/json")
    def xander_task_resource(task_id: str) -> str:
        """Read a task from Xander's configured current workspace."""

        workspace = Path.cwd()
        try:
            task = create_engine(workspace, autonomy="proposal-only").show_task(task_id)
            return json.dumps(task, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:
            return json.dumps(_failure(exc), ensure_ascii=False)

    return mcp


server = create_server()


def run_stdio() -> None:
    server.run(transport="stdio")


def entrypoint() -> None:
    run_stdio()


if __name__ == "__main__":
    entrypoint()
