"""A hidden note Xander leaves in a folder he worked in, for whoever comes next.

The next agent to open this directory starts with nothing: no task record, no
memory, no idea which of these files matter or how to run them. That is a
solvable waste. This writes one dot-file describing what the project is, how to
build and run it, what actually works, and what is still broken.

It is deliberately *not* an agent log. Task records, evidence and lessons stay in
Xander's own storage. This is documentation about the project, written for a
reader — which is why it is allowed in a workspace that Xander does not own.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

FILENAME = ".ai-context.md"
_HEADER = "<!-- Written by Xander for the next agent. Safe to edit or delete. -->"


def _fence(text: str) -> str:
    return str(text).replace("`", "'").strip()


def compose(
    *,
    workspace: Path,
    goal: str,
    status: str,
    files: list[dict[str, Any]],
    commands: list[str],
    open_problems: list[str],
    lesson: str = "",
    task_id: str = "",
) -> str:
    """Build the note from what actually happened, not from what was intended."""

    lines = [
        _HEADER,
        f"# {workspace.name}",
        "",
        f"**What this is:** {_fence(goal) or 'unrecorded'}",
        f"**State:** {_fence(status) or 'unknown'}  ",
        f"**Last touched:** {datetime.now().isoformat(timespec='seconds')}"
        + (f" · task `{_fence(task_id)}`" if task_id else ""),
        "",
    ]

    if files:
        lines += ["## Files", ""]
        for item in files:
            name = _fence(item.get("path", "?"))
            note = _fence(item.get("note", ""))
            flag = "  ⚠ still a placeholder" if item.get("placeholder") else ""
            lines.append(f"- `{name}`{' — ' + note if note else ''}{flag}")
        lines.append("")

    if commands:
        lines += ["## How to build and run", "", "```bash"]
        lines += [_fence(command) for command in commands]
        lines += ["```", ""]

    if open_problems:
        lines += ["## Known problems", "", *[f"- {_fence(item)}" for item in open_problems], ""]
    else:
        lines += ["## Known problems", "", "- None recorded. Verify before trusting that.", ""]

    if lesson:
        lines += ["## Worth knowing", "", _fence(lesson), ""]

    lines += [
        "## For the next agent",
        "",
        "- Check the claims above against the files before relying on them.",
        "- Anything marked a placeholder is not implemented, whatever its name suggests.",
        "- Update this file when you change what is true.",
        "",
    ]
    return "\n".join(lines)


def write(workspace: Path, body: str, *, filename: str = FILENAME) -> Path | None:
    """Write the note. Never raises; a failed note must not fail a run."""

    try:
        target = (workspace / filename).resolve()
        target.relative_to(workspace.resolve())
    except (OSError, ValueError):
        return None
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".md.tmp")
        temporary.write_text(body, encoding="utf-8")
        os.chmod(temporary, 0o644)  # readable: the point is that others read it
        temporary.replace(target)
        return target
    except OSError:
        return None


def from_task(workspace: Path, task: Any, written: list[dict[str, Any]]) -> str:
    """Compose the note from a finished task record."""

    status = getattr(getattr(task, "status", None), "value", "") or "unknown"
    request = getattr(task, "request", None)
    goal = getattr(request, "goal", "") if request else ""

    commands: list[str] = []
    for check in getattr(request, "acceptance_checks", []) or []:
        argv = getattr(check, "argv", None) or []
        if argv:
            commands.append(" ".join(str(part) for part in argv))
    for result in getattr(task, "results", []) or []:
        argv = getattr(result, "argv", None) or []
        if argv and getattr(result, "status", None) and str(result.status) == "ok":
            joined = " ".join(str(part) for part in argv)
            if joined not in commands:
                commands.append(joined)

    problems: list[str] = []
    failure = str(getattr(task, "failure", "") or "")
    if failure:
        problems.append(failure)
    for item in written:
        if item.get("placeholder"):
            problems.append(f"{item.get('path')} is still a placeholder, not an implementation")
    for check in getattr(task, "check_results", []) or []:
        if str(getattr(check, "status", "")) != "ok":
            reason = str(getattr(check, "reason", "") or getattr(check, "name", "") or "a check failed")
            problems.append(f"check failed: {reason}")

    return compose(
        workspace=workspace,
        goal=goal,
        status=status,
        files=written,
        commands=commands[:8],
        open_problems=problems[:8],
        lesson=str(getattr(task, "lesson", "") or ""),
        task_id=str(getattr(task, "id", "")),
    )
