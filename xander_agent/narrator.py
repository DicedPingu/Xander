"""Human-readable narration for Xander's structured event stream.

Turns ``XanderEvent`` payloads into short, lively log lines — Monica's
glyph-and-color voice adapted to a task loop whose shape changes per task.
Each narrated line is also appended, markup-free, to plain session logs under
Xander's state directory so a ``tail -f`` follows along outside the TUI.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.markup import escape

from . import ABILITIES

# event type -> (glyph, rich style, channel)
_STYLES: dict[str, tuple[str, str, str]] = {
    "task": ("◆", "bold #82aaff", "run"),
    "phase": ("»", "bold magenta", "run"),
    "plan": ("▤", "bold #c3e88d", "plan"),
    "research": ("·", "cyan", "research"),
    "action": ("▸", "#ffcb6b", "run"),
    "patch": ("±", "bold #c792ea", "diff"),
    "test": ("⚑", "bold cyan", "tests"),
    "approval": ("?", "bold yellow", "run"),
    "result": ("✓", "bold green", "run"),
    "error": ("✗", "bold red", "run"),
    "voice": ("❝", "italic #f78c6c", "run"),
}

# mantra phase -> the ability Xander is leaning on right now
PHASE_ABILITY: dict[str, str] = {
    "analyze": "llm",
    "research": "oracle",
    "set_up": "quartermaster",
    "work": "agent",
    "test": "algorithm",
    "judge_log": "algorithm",
    "learn": "scribe",
    "repeat": "bot",
}

STATUS_GLYPHS: dict[str, tuple[str, str]] = {
    "completed": ("✓", "bold green"),
    "running": ("…", "cyan"),
    "pending": ("·", "dim"),
    "waiting_approval": ("?", "bold yellow"),
    "unverified": ("!", "bold yellow"),
    "failed": ("✗", "bold red"),
    "interrupted": ("‖", "dim"),
}

_TAIL = 80
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _short(text: str, limit: int = _TAIL) -> str:
    text = _CONTROL_RE.sub("", " ".join(str(text).split()))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _log_dir() -> Path:
    from .paths import state_dir

    path = state_dir() / "logs"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _digest(event_type: str, data: dict[str, Any]) -> list[tuple[str, str]]:
    """Pick the few fields worth reading for this event type."""

    fields: list[tuple[str, str]] = []
    if event_type == "task":
        if data.get("status"):
            fields.append(("status", str(data["status"])))
        if data.get("workspace"):
            fields.append(("here", str(data["workspace"])))
    elif event_type == "plan":
        fields.append(("actions", str(data.get("actions", 0))))
        fields.append(("checks", str(data.get("checks", 0))))
        options = data.get("options") or []
        if options:
            fields.append(("options", ",".join(str(item) for item in options[:4])))
    elif event_type == "research":
        sources = data.get("sources") or []
        skills = [str(name) for name in (data.get("skills") or []) if name]
        tools = data.get("tools") or []
        fields.append(("sources", str(len(sources))))
        if skills:
            fields.append(("skills", ",".join(skills[:3])))
        if tools:
            fields.append(("tools", str(len(tools))))
        warnings = data.get("warnings") or []
        if warnings:
            fields.append(("warning", _short(str(warnings[0]), 100)))
    elif event_type in {"action", "patch", "test"}:
        result = data.get("result")
        if isinstance(result, dict):
            if result.get("returncode") is not None:
                fields.append(("rc", str(result["returncode"])))
            changed = result.get("changed_paths") or []
            if changed:
                fields.append(("paths", ",".join(Path(p).name for p in changed[:3])))
            detail = result.get("reason") or ""
            if not detail and result.get("status") not in {"ok", None}:
                detail = (result.get("stderr") or "").strip().splitlines()[-1:] or [""]
                detail = detail[0]
            if detail:
                fields.append(("detail", _short(detail)))
            if result.get("status") == "ok" and result.get("returncode") is None and result.get("stdout"):
                fields.append(("note", _short(str(result["stdout"]), 120)))
        action = data.get("action")
        if isinstance(action, dict) and action.get("risk") in {"high", "critical"}:
            fields.append(("risk", str(action["risk"])))
        if event_type == "test" and data.get("name"):
            fields.insert(0, ("check", _short(str(data["name"]), 40)))
    elif event_type == "approval":
        options = data.get("options") or []
        if options:
            ids = [str(item.get("id", "?")) if isinstance(item, dict) else str(item) for item in options]
            fields.append(("options", ",".join(ids[:6])))
        selected = data.get("selected") or []
        if selected:
            fields.append(("selected", ",".join(str(item) for item in selected)))
    elif event_type in {"result", "error"}:
        if data.get("task_id"):
            fields.append(("task", str(data["task_id"])))
        if data.get("status"):
            fields.append(("status", str(data["status"])))
    elif event_type == "voice":
        if data.get("moment"):
            fields.append(("moment", str(data["moment"])))
    return fields


class Narrator:
    """Session-scoped renderer: markup for the TUI, plain lines for files."""

    def __init__(
        self,
        *,
        variant: str = "default",
        workspace: Path | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.variant = variant
        self.workspace = workspace.expanduser().resolve() if workspace else None
        self.started = time.monotonic()
        try:
            directory = log_dir or _log_dir()
            self._task_log_dir = directory / "tasks"
            self._task_log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._files: tuple[Path, ...] = tuple({directory / "xander.log", directory / f"{variant}.log"})
        except OSError:
            self._task_log_dir = None
            self._files = ()

    # -- rendering -----------------------------------------------------------
    def narrate(self, payload: dict[str, Any]) -> tuple[str, str]:
        """Return ``(channel, markup_line)`` and append the plain twin to disk."""

        event_type = str(payload.get("type") or payload.get("event") or "action")
        glyph, style, channel = _STYLES.get(event_type, ("▸", "#ffcb6b", "run"))
        message = _short(str(payload.get("message", "")), 200 if event_type == "voice" else 160)
        raw_data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if event_type == "voice" and raw_data.get("speaker"):
            message = f"{_short(str(raw_data['speaker']), 24)} · {message}"
        phase = str(payload.get("phase") or "")
        ability = PHASE_ABILITY.get(phase, "")
        attempt = int(payload.get("attempt") or 0)
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        run_id = _short(str(payload.get("run_id") or ""), 100)
        if event_type == "result" and not data.get("ok", True):
            glyph, style = "!", "bold yellow"
        fields = [(key, _short(value)) for key, value in _digest(event_type, data)]

        elapsed = f"+{time.monotonic() - self.started:6.1f}s"
        parts = [f"[dim]{elapsed}[/]", f"[{style}]{glyph}[/]"]
        if attempt > 1:
            parts.append(f"[dim]a{attempt}[/]")
        if event_type == "phase" and phase:
            parts.append(f"[{style}]{escape(phase)}[/]")
        if ability:
            parts.append(f"[dim]({ability})[/]")
        parts.append(escape(message))
        if fields:
            rendered = " ".join(f"{key}={escape(value)}" for key, value in fields)
            parts.append(f"[dim]| {rendered}[/]")
        line = " ".join(part for part in parts if part)

        plain_fields = "".join(f" {key}={value}" for key, value in fields)
        prefix = f"{phase}:" if phase else ""
        self._write_plain(f"[{event_type.upper()}] {prefix}{message}{plain_fields}", task_id=run_id)
        return channel, line

    def summarize(self, result: dict[str, Any]) -> list[str]:
        """Compact end-of-task report: verdict, evidence, lesson, next step."""

        task = result.get("task") if isinstance(result.get("task"), dict) else {}
        handoff = result.get("handoff") if isinstance(result.get("handoff"), dict) else {}
        status = str(result.get("status") or task.get("status") or handoff.get("status") or "unknown")
        ok = bool(result.get("ok")) or status.casefold() in {"complete", "completed"}
        task_id = str(result.get("task_id") or task.get("id") or handoff.get("task_id") or "?")
        glyph, style = STATUS_GLYPHS.get(status, ("!", "bold yellow"))

        changed = sorted(
            {
                path
                for item in (task.get("results") or handoff.get("results") or [])
                if isinstance(item, dict)
                for path in item.get("changed_paths") or []
            }
        )
        checks = [item for item in task.get("check_results") or handoff.get("checks") or [] if isinstance(item, dict)]
        passed = sum(1 for item in checks if item.get("status") == "ok")
        results = [item for item in task.get("results") or handoff.get("results") or [] if isinstance(item, dict)]
        actions_ok = sum(1 for item in results if item.get("status") == "ok")

        lines = [f"[{style}]{glyph} {escape(status)}[/] [dim]task {escape(task_id)} · attempt {task.get('attempt', 0)}[/]"]
        if results:
            lines.append(f"  did {actions_ok}/{len(results)} planned action(s)")
        if changed:
            shown = ", ".join(changed[:5])
            more = f" (+{len(changed) - 5})" if len(changed) > 5 else ""
            lines.append(f"  changed {len(changed)} path(s): {escape(shown)}{more}")
        if checks:
            check_style = "green" if passed == len(checks) else "yellow"
            lines.append(f"  checks [{check_style}]{passed}/{len(checks)} passed[/]")
        lesson = str(task.get("lesson") or "")
        if lesson:
            lines.append(f"  [dim](scribe)[/] lesson: {escape(_short(lesson, 140))}")
        failure = str(result.get("error") or task.get("failure") or "")
        if not ok and failure:
            lines.append(f"  [bold red]blocked:[/] {escape(_short(failure, 140))}")
        lines.append(f"  [dim]full record: xander tasks show {escape(task_id)}[/]")

        for line in lines:
            self._write_plain(f"[SUMMARY] {_strip_markup(line)}", task_id=task_id)
        return lines

    def record(self, message: str, *, task_id: str = "") -> None:
        self._write_plain(f"[GUIDANCE] {_short(message, 500)}", task_id=task_id)

    # -- plain file twin -----------------------------------------------------
    def _write_plain(self, body: str, *, task_id: str = "") -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        workspace = str(self.workspace) if self.workspace else "unknown"
        task = task_id or "session"
        line = f"[{stamp}] [{self.variant}] [workspace={workspace}] [task={task}] {body}\n"
        targets = list(self._files)
        if self._task_log_dir is not None and re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            targets.append(self._task_log_dir / f"{task_id}.log")
        for target in targets:
            try:
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                continue


def _strip_markup(text: str) -> str:
    from rich.text import Text

    return Text.from_markup(text).plain


def abilities_line() -> str:
    """One-line identity string for status bars and doctor output."""

    return " · ".join(ABILITIES)
