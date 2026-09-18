from __future__ import annotations

import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_TASK_ID_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_MAX_LOG_BYTES = 5 * 1024 * 1024
_LOG_LOCK = threading.Lock()


def _clean(value: Any, limit: int = 1_000) -> str:
    text = _CONTROL_RE.sub(" ", str(value))
    text = " ".join(text.split())
    return text[:limit]


def quick_event_text(
    event_type: str,
    message: Any,
    data: dict[str, Any] | None = None,
) -> str | None:
    """Project one structured event into the small log a human can scan."""

    kind = _clean(event_type, 40).casefold()
    text = _clean(message, 240)
    fields = data or {}
    result = fields.get("result") if isinstance(fields.get("result"), dict) else {}
    action = fields.get("action") if isinstance(fields.get("action"), dict) else {}
    if kind == "task":
        return f"[MISSION] {text}"
    if kind == "guide":
        guide = fields.get("guide") if isinstance(fields.get("guide"), dict) else {}
        progress = _clean(guide.get("progress") or "loop created", 80)
        current = _clean(guide.get("current") or text, 180)
        return f"[LOOP] {progress} - {current}"
    if kind == "plan":
        decision = _clean(fields.get("decision") or text, 220)
        why = _clean(fields.get("why") or "", 220)
        counts = f"{fields.get('actions', 0)} steps, {fields.get('checks', 0)} checks"
        return f"[DECISION] {decision}" + (f" - why: {why}" if why else "") + f" ({counts})"
    if kind == "research":
        warnings = fields.get("warnings") or []
        sources = fields.get("sources") or []
        if not warnings and not sources:
            return None
        detail = f"{len(sources)} relevant source{'s' if len(sources) != 1 else ''}"
        if warnings:
            detail += f" - warning: {_clean(warnings[0], 180)}"
        return f"[RESEARCH] {text} - {detail}"
    if kind == "delegation":
        status = _clean(fields.get("status") or "", 40).casefold()
        error = _clean(fields.get("error") or "", 180)
        outcome = _clean(fields.get("outcome") or "", 80).casefold()
        if not error and status not in {"failed", "blocked"} and outcome not in {"failed", "blocked"}:
            return None
        agent = _clean(fields.get("agent") or "support agent", 60)
        return f"[SUPPORT BLOCKED] {agent} - {error or text}"
    if kind in {"action", "patch"}:
        wrote = fields.get("wrote") if isinstance(fields.get("wrote"), list) else []
        if wrote:
            described = []
            for item in wrote[:6]:
                if not isinstance(item, dict):
                    continue
                name = _clean(item.get("path", "?"), 120)
                if "lines" not in item:
                    described.append(name)
                elif item.get("placeholder"):
                    described.append(f"{name} ({item['lines']} lines - STILL A PLACEHOLDER)")
                else:
                    described.append(f"{name} ({item['lines']} lines)")
            if described:
                return "[WROTE] " + ", ".join(described)
        changed = [_clean(path, 180) for path in result.get("changed_paths") or []]
        status = _clean(result.get("status") or "", 60).casefold()
        if changed:
            return f"[CHANGED] {', '.join(changed[:6])}"
        if status and status != "ok":
            reason = _clean(result.get("reason") or result.get("stderr") or text, 240)
            return f"[STEP {status.upper()}] {reason}"
        if action.get("risk") in {"high", "critical"}:
            return f"[RISK] {_clean(action.get('expected') or text, 220)}"
        return None
    if kind == "logic_change":
        reason = _clean(fields.get("reason") or text, 220)
        replacement = _clean(
            fields.get("decision") or fields.get("replacement_summary") or fields.get("summary") or "",
            220,
        )
        return f"[APPROACH CHANGED] {reason}" + (f" - next: {replacement}" if replacement else "")
    if kind == "test":
        if not result:
            return None
        name = _clean(fields.get("name") or text or "acceptance check", 180)
        status = _clean(result.get("status") or text or "recorded", 60)
        return f"[PROOF {status.upper()}] {name}"
    if kind == "approval":
        return f"[DECISION NEEDED] {text}"
    if kind == "result":
        status = _clean(fields.get("status") or "recorded", 60)
        return f"[RESULT {status.upper()}] {text}"
    if kind == "error":
        return f"[BLOCKED] {text}"
    if kind == "learn":
        progress = _clean(fields.get("progress") or "", 60)
        return f"[LEARN] {text}" + (f" ({progress})" if progress else "")
    if kind == "steering":
        # What the operator said mid-run and what it actually changed. This is
        # the audit trail for "why did he suddenly switch to that?".
        moment = _clean(fields.get("moment") or "", 40)
        return f"[STEER] {text}" + (f" [at {moment}]" if moment else "")
    if kind == "phase":
        # The spine of "what is he doing right now". Dropping these left a
        # supervised run with five log lines and no visible progress at all.
        phase = _clean(fields.get("phase") or "", 40) or "working"
        return f"[PHASE] {phase}:{text}" if text else f"[PHASE] {phase}"
    if kind == "voice":
        # ...and "what is he thinking". Only `review` survived before, so the
        # Master's verdicts, setbacks and course corrections were all invisible.
        moment = _clean(fields.get("moment") or "", 40).casefold()
        if not text or "no result" in text.casefold():
            return None
        speaker = _clean(fields.get("speaker") or "", 40)
        label = {"review": "REVIEW", "verdict": "VERDICT", "setback": "SETBACK"}.get(moment, "THINKING")
        return f"[{label}] {speaker + ' - ' if speaker else ''}{text}"
    return None


def append_global_event(
    variant: str,
    workspace: Path,
    task_id: str,
    event_type: str,
    message: Any,
    data: dict[str, Any] | None = None,
) -> None:
    from .paths import logs_dir

    directory = logs_dir()
    fields = data or {}
    body = quick_event_text(event_type, message, fields)
    if body is None:
        return
    stamp = datetime.now().isoformat(timespec="seconds")
    workspace_text = _clean(workspace.expanduser().resolve(strict=False), 500)
    task_text = task_id if _TASK_ID_RE.fullmatch(task_id) else "session"
    variant_text = re.sub(r"[^A-Za-z0-9_.-]", "_", _clean(variant, 80)) or "default"
    line = f"[{stamp}] [{variant_text}] [workspace={workspace_text}] [task={task_text}] {body}\n"
    targets = [directory / "xander.log", directory / f"{variant_text}.log"]
    if task_text != "session":
        targets.append(directory / "tasks" / f"{task_text}.log")
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        for target in dict.fromkeys(targets):
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with _LOG_LOCK:
                if target.exists() and target.stat().st_size + len(line.encode("utf-8")) > _MAX_LOG_BYTES:
                    target.replace(target.with_suffix(target.suffix + ".1"))
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(line)
                os.chmod(target, 0o600)
    except OSError:
        return
