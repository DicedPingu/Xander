"""The `/learn` loop: work the queue, listen between steps, keep going.

One target at a time, and between every target — and between every action inside
a target — the steering inbox is drained. That cadence is the whole contract: the
operator can type at any moment and never waits for the engine, while Xander
never has a directive applied underneath a step that is already running.

Targets default to ``research`` mode: reading, fetching and summarising, with no
mutations. Learning should not edit a workspace unless the operator says so.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .learn import LearnQueue, LearnStore, LearnTarget, constraints_for
from .steering import SteeringInbox, drain

# What "learn this" actually means, per kind. Concrete beats generic: a vague
# goal is what produces the vague, non-executable plans.
_GOALS: dict[str, str] = {
    "url": (
        "Read {subject} and record what it is, what problem it solves, the parts worth reusing, "
        "and its limits. Quote the specifics you actually retrieved; do not summarise from memory."
    ),
    "repo": (
        "Study the repository {subject}: its purpose, how it is structured, its public entry points, "
        "how it is maintained, and whether it is worth depending on. Use what you actually fetched."
    ),
    "package": (
        "Learn the package {subject}: what it does, the current public API, how it is installed, the "
        "version in use here if any, and the pitfalls. Prefer current documentation over recall."
    ),
    "skill": (
        "Study the skill '{subject}' already available to you: when it applies, what it instructs, and "
        "how it changes your approach. Record when NOT to load it."
    ),
    "category": (
        "Survey the '{subject}' category: which available skills belong to it, what they have in common, "
        "and which one to reach for in which situation."
    ),
    "topic": (
        "Learn about {subject}: the core ideas, the vocabulary, how it is applied in practice, and the "
        "common mistakes. Ground every claim in something you actually read."
    ),
}


def goal_for(target: LearnTarget) -> str:
    template = _GOALS.get(target.kind, _GOALS["topic"])
    return template.format(subject=target.subject)


def run_learning(
    workspace: Path,
    *,
    variant: str = "default",
    caller: str = "human",
    max_targets: int = 0,
    event_sink: Callable[[Any], None] | None = None,
    store: LearnStore | None = None,
    invoke: Callable[..., dict[str, Any]] | None = None,
    backend: Any | None = None,
    deadline: float | None = None,
    inbox: SteeringInbox | None = None,
) -> dict[str, Any]:
    """Drive the queue until it empties, the budget runs out, or nothing is left.

    ``invoke`` and ``store`` are injectable so the loop is testable without a
    model. ``max_targets`` of 0 means "work the whole queue".
    """

    store = store or LearnStore()
    queue = store.load(workspace)
    inbox = inbox or SteeringInbox(workspace)
    if invoke is None:
        from .cli import invoke_engine as invoke  # lazy: cli imports engine

    def emit(kind: str, message: str, data: dict[str, Any] | None = None) -> None:
        if event_sink is None:
            return
        try:
            event_sink({"type": kind, "phase": "learn", "message": message, "data": data or {}})
        except Exception:
            pass

    def listen(moment: str) -> list[str]:
        """Drain the back channel. Safe point only — never inside a step."""

        handled = drain(workspace, queue, store, backend=backend, inbox=inbox)
        lines: list[str] = []
        for note, outcome in handled:
            kind = note.directive.kind if note.directive else "note"
            line = f"{kind}: {note.text} -> {outcome}"
            lines.append(line)
            emit("steering", line, {"moment": moment, "kind": kind, "outcome": outcome})
        return lines

    worked: list[dict[str, Any]] = []
    stopped = ""
    listen("start")

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            stopped = "time budget reached"
            break
        if max_targets and len(worked) >= max_targets:
            stopped = f"reached the {max_targets}-target limit for this run"
            break
        target = queue.next_target()
        if target is None:
            stopped = "queue is empty"
            break

        store.set_state(queue, target.id, "active")
        emit(
            "learn",
            f"learning {target.kind}: {target.subject}",
            {"target": target.id, "kind": target.kind, "progress": queue.progress()},
        )

        budget = queue.settings.per_target_seconds
        if deadline is not None:
            budget = max(60, min(budget, int(deadline - time.monotonic())))
        try:
            result = invoke(
                queue.settings.mode,
                workspace=workspace,
                goal=goal_for(target),
                variant=variant,
                caller=caller,
                autonomy=queue.settings.autonomy,
                constraints=constraints_for(queue),
                timeout=budget,
                event_sink=event_sink,
                log_events=True,
            )
        except Exception as exc:
            store.set_state(queue, target.id, "blocked", evidence=f"{type(exc).__name__}: {exc}"[:400])
            emit("learn", f"blocked on {target.subject}: {exc}"[:300], {"target": target.id})
            listen("after-target")
            continue

        # The operator may have skipped this target while it ran; respect that
        # rather than overwriting their decision with the stale outcome.
        current = next((item for item in queue.targets if item.id == target.id), None)
        if current is not None and current.state == "skipped":
            emit("learn", f"skipped mid-flight by the operator: {target.subject}", {"target": target.id})
            listen("after-target")
            continue

        task = result.get("task") if isinstance(result.get("task"), dict) else {}
        ok = bool(result.get("ok")) or str(result.get("status") or "").casefold() in {"complete", "completed"}
        lesson = str(task.get("lesson") or "")
        evidence = str(task.get("failure") or "") if not ok else f"task {result.get('task_id', '?')}"
        store.set_state(queue, target.id, "done" if ok else "blocked", evidence=evidence, lesson=lesson)
        worked.append({"target": target.subject, "kind": target.kind, "ok": ok, "lesson": lesson})
        emit(
            "learn",
            f"{'learned' if ok else 'could not learn'} {target.subject} — {queue.progress()}",
            {"target": target.id, "ok": ok, "lesson": lesson[:300]},
        )

        if ok and queue.settings.depth > 0:
            follow = _follow_ups(result, queue, target)
            added = store.add_many(queue, follow, origin="discovered", parent=target.id)
            if added:
                emit(
                    "learn",
                    "found worth following: " + ", ".join(item.subject for item in added),
                    {"parent": target.id, "count": len(added)},
                )

        listen("after-target")

    summary = {
        "event": "learn",
        "workspace": str(workspace),
        "worked": worked,
        "stopped_because": stopped,
        "progress": queue.progress(),
        "pending": [target.subject for target in queue.pending()][:20],
        "mission": queue.mission,
        "restrictions": list(queue.restrictions),
    }
    emit("learn", f"run finished — {queue.progress()}; {stopped}", {"worked": len(worked)})
    return summary


def _follow_ups(result: dict[str, Any], queue: LearnQueue, target: LearnTarget) -> list[str]:
    """Sources the run actually used, capped by depth. Never invented."""

    handoff = result.get("handoff") if isinstance(result.get("handoff"), dict) else {}
    sources = handoff.get("research_sources") or []
    known = {item.subject.casefold() for item in queue.targets}
    picked: list[str] = []
    for source in sources:
        text = " ".join(str(source).split())
        if not text or text.casefold() in known or text.casefold() == target.subject.casefold():
            continue
        picked.append(text)
        known.add(text.casefold())
        if len(picked) >= queue.settings.depth:
            break
    return picked
