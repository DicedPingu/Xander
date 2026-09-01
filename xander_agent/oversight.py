"""Read-only visibility into the current Xander mission."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .models import TaskRecord, TaskStatus
from .paths import agent_dir, logs_dir, projects_dir, shared_dir, state_dir
from .tasks import TaskStore

_ACTIVE_WINDOW = timedelta(hours=24)


def _same_workspace(task: TaskRecord, workspace: Path) -> bool:
    return task.request.workspace.expanduser().resolve(strict=False) == workspace


def _task_view(task: TaskRecord) -> dict[str, Any]:
    guide = task.guide
    routes = [
        {
            key: item[key]
            for key in ("phase", "role", "reason", "model")
            if key in item and item[key]
        }
        for item in task.evidence
        if item.get("kind") == "model_route"
    ]
    delegations = [
        {
            key: item[key]
            for key in ("agent", "role", "operation", "status", "outcome", "model", "error")
            if key in item and item[key]
        }
        for item in task.evidence
        if item.get("kind") == "delegation"
    ]
    changes = [
        path
        for result in [*task.results, *task.check_results]
        for path in result.changed_paths
    ]
    return {
        "id": task.id,
        "goal": task.request.goal,
        "status": task.status,
        "phase": task.phase,
        "attempt": task.attempt,
        "updated_at": task.updated_at,
        "progress": guide.progress if guide else "",
        "current": guide.current if guide else "",
        "next": guide.questions[:3] if guide else [],
        "failure": task.failure,
        "last_change": guide.last_change if guide else "",
        "changed_paths": list(dict.fromkeys(changes))[-12:],
        "model_routes": routes[-12:],
        "delegations": delegations[-24:],
        "evidence_count": len(task.evidence),
    }


def _recent(task: TaskRecord) -> bool:
    try:
        updated = datetime.fromisoformat(task.updated_at)
    except ValueError:
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    return datetime.now(UTC) - updated <= _ACTIVE_WINDOW


def oversight_payload(workspace: Path, *, task_store: TaskStore | None = None) -> dict[str, Any]:
    root = workspace.expanduser().resolve(strict=False)
    store = task_store or TaskStore()
    tasks = [task for task in store.list(limit=500) if _same_workspace(task, root)]
    latest = tasks[0] if tasks else None
    active_candidates = [
        task
        for task in tasks
        if task.status in {TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.WAITING_APPROVAL}
    ]
    active = next(
        (
            task
            for task in active_candidates
            if _recent(task)
        ),
        None,
    )
    return {
        "event": "oversight",
        "schema": "xander.oversight/v1",
        "workspace": str(root),
        "active": _task_view(active) if active else None,
        "stale_active": [_task_view(task) for task in active_candidates if not _recent(task)],
        "latest": _task_view(latest) if latest else None,
        "task_count": len(tasks),
        "storage": {
            "agent": str(agent_dir()),
            "state": str(state_dir()),
            "logs": str(logs_dir()),
            "projects": str(projects_dir()),
            "shared": str(shared_dir()),
        },
    }
