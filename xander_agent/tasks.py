from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .models import Handoff, TaskRecord, XanderRequest


def _task_dir() -> Path:
    try:
        from .paths import tasks_dir

        return tasks_dir()
    except ImportError:
        path = Path.home() / ".local" / "state" / "xander" / "tasks"
        path.mkdir(parents=True, exist_ok=True)
        return path


def new_task_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:10]}"


class TaskStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or _task_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, request: XanderRequest) -> TaskRecord:
        task = TaskRecord(id=new_task_id(), request=request)
        self.save(task)
        return task

    def path(self, task_id: str) -> Path:
        if not task_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in task_id):
            raise ValueError("invalid task id")
        return self.root / f"{task_id}.json"

    def save(self, task: TaskRecord) -> Path:
        task.touch()
        destination = self.path(task.id)
        temporary = destination.with_suffix(".json.tmp")
        payload = task.model_dump(mode="json", by_alias=True)
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
        return destination

    def load(self, task_id: str) -> TaskRecord:
        path = self.path(task_id)
        return TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self, limit: int = 100) -> list[TaskRecord]:
        records: list[TaskRecord] = []
        for path in sorted(self.root.glob("*.json"), reverse=True):
            try:
                records.append(TaskRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if len(records) >= limit:
                break
        return records

    def latest(self) -> TaskRecord | None:
        records = self.list(limit=1)
        return records[0] if records else None

    def handoff(self, task: TaskRecord) -> Handoff:
        unresolved: list[str] = []
        if task.failure:
            unresolved.append(task.failure)
        for result in [*task.results, *task.check_results]:
            if result.status.value not in {"ok"}:
                unresolved.append(result.reason or result.stderr or f"action {result.action_id} did not pass")
        return Handoff(
            task_id=task.id,
            workspace=str(task.request.workspace),
            goal=task.request.goal,
            status=task.status,
            snapshot=task.snapshot,
            research_sources=task.research.sources if task.research else [],
            plan=task.plan,
            results=task.results,
            checks=task.check_results,
            unresolved=unresolved,
        )

