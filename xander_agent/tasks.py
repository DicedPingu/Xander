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
        path = Path(__file__).resolve().parents[1] / "state" / "tasks"
        path.mkdir(parents=True, exist_ok=True)
        return path


def new_task_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid4().hex[:10]}"


class TaskStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or _task_dir()
        self._legacy_root: Path | None = None
        if root is None:
            try:
                from .paths import legacy_tasks_dir

                legacy = legacy_tasks_dir()
                if legacy and legacy.resolve(strict=False) != self.root.resolve(strict=False):
                    self._legacy_root = legacy
            except ImportError:
                self._legacy_root = None
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
        paths = [self.path(task_id)]
        if self._legacy_root:
            paths.append(self._legacy_root / f"{task_id}.json")
        for path in paths:
            if not path.is_file():
                continue
            return TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(paths[0])

    def delete(self, task_id: str) -> None:
        """Delete one explicitly selected persisted task."""

        self.path(task_id).unlink()

    def list(self, limit: int = 100) -> list[TaskRecord]:
        candidates: dict[str, tuple[int, TaskRecord]] = {}
        roots = [(0, self.root)]
        if self._legacy_root and self._legacy_root.is_dir():
            roots.append((1, self._legacy_root))
        for priority, root in roots:
            for path in root.glob("*.json"):
                try:
                    record = TaskRecord.model_validate_json(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                current = candidates.get(record.id)
                if current is None or priority < current[0]:
                    candidates[record.id] = (priority, record)
        records = [record for _, record in candidates.values()]
        records.sort(key=lambda record: record.updated_at, reverse=True)
        return records[:limit]

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
            guide=task.guide,
            research_sources=task.research.sources if task.research else [],
            plan=task.plan,
            results=task.results,
            checks=task.check_results,
            unresolved=unresolved,
        )
