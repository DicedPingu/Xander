"""Human-facing Mission views over Xander's durable task records."""

from __future__ import annotations

from typing import Any, Literal

from . import tasks as tasks_module
from .models import TaskRecord


MissionMoment = Literal["guide", "change", "milestone", "thought", "idea", "rebirth", "result"]


class Mission:
    """A readable, workspace-scoped projection of one persisted task.

    Tasks remain the execution record and wire format. Mission is the vocabulary
    the human UI uses: one goal in one folder, with a timeline and a result.
    """

    def __init__(self, task: TaskRecord) -> None:
        self.task = task

    @property
    def id(self) -> str:
        return self.task.id

    @property
    def workspace(self) -> str:
        return str(self.task.request.workspace)

    @property
    def goal(self) -> str:
        return self.task.request.goal

    @property
    def status(self) -> str:
        return self.task.status.value

    @property
    def phase(self) -> str:
        return self.task.phase.value

    @property
    def result(self) -> str:
        if self.task.status.value == "completed":
            return "completed with verified evidence"
        if self.task.failure:
            return self.task.failure
        if self.task.lesson:
            return self.task.lesson
        return f"{self.task.status.value} in {self.task.phase.value}"

    def timeline(self, limit: int = 40) -> list[dict[str, Any]]:
        moments: list[dict[str, Any]] = []
        if self.task.guide:
            moments.append({"kind": "guide", "message": self.task.guide.statement})
        for evidence in self.task.evidence:
            kind = str(evidence.get("kind", "thought"))
            if kind not in {"change", "milestone", "thought", "idea", "rebirth", "result", "guide"}:
                kind = "thought"
            moments.append({"kind": kind, "message": _evidence_message(evidence)})
        for item in [*self.task.results, *self.task.check_results]:
            moments.append(
                {
                    "kind": "change" if item.changed_paths else "result",
                    "message": _result_message(item),
                }
            )
        if self.task.failure:
            moments.append({"kind": "result", "message": self.task.failure})
        if self.task.status.value in {"completed", "failed", "unverified", "interrupted"}:
            moments.append({"kind": "result", "message": self.result})
        return moments[-limit:]

    def summary_lines(self) -> list[str]:
        lines = [
            f"Mission {self.id}",
            f"  folder: {self.workspace}",
            f"  goal:   {self.goal}",
            f"  state:  {self.status} · {self.phase} · attempt {self.task.attempt}",
            f"  result: {self.result}",
        ]
        if self.task.guide:
            lines.extend(
                [
                    f"  guide:  {self.task.guide.statement}",
                    f"  progress: {self.task.guide.progress}",
                    f"  current:  {self.task.guide.current}",
                ]
            )
            for question in self.task.guide.questions:
                lines.append(f"  question: {question}")
        return lines


class MissionStore:
    """List, inspect, and explicitly delete Missions for one workspace."""

    def __init__(self, task_store: tasks_module.TaskStore | None = None) -> None:
        self.tasks = task_store or tasks_module.TaskStore()

    def list(self, workspace: Any, limit: int = 100) -> list[Mission]:
        root = workspace.expanduser().resolve(strict=False)
        return [
            Mission(task)
            for task in self.tasks.list(limit=limit)
            if task.request.workspace.expanduser().resolve(strict=False) == root
        ]

    def load(self, workspace: Any, mission_id: str) -> Mission:
        mission = Mission(self.tasks.load(mission_id))
        root = workspace.expanduser().resolve(strict=False)
        if mission.task.request.workspace.expanduser().resolve(strict=False) != root:
            raise ValueError("mission belongs to a different workspace")
        return mission

    def delete(self, workspace: Any, mission_id: str) -> None:
        mission = self.load(workspace, mission_id)
        self.tasks.delete(mission.id)


def _evidence_message(evidence: dict[str, Any]) -> str:
    for key in ("message", "reason", "summary", "text", "lesson"):
        value = evidence.get(key)
        if value:
            return " ".join(str(value).split())
    return "recorded evidence"


def _result_message(result: Any) -> str:
    if result.changed_paths:
        paths = ", ".join(result.changed_paths[:4])
        suffix = " …" if len(result.changed_paths) > 4 else ""
        return f"{result.status.value}: {paths}{suffix}"
    return result.reason or result.stderr or f"{result.status.value} (exit {result.returncode})"
