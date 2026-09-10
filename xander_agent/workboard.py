"""Workspace-scoped TODO guidance and learning direction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from .models import StrictModel, utc_now


class BoardTodo(StrictModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    text: str
    state: Literal["todo", "active", "done", "blocked"] = "todo"
    evidence: str = ""


class BoardGoal(StrictModel):
    """A stored outcome the operator wants for this workspace.

    Goals are direction, not authorization: storing one never starts work.
    ``/work`` with no argument picks the newest open goal.
    """

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    text: str
    state: Literal["open", "done", "dropped"] = "open"
    created_at: str = Field(default_factory=utc_now)
    evidence: str = ""


class Workboard(StrictModel):
    schema_: Literal["xander.workboard/v1"] = Field(
        default="xander.workboard/v1", alias="schema"
    )
    workspace: str
    goals: list[BoardGoal] = Field(default_factory=list)
    todos: list[BoardTodo] = Field(default_factory=list)
    learning_focus: str = ""
    learning_sources: list[str] = Field(default_factory=list)
    observed_lessons: list[str] = Field(default_factory=list)
    contest_comments: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now)


class WorkboardStore:
    def __init__(self, root: Path | None = None) -> None:
        self._legacy_root: Path | None = None
        if root is None:
            from .paths import legacy_workboards_dir, state_dir

            root = state_dir() / "workboards"
            legacy = legacy_workboards_dir()
            if legacy and legacy.resolve(strict=False) != root.resolve(strict=False):
                self._legacy_root = legacy
        self.root = root

    def path(self, workspace: Path) -> Path:
        resolved = workspace.expanduser().resolve(strict=False)
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:24]
        return self.root / f"{digest}.json"

    def load(self, workspace: Path) -> Workboard:
        resolved = str(workspace.expanduser().resolve(strict=False))
        path = self.path(workspace)
        paths = [path]
        if self._legacy_root:
            paths.append(self._legacy_root / path.name)
        for candidate in paths:
            if not candidate.is_file():
                continue
            try:
                board = Workboard.model_validate_json(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if board.workspace == resolved:
                return board
        return Workboard(workspace=resolved)

    def save(self, board: Workboard) -> Path:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        board.updated_at = utc_now()
        destination = self.path(Path(board.workspace))
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(board.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
        return destination

    def add_todo(self, board: Workboard, text: str) -> BoardTodo | None:
        text = " ".join(text.split())
        if not text:
            return None
        todo = BoardTodo(text=text)
        board.todos.append(todo)
        self.save(board)
        return todo

    def add_goal(self, board: Workboard, text: str) -> BoardGoal | None:
        """Store a goal once; repeating an open goal returns the existing one."""

        text = " ".join(text.split())
        if not text:
            return None
        for goal in board.goals:
            if goal.state == "open" and goal.text.casefold() == text.casefold():
                return goal
        goal = BoardGoal(text=text)
        board.goals.append(goal)
        board.goals = board.goals[-100:]
        self.save(board)
        return goal

    def open_goals(self, board: Workboard) -> list[BoardGoal]:
        return [goal for goal in board.goals if goal.state == "open"]

    def set_goal_state(self, board: Workboard, goal_id: str, state: str, evidence: str = "") -> bool:
        for goal in board.goals:
            if goal.id == goal_id:
                goal.state = state
                goal.evidence = " ".join(evidence.split())
                self.save(board)
                return True
        return False

    def set_learning(self, board: Workboard, focus: str, sources: list[str]) -> None:
        board.learning_focus = " ".join(focus.split())
        board.learning_sources = list(dict.fromkeys(" ".join(item.split()) for item in sources if item.strip()))
        self.save(board)

    def add_comment(self, board: Workboard, text: str) -> bool:
        text = " ".join(text.split())
        if not text:
            return False
        board.contest_comments.append(text)
        board.contest_comments = board.contest_comments[-100:]
        self.save(board)
        return True

    def add_observed_lesson(self, board: Workboard, text: str) -> bool:
        text = " ".join(text.split())
        if not text or text in board.observed_lessons:
            return False
        board.observed_lessons.append(text)
        board.observed_lessons = board.observed_lessons[-100:]
        self.save(board)
        return True

    def set_todo_state(self, board: Workboard, todo_id: str, state: str, evidence: str = "") -> bool:
        for todo in board.todos:
            if todo.id == todo_id:
                todo.state = state
                todo.evidence = " ".join(evidence.split())
                self.save(board)
                return True
        return False
