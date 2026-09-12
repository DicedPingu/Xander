from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = "1"
MANTRA = (
    "analyze",
    "research",
    "set_up",
    "work",
    "test",
    "judge_log",
    "learn",
    "repeat",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Phase(StrEnum):
    ANALYZE = "analyze"
    RESEARCH = "research"
    SET_UP = "set_up"
    WORK = "work"
    TEST = "test"
    JUDGE_LOG = "judge_log"
    LEARN = "learn"
    REPEAT = "repeat"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    FAILED = "failed"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    UNVERIFIED = "unverified"


class ActionKind(StrEnum):
    INSPECT = "inspect"
    COMMAND = "command"
    PIPELINE = "pipeline"
    PATCH = "patch"
    CREATE = "create"
    NOTE = "note"


class ActionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    OK = "ok"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class Risk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AcceptanceCheck(StrictModel):
    name: str
    argv: list[str]
    cwd: str = "."
    timeout: int = Field(default=300, ge=1, le=3600)
    required: bool = True

    @field_validator("argv")
    @classmethod
    def argv_is_not_empty(cls, value: list[str]) -> list[str]:
        if not value or not value[0].strip():
            raise ValueError("acceptance check argv cannot be empty")
        return value


class Action(StrictModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    kind: ActionKind = Field(
        description="Executable action type. inspect and command require a non-empty argv array."
    )
    cwd: str = Field(
        default=".",
        description="Working directory relative to the exact workspace; normally '.'.",
    )
    argv: list[str] = Field(
        default_factory=list,
        description=(
            "Full executable and arguments. REQUIRED and non-empty for inspect and command; "
            "empty for create, patch, note, and pipeline."
        ),
    )
    pipeline: list[list[str]] = Field(default_factory=list)
    patch: str = ""
    path: str = Field(
        default="",
        description=(
            "Target path relative to the exact workspace; only for create actions. "
            "Missing parent directories are created automatically."
        ),
    )
    content: str = ""
    expected: str = ""
    acceptance_check: str | None = None
    blocking: bool = True
    risk: Risk = Risk.LOW
    preimage_hashes: dict[str, str | None] = Field(default_factory=dict)
    option_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    parallel_group: str | None = None

    @field_validator("argv")
    @classmethod
    def no_empty_command(cls, value: list[str]) -> list[str]:
        if value and not value[0].strip():
            raise ValueError("action argv executable cannot be empty")
        return value


class ActionResult(StrictModel):
    action_id: str
    action_hash: str = ""
    status: ActionStatus
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    changed_paths: list[str] = Field(default_factory=list)
    started_at: str = Field(default_factory=utc_now)
    finished_at: str = Field(default_factory=utc_now)
    reason: str = ""


class WorkspaceSnapshot(StrictModel):
    root: str
    git: bool = False
    branch: str = ""
    dirty_count: int = 0
    dirty: dict[str, str | None] = Field(default_factory=dict)
    taken_at: str = Field(default_factory=utc_now)


class ResearchBundle(StrictModel):
    local_context: str = ""
    documentation: str = ""
    skills: list[dict[str, Any]] = Field(default_factory=list)
    tools: dict[str, str] = Field(default_factory=dict)
    sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class XanderRequest(StrictModel):
    schema_: Literal["xander.request/v1"] = Field(
        default="xander.request/v1", alias="schema"
    )
    caller: Literal["human", "codex", "claude", "automation"] = "human"
    mode: Literal["inspect", "research", "plan", "implement", "test-triage", "answer"]
    workspace: Path
    goal: str
    constraints: list[str] = Field(default_factory=list)
    acceptance_checks: list[AcceptanceCheck] = Field(default_factory=list)
    allowed_paths: list[str] = Field(default_factory=list)
    timeout: int = Field(default=900, ge=1, le=14_400)
    time_budget_seconds: int | None = Field(default=None, ge=60, le=86_400)
    variant: str = "default"
    autonomy: Literal["proposal", "supervised", "full-auto"] = "full-auto"
    setup_policy: Literal["ask", "allow", "never"] = "ask"
    selected_options: list[str] = Field(default_factory=list)

    @field_validator("autonomy", mode="before")
    @classmethod
    def normalize_autonomy(cls, value: str) -> str:
        return "proposal" if value == "proposal-only" else value

    @field_validator("goal")
    @classmethod
    def goal_is_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("goal cannot be empty")
        return value


class PlanOption(StrictModel):
    id: str
    title: str
    summary: str
    benefit: str = ""
    effort: str = ""
    requires: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    selected_by_default: bool = False


class GuideStep(StrictModel):
    id: str
    text: str
    state: Literal["todo", "active", "done", "blocked"] = "todo"
    evidence: str = ""


class MissionGuide(StrictModel):
    statement: str
    todo: list[GuideStep] = Field(default_factory=list)
    current: str = ""
    progress: str = "0/0 complete"
    questions: list[str] = Field(default_factory=list)
    last_change: str = ""
    result: str = ""
    updated_at: str = Field(default_factory=utc_now)


class ModelPlan(StrictModel):
    summary: str
    decision: str = ""
    why: str = ""
    options: list[PlanOption] = Field(default_factory=list)
    selection_required: bool = False
    actions: list[Action] = Field(default_factory=list)
    acceptance_checks: list[AcceptanceCheck] = Field(default_factory=list)


class TaskRecord(StrictModel):
    schema_: Literal["xander.task/v1"] = Field(default="xander.task/v1", alias="schema")
    id: str
    request: XanderRequest
    status: TaskStatus = TaskStatus.PENDING
    phase: Phase = Phase.ANALYZE
    attempt: int = 0
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    snapshot: WorkspaceSnapshot | None = None
    subject: str = ""
    effective_constraints: list[str] = Field(default_factory=list)
    research: ResearchBundle | None = None
    tool_readiness: dict[str, Any] = Field(default_factory=dict)
    plan: ModelPlan | None = None
    guide: MissionGuide | None = None
    results: list[ActionResult] = Field(default_factory=list)
    check_results: list[ActionResult] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    failure: str = ""
    lesson: str = ""

    def touch(self) -> None:
        self.updated_at = utc_now()


class XanderEvent(StrictModel):
    schema_: Literal["xander.event/v1"] = Field(default="xander.event/v1", alias="schema")
    run_id: str
    sequence: int
    type: Literal[
        "task",
        "guide",
        "phase",
        "plan",
        "research",
        "delegation",
        "action",
        "patch",
        "logic_change",
        "test",
        "approval",
        "result",
        "error",
        "voice",
        "steering",
    ]
    timestamp: str = Field(default_factory=utc_now)
    phase: Phase | None = None
    attempt: int = 0
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class Handoff(StrictModel):
    schema_: Literal["xander.handoff/v1"] = Field(
        default="xander.handoff/v1", alias="schema"
    )
    task_id: str
    workspace: str
    goal: str
    status: TaskStatus
    snapshot: WorkspaceSnapshot | None = None
    guide: MissionGuide | None = None
    research_sources: list[str] = Field(default_factory=list)
    plan: ModelPlan | None = None
    results: list[ActionResult] = Field(default_factory=list)
    checks: list[ActionResult] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
