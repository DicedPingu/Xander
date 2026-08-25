import json
from pathlib import Path

import pytest

from xander_agent.engine import Engine
from xander_agent.memory import MemoryStore
from xander_agent.models import (
    AcceptanceCheck,
    Action,
    ActionKind,
    ActionResult,
    ActionStatus,
    ModelPlan,
    ResearchBundle,
    TaskStatus,
    WorkspaceSnapshot,
    XanderRequest,
)
from xander_agent.tasks import TaskStore


class UnusedBackend:
    def available(self) -> bool:
        return False


class StubSkills:
    def load_selected(self, query: str, limit: int = 3, max_chars: int = 24_000) -> list[dict[str, str]]:
        return []


class StubResearcher:
    def gather(self, goal: str, subject: str = "") -> ResearchBundle:
        return ResearchBundle()


def make_engine(workspace: Path, task_store: TaskStore, *, autonomy: str = "full-auto", backend=None) -> Engine:
    return Engine(
        workspace=workspace,
        autonomy=autonomy,
        backend=backend or UnusedBackend(),
        task_store=task_store,
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=task_store.root.parent / f"memory-{workspace.name}.json"),
    )


def make_waiting_task(task_store: TaskStore, workspace: Path, *, autonomy: str):
    task = task_store.create(
        XanderRequest(
            mode="implement",
            workspace=workspace,
            goal="create the resume marker",
            autonomy=autonomy,
        )
    )
    task.status = TaskStatus.WAITING_APPROVAL
    task.snapshot = WorkspaceSnapshot(root=str(workspace))
    task.subject = "resume authority"
    task.research = ResearchBundle()
    task.plan = ModelPlan(
        summary="persisted mutation",
        actions=[
            Action(
                kind=ActionKind.CREATE,
                path="resume-marker.txt",
                content="mutated\n",
                expected="create a marker",
            )
        ],
        acceptance_checks=[AcceptanceCheck(name="marker exists", argv=["test", "-f", "resume-marker.txt"])],
    )
    task_store.save(task)
    return task


@pytest.mark.parametrize(
    ("persisted_autonomy", "resuming_autonomy"),
    [
        pytest.param("full-auto", "proposal", id="proposal-engine-lowers-full-auto-task"),
        pytest.param("proposal", "full-auto", id="full-auto-engine-cannot-elevate-proposal-task"),
    ],
)
def test_resume_authority_is_monotonic_and_never_mutates_in_proposal_mode(
    tmp_path: Path,
    persisted_autonomy: str,
    resuming_autonomy: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    task_store = TaskStore(root=tmp_path / "tasks")
    task = make_waiting_task(task_store, workspace, autonomy=persisted_autonomy)

    result = make_engine(workspace, task_store, autonomy=resuming_autonomy).resume(task.id)

    assert result["task"]["request"]["autonomy"] == "proposal"
    assert task_store.load(task.id).request.autonomy == "proposal"
    assert result["task"]["status"] == TaskStatus.UNVERIFIED
    assert not (workspace / "resume-marker.txt").exists()


def test_task_list_and_show_are_scoped_to_the_engine_workspace(tmp_path: Path) -> None:
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    task_store = TaskStore(root=tmp_path / "tasks")
    first = task_store.create(
        XanderRequest(mode="inspect", workspace=first_workspace, goal="first workspace task")
    )
    second = task_store.create(
        XanderRequest(mode="inspect", workspace=second_workspace, goal="second workspace task")
    )
    engine = make_engine(first_workspace, task_store)

    assert [task["id"] for task in engine.list_tasks()] == [first.id]
    assert engine.show_task(first.id)["task"]["id"] == first.id
    with pytest.raises(ValueError, match="different workspace"):
        engine.show_task(second.id)


def test_each_acceptance_check_supplies_its_own_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_store = TaskStore(root=tmp_path / "tasks")
    engine = make_engine(tmp_path, task_store)
    task = task_store.create(
        XanderRequest(mode="implement", workspace=tmp_path, goal="run checks")
    )
    task.snapshot = WorkspaceSnapshot(root=str(tmp_path))
    observed_timeouts: list[int] = []

    class RecordingExecutor:
        def __init__(self, *args, default_timeout: int, **kwargs) -> None:
            observed_timeouts.append(default_timeout)

        def run(self, action: Action) -> ActionResult:
            return ActionResult(action_id=action.id, status=ActionStatus.OK, returncode=0)

    monkeypatch.setattr("xander_agent.engine.ActionExecutor", RecordingExecutor)
    checks = [
        AcceptanceCheck(name="fast", argv=["true"], timeout=1),
        AcceptanceCheck(name="slow", argv=["true"], timeout=17),
    ]

    results = engine._run_checks(task, checks)

    assert observed_timeouts == [1, 17]
    assert [result.status for result in results] == [ActionStatus.OK, ActionStatus.OK]


def test_acceptance_check_timeout_is_capped_by_the_task_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_store = TaskStore(root=tmp_path / "tasks")
    engine = make_engine(tmp_path, task_store)
    task = task_store.create(
        XanderRequest(mode="implement", workspace=tmp_path, goal="run a bounded check")
    )
    task.snapshot = WorkspaceSnapshot(root=str(tmp_path))
    observed_timeouts: list[int] = []

    class RecordingExecutor:
        def __init__(self, *args, default_timeout: int, **kwargs) -> None:
            observed_timeouts.append(default_timeout)

        def run(self, action: Action) -> ActionResult:
            return ActionResult(action_id=action.id, status=ActionStatus.OK, returncode=0)

    monkeypatch.setattr("xander_agent.engine.ActionExecutor", RecordingExecutor)
    monkeypatch.setattr("xander_agent.engine.time.monotonic", lambda: 100.0)

    results = engine._run_checks(
        task,
        [AcceptanceCheck(name="long check", argv=["true"], timeout=300)],
        deadline=103.75,
    )

    assert observed_timeouts == [3]
    assert results[0].status == ActionStatus.OK


def test_planning_exception_is_persisted_as_a_terminal_failure(tmp_path: Path) -> None:
    class PlanningFailureBackend:
        def available(self) -> bool:
            return True

        def generate(self, prompt, role="coder", schema=None, think=False, timeout=None) -> str:
            if schema is not None and schema.__name__ == "Analysis":
                return json.dumps(
                    {"subject": "planning failure", "constraints": [], "task_type": "coding", "complexity": 2}
                )
            raise RuntimeError("planner exploded")

    task_store = TaskStore(root=tmp_path / "tasks")
    engine = make_engine(tmp_path, task_store, backend=PlanningFailureBackend())

    result = engine.execute(mode="implement", goal="persist a planning failure", timeout=60)

    persisted = task_store.load(result["task_id"])
    assert result["task"]["status"] == TaskStatus.FAILED
    assert persisted.status == TaskStatus.FAILED
    assert persisted.status != TaskStatus.RUNNING
    assert "RuntimeError: planner exploded" in persisted.failure
