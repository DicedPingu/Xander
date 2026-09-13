import difflib
import hashlib
import json
import subprocess
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


@pytest.mark.parametrize("payload", ["missing", "content", "create"])
def test_proposal_resume_returns_applicable_diff_without_running_checks(tmp_path, payload):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    before = "# Guide\n\nOld command\n" + "Keep this documentation.\n" * 300
    if payload == "create":
        before = "Old command\n"
    after = before.replace("Old command", "New command")
    (workspace / "README.md").write_text(before)
    patch = "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile="a/README.md", tofile="b/README.md",
    ))

    class Coder(UnusedBackend):
        def generate(self, prompt, **kwargs):
            assert before in prompt
            return patch

    task_store = TaskStore(root=tmp_path / "tasks")
    task = make_waiting_task(task_store, workspace, autonomy="proposal")
    task.request.goal = "Correct the documented command"
    task.request.allowed_paths = ["README.md"]
    task.plan.actions = [Action(
        kind=ActionKind.CREATE if payload == "create" else ActionKind.PATCH,
        path="README.md", content=after if payload == "create" else patch if payload == "content" else "",
        expected="Replace Old command with New command",
    )]
    task.plan.acceptance_checks = [AcceptanceCheck(
        name="must remain pending", argv=["touch", "checks-were-run"],
    )]
    task_store.save(task)
    engine = make_engine(workspace, task_store, autonomy="proposal", backend=Coder())

    result = engine.resume(task.id)["task"]

    assert result["status"] == TaskStatus.UNVERIFIED
    assert result["check_results"] == []
    assert (workspace / "README.md").read_text() == before
    assert not (workspace / "checks-were-run").exists()
    proposed = result["plan"]["actions"][0]
    assert proposed["kind"] == "patch"
    assert proposed["patch"] == patch
    assert proposed["preimage_hashes"] == {"README.md": hashlib.sha256(before.encode()).hexdigest()}
    assert any(item["kind"] == "proposal-validation" for item in result["evidence"])
    applied = subprocess.run(
        ["git", "apply", "-"], input=proposed["patch"], cwd=workspace,
        text=True, capture_output=True, check=False,
    )
    assert applied.returncode == 0, applied.stderr
    assert (workspace / "README.md").read_text() == after


@pytest.mark.parametrize("failure", ["bad-context", "allowlist", "escape", "secret", "preimage", "inspection"])
def test_invalid_proposal_never_reports_ready_or_mutates(tmp_path, failure):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("current\n")
    task_store = TaskStore(root=tmp_path / "tasks")
    task = make_waiting_task(task_store, workspace, autonomy="proposal")
    task.request.goal = "Correct the documented command"
    path = "../outside.txt" if failure == "escape" else ".env" if failure == "secret" else "README.md"
    patch = f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{'wrong' if failure == 'bad-context' else 'current'}\n+changed\n"
    task.plan.actions = [Action(kind=ActionKind.PATCH, path=path, patch=patch)]
    if failure == "allowlist":
        task.request.allowed_paths = ["other.md"]
    if failure == "preimage":
        task.plan.actions[0].preimage_hashes = {"README.md": "stale"}
    if failure == "inspection":
        task.plan.actions.insert(0, Action(kind=ActionKind.INSPECT, argv=["README.md"]))
    task_store.save(task)
    engine = make_engine(workspace, task_store, autonomy="proposal")
    events = []
    engine.event_sink = events.append

    result = engine.resume(task.id)["task"]

    assert result["status"] != TaskStatus.COMPLETED
    assert not any(event.message == "proposal ready" for event in events)
    assert result["check_results"] == []
    assert (workspace / "README.md").read_text() == "current\n"
    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.parametrize("path_only", [False, True, "repeated"])
def test_proposal_discovery_replans_using_real_inspection(tmp_path, monkeypatch, path_only):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("current\n")
    task_store = TaskStore(root=tmp_path / "tasks")
    task = make_waiting_task(task_store, workspace, autonomy="proposal")
    task.request.goal = "Correct the documented command"
    task.request.allowed_paths = ["README.md"]
    task.plan.actions = [Action(
        kind=ActionKind.INSPECT, path="README.md",
        argv=[] if path_only else ["cat", "README.md"],
    )]
    task.plan.acceptance_checks = []
    task_store.save(task)
    patch = "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-current\n+changed\n"

    class Coder(UnusedBackend):
        def generate(self, prompt, **kwargs):
            assert "CURRENT FILE:\ncurrent\n" in prompt
            return patch

    engine = make_engine(workspace, task_store, autonomy="proposal", backend=Coder())

    def plan_from_evidence(task, skills, failure=""):
        assert "DISCOVERY COMPLETE" in failure
        assert task.results[-1].stdout == "current\n"
        if path_only == "repeated":
            return ModelPlan(summary="Read again", actions=[Action(kind=ActionKind.INSPECT, path="README.md")])
        return ModelPlan(
            summary="Update the inspected command",
            actions=[Action(kind=ActionKind.PATCH, path="README.md", patch=(
                "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-current\n+changed\n"
            ))],
            acceptance_checks=[AcceptanceCheck(name="documentation exists", argv=["test", "-f", "README.md"])],
        )

    monkeypatch.setattr(engine, "_plan", plan_from_evidence)
    result = engine.resume(task.id)["task"]

    assert result["attempt"] == 2
    assert any(item["kind"] == "proposal-validation" for item in result["evidence"])
    if path_only == "repeated":
        assert any(item["kind"] == "proposal-discovery-fallback" for item in result["evidence"])
    assert (workspace / "README.md").read_text() == "current\n"
    applied = subprocess.run(
        ["git", "apply", "-"], input=result["plan"]["actions"][0]["patch"],
        cwd=workspace, text=True, capture_output=True, check=False,
    )
    assert applied.returncode == 0, applied.stderr
    assert (workspace / "README.md").read_text() == "changed\n"


@pytest.mark.parametrize("target", ["../outside", ".env", "other.md", "large.md", "missing.md"])
def test_proposal_path_only_inspection_repair_preserves_boundaries(tmp_path, target):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside").write_text("private\n")
    (workspace / ".env").write_text("private\n")
    (workspace / "other.md").write_text("unselected\n")
    (workspace / "large.md").write_text("x" * 64_001)
    task_store = TaskStore(root=tmp_path / "tasks")
    task = make_waiting_task(task_store, workspace, autonomy="proposal")
    task.request.allowed_paths = ["README.md", ".env", "large.md", "missing.md"]
    action = Action(kind=ActionKind.INSPECT, path=target)
    task.plan.actions = [action]
    engine = make_engine(workspace, task_store, autonomy="proposal")

    engine._repair_plan(task)

    assert action.argv == []


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


def test_explicit_setup_allowance_covers_install_but_not_removal(tmp_path: Path) -> None:
    task_store = TaskStore(root=tmp_path / "tasks")
    engine = make_engine(tmp_path, task_store)
    task = task_store.create(
        XanderRequest(
            mode="implement",
            workspace=tmp_path,
            goal="prepare the local toolchain",
            setup_policy="allow",
        )
    )
    install = Action(kind=ActionKind.COMMAND, argv=["apt-get", "install", "example"])
    remove = Action(kind=ActionKind.COMMAND, argv=["apt-get", "remove", "example"])

    assert engine._approve(task)(install, "package action") is True
    assert engine._approve(task)(remove, "package removal") is False
