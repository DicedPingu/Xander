"""The replan guard must stop as soon as the model repeats itself."""

import json
from pathlib import Path

from xander_agent.engine import Engine
from xander_agent.memory import MemoryStore
from xander_agent.models import AcceptanceCheck, Action, ActionKind, MissionGuide, ModelPlan, ResearchBundle, TaskStatus, XanderRequest
from xander_agent.tasks import TaskStore


class ScriptedBackend:
    def __init__(self, plans: list[ModelPlan]) -> None:
        self.plans = list(plans)
        self.prompts: list[str] = []

    def available(self) -> bool:
        return True

    def generate(self, prompt, role="coder", schema=None, think=False, timeout=None) -> str:
        self.prompts.append(prompt)
        if schema is None:
            return "LGTM"  # squad critic calls (Master review/verdict) stay content
        if schema.__name__ == "Analysis":
            return json.dumps({"subject": "replan probe", "constraints": [], "task_type": "coding", "complexity": 2})
        return self.plans.pop(0).model_dump_json()


class RecoveringBackend:
    def __init__(self) -> None:
        self.working = False

    def available(self) -> bool:
        return True

    def generate(self, prompt, role="coder", schema=None, think=False, timeout=None) -> str:
        if schema is None:
            return "LGTM"
        if schema.__name__ == "Analysis":
            return json.dumps(
                {"subject": "resume recovery", "constraints": [], "task_type": "coding", "complexity": 2}
            )
        if not self.working:
            raise RuntimeError("planner temporarily unavailable")
        return ModelPlan(
            summary="recovered",
            actions=[Action(kind=ActionKind.INSPECT, argv=["true"], expected="probe")],
            acceptance_checks=[AcceptanceCheck(name="passes", argv=["true"])],
        ).model_dump_json()


class StubSkills:
    def load_selected(self, query: str, limit: int = 3, max_chars: int = 24_000) -> list[dict[str, str]]:
        return []


class StubResearcher:
    def gather(self, goal: str, subject: str = "") -> ResearchBundle:
        return ResearchBundle()


def make_plan(expected: str) -> ModelPlan:
    return ModelPlan(
        summary=f"plan: {expected}",
        actions=[Action(kind=ActionKind.INSPECT, argv=["true"], expected=expected)],
        acceptance_checks=[AcceptanceCheck(name="always-fails", argv=["false"])],
    )


def test_replanning_stops_when_the_new_plan_repeats_the_one_just_run(tmp_path: Path) -> None:
    backend = ScriptedBackend([make_plan("first probe"), make_plan("second probe"), make_plan("second probe")])
    engine = Engine(
        workspace=tmp_path,
        backend=backend,
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )
    result = engine.execute(mode="implement", goal="exercise the replan guard", caller="human")

    assert result["ok"] is False
    task = result["task"]
    assert task["attempt"] == 2, "the repeated plan must never be executed a third time"
    assert "repeated the same approach" in task["failure"]
    assert backend.plans == [], "all scripted plans were consumed"


def test_new_mission_writes_a_living_guide_before_the_engine_starts(tmp_path: Path) -> None:
    backend = ScriptedBackend([make_plan("first probe")])
    events = []
    engine = Engine(
        workspace=tmp_path,
        backend=backend,
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
        event_sink=events.append,
    )

    result = engine.execute(mode="implement", goal="make the Mission explainable", caller="human")

    assert events[0].type == "guide"
    initial_guide = events[0].data["guide"]
    guide = result["task"]["guide"]
    assert guide["statement"] == "Deliver a verified result for: make the Mission explainable"
    assert guide["todo"]
    assert guide["progress"]
    assert initial_guide["questions"] == []


def test_inspection_slice_feeds_evidence_into_the_next_implementation_plan(tmp_path: Path) -> None:
    discovery = ModelPlan(
        summary="inspect the project before choosing a change",
        decision="Inspect the project once",
        why="The request is broad and the current files are the missing evidence.",
        actions=[Action(kind=ActionKind.INSPECT, argv=["true"], expected="confirm the workspace")],
    )
    implementation = ModelPlan(
        summary="create the verified marker",
        decision="Create the marker and verify it",
        why="The discovery slice completed, so a small file plus a deterministic check is sufficient.",
        actions=[Action(kind=ActionKind.CREATE, path="marker.txt", content="ready\n", expected="write the marker")],
        acceptance_checks=[AcceptanceCheck(name="marker exists", argv=["test", "-f", "marker.txt"])],
    )
    backend = ScriptedBackend([discovery, implementation])
    engine = Engine(
        workspace=tmp_path,
        backend=backend,
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )

    result = engine.execute(mode="implement", goal="find a useful improvement and prove it", caller="human")

    assert result["ok"] is True
    task = result["task"]
    assert task["attempt"] == 2
    assert task["status"] == "completed"
    assert any(item["kind"] == "discovery" for item in task["evidence"])
    assert (tmp_path / "marker.txt").read_text(encoding="utf-8") == "ready\n"


def test_retrying_a_written_file_does_not_hit_the_create_overwrite_wall(tmp_path: Path) -> None:
    first = ModelPlan(
        summary="write the marker, then force a retry",
        actions=[
            Action(kind=ActionKind.CREATE, path="marker.txt", content="ready\n", expected="write the marker"),
            Action(kind=ActionKind.COMMAND, argv=["false"], expected="force a retry"),
        ],
        acceptance_checks=[AcceptanceCheck(name="marker exists", argv=["test", "-f", "marker.txt"])],
    )
    retry = ModelPlan(
        summary="write the marker and finish",
        actions=[
            Action(kind=ActionKind.CREATE, path="marker.txt", content="ready\n", expected="confirm the marker remains"),
            Action(kind=ActionKind.COMMAND, argv=["true"], expected="the changed route passes"),
        ],
        acceptance_checks=[AcceptanceCheck(name="marker exists", argv=["test", "-f", "marker.txt"])],
    )
    backend = ScriptedBackend([first, retry])
    engine = Engine(
        workspace=tmp_path,
        backend=backend,
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )

    result = engine.execute(mode="implement", goal="retry a safe marker write", caller="human")

    assert result["ok"] is True
    assert result["task"]["attempt"] == 2
    assert result["task"]["status"] == "completed"
    assert (tmp_path / "marker.txt").read_text(encoding="utf-8") == "ready\n"
    assert result["task"]["plan"]["actions"][0]["kind"] == "note"


def test_lint_flags_empty_implementation_plans_before_execution() -> None:
    failure = Engine._lint_plan(ModelPlan(summary="empty"), require_execution=True, require_checks=True)

    assert "no executable actions" in failure
    assert "no executable change" in failure
    assert "no required acceptance checks" in failure


def test_replan_hash_ignores_narrative_but_keeps_action_expectations() -> None:
    first = ModelPlan(
        summary="first wording",
        decision="choose it",
        why="because",
        actions=[Action(kind=ActionKind.INSPECT, argv=["true"], expected="inspect files")],
    )
    same_action = ModelPlan(
        summary="different wording",
        decision="another sentence",
        why="new prose",
        actions=[Action(kind=ActionKind.INSPECT, argv=["true"], expected="inspect files")],
    )
    changed_action = ModelPlan(
        summary="different wording",
        actions=[Action(kind=ActionKind.INSPECT, argv=["true"], expected="inspect tests")],
    )

    assert Engine._plan_hash(first) == Engine._plan_hash(same_action)
    assert Engine._plan_hash(first) != Engine._plan_hash(changed_action)


def test_completed_mission_can_be_reopened_for_further_work(tmp_path: Path) -> None:
    store = TaskStore(root=tmp_path / "tasks")
    task = store.create(XanderRequest(mode="inspect", workspace=tmp_path, goal="revisit this result"))
    task.status = TaskStatus.COMPLETED
    task.guide = MissionGuide(statement="Deliver a verified result for: revisit this result")
    store.save(task)
    engine = Engine(
        workspace=tmp_path,
        backend=ScriptedBackend([]),
        task_store=store,
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )

    resumed = engine.resume(task.id)

    assert resumed["ok"] is True
    assert resumed["task"]["status"] == "completed"
    assert resumed["task"]["guide"]["result"] == "read-only task complete"


def test_invalid_plans_are_linted_and_replanned_with_precise_feedback(tmp_path: Path) -> None:
    from xander_agent.models import ActionStatus

    doomed = ModelPlan(
        summary="doomed",
        actions=[Action(kind=ActionKind.INSPECT, argv=[], expected="look around")],
        acceptance_checks=[AcceptanceCheck(name="passes", argv=["true"])],
    )
    good = ModelPlan(
        summary="good",
        actions=[Action(kind=ActionKind.INSPECT, argv=["true"], expected="probe")],
        acceptance_checks=[AcceptanceCheck(name="passes", argv=["true"])],
    )
    backend = ScriptedBackend([doomed, good])
    engine = Engine(
        workspace=tmp_path,
        backend=backend,
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )
    result = engine.execute(mode="implement", goal="exercise the plan lint", caller="human")

    assert result["ok"] is True
    task = result["task"]
    assert task["attempt"] == 2
    assert all(item["status"] == ActionStatus.OK for item in task["results"]), (
        "the doomed plan must never reach the executor"
    )


def test_lint_rejects_pipelines_with_empty_stages() -> None:
    plan = ModelPlan(
        summary="pipeline probe",
        actions=[Action(kind=ActionKind.PIPELINE, pipeline=[["echo", "hi"], []], expected="stream")],
    )
    assert "pipeline" in Engine._lint_plan(plan)
    assert Engine._lint_plan(make_plan("fine")) == ""


def test_lint_rejects_duplicate_action_ids_and_check_names() -> None:
    plan = ModelPlan(
        summary="ambiguous identities",
        actions=[
            Action(id="same", kind=ActionKind.INSPECT, argv=["true"]),
            Action(id="same", kind=ActionKind.INSPECT, argv=["true"]),
        ],
        acceptance_checks=[
            AcceptanceCheck(name="same check", argv=["true"]),
            AcceptanceCheck(name="same check", argv=["false"]),
        ],
    )

    failure = Engine._lint_plan(plan)

    assert "duplicate action ids: same" in failure
    assert "duplicate acceptance-check names: same check" in failure


def test_planner_contract_shows_executable_actions_and_workspace_relative_paths(tmp_path: Path) -> None:
    backend = ScriptedBackend([make_plan("probe")])
    engine = Engine(
        workspace=tmp_path,
        backend=backend,
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )

    result = engine.execute(mode="implement", goal="make a small project", caller="human")

    assert result["ok"] is False
    planner_prompt = next(prompt for prompt in backend.prompts if "structured coding plan" in prompt)
    assert '"argv":["rg","--files"]' in planner_prompt
    assert "All cwd and path values are relative to WORKSPACE" in planner_prompt
    assert "normally leave depends_on empty" in planner_prompt


def test_resume_gets_fresh_attempts_after_a_terminal_planner_failure(tmp_path: Path) -> None:
    backend = RecoveringBackend()
    store = TaskStore(root=tmp_path / "tasks")
    engine = Engine(
        workspace=tmp_path,
        backend=backend,
        task_store=store,
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )
    failed = engine.execute(mode="implement", goal="recover after a fixed backend", timeout=60)
    attempts_before = failed["task"]["attempt"]
    backend.working = True

    resumed = engine.resume(failed["task_id"])

    assert resumed["ok"] is True
    assert resumed["task"]["attempt"] > attempts_before


def test_completed_read_only_learning_stays_in_memory_not_an_authored_skill(tmp_path: Path) -> None:
    class TrackingSkills(StubSkills):
        def __init__(self) -> None:
            self.promotions = 0

        def record_experience(self, *args, **kwargs):
            self.promotions += 1
            return {"name": "must-not-exist"}

    skills = TrackingSkills()
    engine = Engine(
        workspace=tmp_path,
        backend=ScriptedBackend([]),
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=skills,
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )

    result = engine.execute(mode="research", goal="inspect the architecture", caller="human")

    assert result["ok"] is True
    assert result["task"]["lesson"]
    assert skills.promotions == 0


def test_replan_reusing_an_id_with_changed_content_executes_the_correction(tmp_path: Path) -> None:
    first = ModelPlan(
        summary="first",
        actions=[Action(id="same", kind=ActionKind.CREATE, path="a.txt", content="a")],
        acceptance_checks=[AcceptanceCheck(name="not yet", argv=["false"])],
    )
    corrected = ModelPlan(
        summary="corrected",
        actions=[Action(id="same", kind=ActionKind.CREATE, path="b.txt", content="b")],
        acceptance_checks=[AcceptanceCheck(name="b exists", argv=["test", "-f", "b.txt"])],
    )
    engine = Engine(
        workspace=tmp_path,
        backend=ScriptedBackend([first, corrected]),
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )

    result = engine.execute(mode="implement", goal="apply a corrected action", caller="human")

    assert result["ok"] is True
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "b"


def test_resume_reconstructs_completed_actions_from_persisted_fingerprints(tmp_path: Path) -> None:
    action = Action(id="old", kind=ActionKind.CREATE, path="once.txt", content="once")
    fingerprint = Engine._action_hash(action)
    from xander_agent.models import ActionResult, ActionStatus, TaskStatus, XanderRequest

    store = TaskStore(root=tmp_path / "tasks")
    task = store.create(
        XanderRequest(mode="implement", workspace=tmp_path, goal="resume exact prior work")
    )
    task.status = TaskStatus.FAILED
    task.attempt = 1
    task.plan = ModelPlan(
        summary="same action after restart",
        actions=[action],
        acceptance_checks=[AcceptanceCheck(name="exists", argv=["test", "-f", "once.txt"])],
    )
    (tmp_path / "once.txt").write_text("once", encoding="utf-8")
    task.results = [
        ActionResult(action_id="old", action_hash=fingerprint, status=ActionStatus.OK)
    ]
    store.save(task)
    engine = Engine(
        workspace=tmp_path,
        backend=ScriptedBackend([]),
        task_store=store,
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )

    resumed = engine.resume(task.id)

    assert resumed["ok"] is True
    assert (tmp_path / "once.txt").read_text(encoding="utf-8") == "once"
