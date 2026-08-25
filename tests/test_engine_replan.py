"""The replan guard must stop as soon as the model repeats itself."""

import json
from pathlib import Path

from xander_agent.engine import Engine
from xander_agent.memory import MemoryStore
from xander_agent.models import AcceptanceCheck, Action, ActionKind, ModelPlan, ResearchBundle
from xander_agent.tasks import TaskStore


class ScriptedBackend:
    def __init__(self, plans: list[ModelPlan]) -> None:
        self.plans = list(plans)

    def available(self) -> bool:
        return True

    def generate(self, prompt, role="coder", schema=None, think=False, timeout=None) -> str:
        if schema is not None and schema.__name__ == "Analysis":
            return json.dumps({"subject": "replan probe", "constraints": [], "task_type": "coding", "complexity": 2})
        return self.plans.pop(0).model_dump_json()


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
