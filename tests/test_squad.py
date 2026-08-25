"""The squad: Master judges, Lurker digs, and Xander picks a form per mission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xander_agent.engine import Engine
from xander_agent.memory import MemoryStore
from xander_agent.models import AcceptanceCheck, Action, ActionKind, ModelPlan, ResearchBundle
from xander_agent.squad import LURKER, MASTER, Squad, choose_form
from xander_agent.tasks import TaskStore


# -- muster heuristics (no model needed) --------------------------------------
def test_master_always_walks_with_mutating_missions() -> None:
    squad = Squad.muster("make the background green", complexity=1, has_checks=False, mode="implement")
    assert squad.master and not squad.lurker


def test_size_budget_summons_the_lurker() -> None:
    squad = Squad.muster(
        "create a website with the game of tic tac toe that is less than 30 KB",
        complexity=2,
        has_checks=False,
        mode="implement",
    )
    assert squad.helpers == [MASTER, LURKER]


def test_high_complexity_summons_the_lurker_without_a_budget() -> None:
    squad = Squad.muster("migrate the whole architecture", complexity=4, has_checks=True, mode="implement")
    assert squad.lurker


def test_read_only_missions_walk_alone() -> None:
    squad = Squad.muster("explain the executor", complexity=1, has_checks=False, mode="research")
    assert squad.helpers == []


@pytest.mark.parametrize(
    ("mode", "goal", "expected"),
    [
        ("implement", "make the background green", "Smith"),
        ("implement", "fix the failing test in tests/", "Medic"),
        ("test-triage", "why is CI red", "Medic"),
        ("research", "how do wasm bundles shrink", "Lurker"),
        ("answer", "should I use flexbox", "Lurker"),
        ("plan", "propose a refactor", "Master"),
    ],
)
def test_choose_form_is_honest_about_the_work(mode: str, goal: str, expected: str) -> None:
    assert choose_form(mode, goal) == expected


# -- bounded reviews -----------------------------------------------------------
class GrumpyCritic:
    def __init__(self) -> None:
        self.calls = 0

    def available(self) -> bool:
        return True

    def generate(self, prompt, role="coder", schema=None, think=False, timeout=None) -> str:
        self.calls += 1
        return "This plan ignores the size budget entirely."


def test_master_reviews_are_capped_at_two_per_task() -> None:
    squad = Squad.muster("build it", complexity=1, has_checks=False, mode="implement")
    backend = GrumpyCritic()
    assert squad.master_review(backend, "build it", "plan A") != ""
    assert squad.master_review(backend, "build it", "plan B") != ""
    assert squad.master_review(backend, "build it", "plan C") == ""
    assert backend.calls == 2


def test_helpers_never_speak_when_the_backend_is_down() -> None:
    class Down:
        def available(self) -> bool:
            return False

    squad = Squad.muster("build it under 10 KB", complexity=3, has_checks=False, mode="implement")
    assert squad.master_review(Down(), "g", "p") == ""
    assert squad.master_verdict(Down(), "g", "e") == ""
    assert squad.lurker_brief(Down(), "g") == ""


# -- the Master's verdict gate through the engine ------------------------------
class StubSkills:
    def load_selected(self, query: str, limit: int = 3, max_chars: int = 24_000) -> list[dict[str, str]]:
        return []


class StubResearcher:
    def gather(self, goal: str, subject: str = "") -> ResearchBundle:
        return ResearchBundle()


def _plan(expected: str) -> ModelPlan:
    return ModelPlan(
        summary=f"plan: {expected}",
        actions=[Action(kind=ActionKind.INSPECT, argv=["true"], expected=expected)],
        acceptance_checks=[AcceptanceCheck(name="passes", argv=["true"])],
    )


class UnhappyThenHappyBackend:
    """Plans succeed; the Master rejects the first result, accepts the second."""

    def __init__(self) -> None:
        self.plans = [_plan("first shape"), _plan("second shape")]
        self.critic_replies = ["The checks are trivial; nothing proves the goal.", "LGTM"]

    def available(self) -> bool:
        return True

    def generate(self, prompt, role="coder", schema=None, think=False, timeout=None) -> str:
        if schema is None:
            if "judging the finished work" in prompt:
                return self.critic_replies.pop(0)
            return "LGTM"  # plan reviews stay content
        if schema.__name__ == "Analysis":
            return json.dumps({"subject": "verdict probe", "constraints": [], "task_type": "coding", "complexity": 2})
        return self.plans.pop(0).model_dump_json()


def test_an_unhappy_master_buys_exactly_one_reshaped_attempt(tmp_path: Path) -> None:
    backend = UnhappyThenHappyBackend()
    engine = Engine(
        workspace=tmp_path,
        backend=backend,
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
    )
    events = []
    engine.event_sink = events.append

    result = engine.execute(mode="implement", goal="satisfy the master", caller="human")

    assert result["ok"] is True
    task = result["task"]
    assert task["attempt"] == 2, "the unhappy verdict must trigger exactly one replan"
    verdicts = [item for item in task["evidence"] if item.get("kind") == "master_verdict"]
    assert [item["happy"] for item in verdicts] == [False, True]
    voices = [event for event in events if event.type == "voice"]
    speakers = {event.data.get("speaker") for event in voices}
    assert MASTER in speakers, "the Master must speak his verdict aloud"
    progress = [event for event in voices if event.data.get("moment") == "progress"]
    assert progress, "the worker must report progress to the Master"
    assert progress[0].message.startswith("Master — attempt 1")
