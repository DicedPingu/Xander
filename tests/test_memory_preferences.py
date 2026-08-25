"""Preference learning: standing likes/dislikes persist and reach the planner."""

from __future__ import annotations

import json
from pathlib import Path

from xander_agent.engine import Engine
from xander_agent.memory import MemoryStore
from xander_agent.models import AcceptanceCheck, Action, ActionKind, ModelPlan, ResearchBundle
from xander_agent.tasks import TaskStore


def test_add_preference_persists_and_roundtrips(tmp_path: Path) -> None:
    store = MemoryStore(path=tmp_path / "memory.json")
    assert store.add_preference("no emoji in commit messages", kind="dislike") is True
    reloaded = MemoryStore(path=tmp_path / "memory.json")
    rows = reloaded.preferences()
    assert len(rows) == 1
    assert rows[0]["text"] == "no emoji in commit messages"
    assert rows[0]["kind"] == "dislike"
    assert rows[0]["source"] == "explicit"


def test_repeated_preference_gains_weight_instead_of_duplicating(tmp_path: Path) -> None:
    store = MemoryStore(path=tmp_path / "memory.json")
    assert store.add_preference("short summaries", kind="like") is True
    assert store.add_preference("Short summaries", kind="like") is False
    rows = store.preferences()
    assert len(rows) == 1
    assert rows[0]["weight"] == 1.5


def test_inferred_repeat_never_downgrades_explicit_source(tmp_path: Path) -> None:
    store = MemoryStore(path=tmp_path / "memory.json")
    store.add_preference("prefer uv over pip", kind="style", source="explicit")
    store.add_preference("prefer uv over pip", kind="style", source="inferred")
    assert store.preferences()[0]["source"] == "explicit"


def test_invalid_kind_or_empty_text_is_rejected(tmp_path: Path) -> None:
    store = MemoryStore(path=tmp_path / "memory.json")
    assert store.add_preference("", kind="like") is False
    assert store.add_preference("whatever", kind="rant") is False
    assert store.preferences() == []


def test_preference_lines_are_weighted_and_prompt_ready(tmp_path: Path) -> None:
    store = MemoryStore(path=tmp_path / "memory.json")
    store.add_preference("light background colors", kind="like")
    store.add_preference("walls of comments", kind="dislike")
    store.add_preference("walls of comments", kind="dislike")  # weight bump
    lines = store.preference_lines(limit=2)
    assert lines[0] == "[dislike] walls of comments"
    assert lines[1] == "[like] light background colors"


class PromptRecordingBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def available(self) -> bool:
        return True

    def generate(self, prompt, role="coder", schema=None, think=False, timeout=None, **kwargs) -> str:
        self.prompts.append(prompt)
        if schema is not None and getattr(schema, "__name__", "") == "Analysis":
            return json.dumps({"subject": "probe", "constraints": [], "task_type": "coding", "complexity": 2})
        return ModelPlan(
            summary="probe plan",
            actions=[Action(kind=ActionKind.INSPECT, argv=["true"], expected="probe")],
            acceptance_checks=[AcceptanceCheck(name="passes", argv=["true"])],
        ).model_dump_json()


class StubSkills:
    def load_selected(self, query: str, limit: int = 3, max_chars: int = 24_000) -> list[dict[str, str]]:
        return []


class StubResearcher:
    def gather(self, goal: str, subject: str = "") -> ResearchBundle:
        return ResearchBundle()


def test_planner_prompt_carries_operator_preferences(tmp_path: Path) -> None:
    memory = MemoryStore(path=tmp_path / "memory.json")
    memory.add_preference("green backgrounds", kind="like")
    backend = PromptRecordingBackend()
    engine = Engine(
        workspace=tmp_path,
        backend=backend,
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=memory,
    )
    result = engine.execute(mode="implement", goal="paint the landing page", caller="human")
    assert result["ok"] is True
    plan_prompts = [prompt for prompt in backend.prompts if "OPERATOR PREFERENCES" in prompt]
    assert plan_prompts, "the planner prompt must carry an OPERATOR PREFERENCES block"
    assert "[like] green backgrounds" in plan_prompts[0]
