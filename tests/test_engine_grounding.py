"""The "purge stegosuite" loop from the 2026-09-12 screenshot, piece by piece.

He asked to run ``apt purge`` and then ran it as the user (permission denied);
he planned ``rm remove-stegosuite.sh`` for a script nobody wrote; and he
alternated between those two failing plans without noticing. Each of those
is one guard here. The gear-up step ("upgrade yourself before the work")
is the fourth.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from xander_agent.engine import Engine
from xander_agent.executor import ActionExecutor
from xander_agent.memory import MemoryStore
from xander_agent.models import (
    AcceptanceCheck,
    Action,
    ActionKind,
    ModelPlan,
    ResearchBundle,
    WorkspaceSnapshot,
    XanderEvent,
)
from xander_agent.policy import harden_argv, needs_root
from xander_agent.readiness import ToolNeed, ToolReadiness
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
            return "LGTM"
        if schema.__name__ == "Analysis":
            return json.dumps({"subject": "probe", "constraints": [], "task_type": "coding", "complexity": 2})
        if not self.plans:
            raise RuntimeError("no scripted plan left")
        return self.plans.pop(0).model_dump_json()


class StubSkills:
    def load_selected(self, query: str, limit: int = 3, max_chars: int = 24_000) -> list[dict[str, str]]:
        return []


class StubResearcher:
    def gather(self, goal: str, subject: str = "") -> ResearchBundle:
        return ResearchBundle()


def _engine(tmp_path: Path, backend, **kwargs) -> Engine:
    return Engine(
        workspace=tmp_path,
        backend=backend,
        task_store=TaskStore(root=tmp_path / "tasks"),
        skill_registry=StubSkills(),
        researcher=StubResearcher(),
        memory=MemoryStore(path=tmp_path / "memory.json"),
        **kwargs,
    )


# -- root and unattended flags --------------------------------------------------
def test_package_mutations_need_root_and_queries_do_not() -> None:
    assert needs_root(["apt", "purge", "stegosuite"])
    assert needs_root(["apt-get", "install", "-y", "wabt"])
    assert needs_root(["dpkg", "-P", "stegosuite"])
    assert needs_root(["snap", "remove", "thing"])
    assert not needs_root(["dpkg-query", "-W", "stegosuite"])
    assert not needs_root(["apt", "list", "--installed"])
    assert not needs_root(["sudo", "apt", "purge", "x"]), "already elevated"
    assert not needs_root(["pip", "uninstall", "-y", "x"]), "user-level manager"


def test_harden_argv_makes_apt_purge_runnable_unattended() -> None:
    if os.geteuid() == 0:
        return  # root needs no sudo; nothing to harden
    assert harden_argv(["apt", "purge", "stegosuite"]) == ["sudo", "-n", "apt-get", "purge", "-y", "stegosuite"]
    assert harden_argv(["apt-get", "purge", "-y", "stegosuite"]) == ["sudo", "-n", "apt-get", "purge", "-y", "stegosuite"]
    assert harden_argv(["ls", "-la"]) == ["ls", "-la"]
    assert harden_argv(["dpkg-query", "-W", "x"]) == ["dpkg-query", "-W", "x"]


def test_executor_asks_about_the_hardened_command_not_the_soft_one(tmp_path: Path) -> None:
    asked: list[list[str]] = []

    def approve(action: Action, reason: str) -> bool:
        asked.append(list(action.argv))
        return False  # decline; the point is what was shown

    executor = ActionExecutor(tmp_path, WorkspaceSnapshot(root=str(tmp_path)), approve=approve)
    action = Action(kind=ActionKind.COMMAND, argv=["apt", "purge", "stegosuite"], expected="gone")
    result = executor.run(action)

    assert result.status == "blocked"
    assert asked and asked[0][:3] == ["sudo", "-n", "apt-get"]
    assert action.argv[:2] == ["sudo", "-n"], "the plan record shows what would really run"


# -- phantom files -------------------------------------------------------------------
def test_phantom_lint_names_the_missing_file_and_what_is_really_here(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    engine = _engine(tmp_path, ScriptedBackend([]))
    plan = ModelPlan(
        summary="remove via a script",
        actions=[Action(kind=ActionKind.COMMAND, argv=["rm", "remove-stegosuite.sh"], expected="remove it")],
    )

    failure = engine._phantom_failure(plan)

    assert "remove-stegosuite.sh" in failure
    assert "notes.txt" in failure
    assert "do not exist" in failure


def test_phantom_lint_accepts_files_the_plan_creates_first(tmp_path: Path) -> None:
    engine = _engine(tmp_path, ScriptedBackend([]))
    plan = ModelPlan(
        summary="write then run",
        actions=[
            Action(kind=ActionKind.CREATE, path="game.py", content="print(1)\n", expected="the game"),
            Action(kind=ActionKind.COMMAND, argv=["python3", "game.py"], expected="runs"),
            Action(kind=ActionKind.COMMAND, argv=["rm", "-rf", "target"], expected="cleanup of a real dir"),
        ],
    )
    (tmp_path / "target").mkdir()

    assert engine._phantom_failure(plan) == ""


def test_phantom_plan_is_replanned_with_the_precise_reason(tmp_path: Path) -> None:
    phantom = ModelPlan(
        summary="phantom",
        actions=[Action(kind=ActionKind.COMMAND, argv=["rm", "ghost.sh"], expected="remove the helper")],
        acceptance_checks=[AcceptanceCheck(name="passes", argv=["true"])],
    )
    grounded = ModelPlan(
        summary="grounded",
        actions=[Action(kind=ActionKind.CREATE, path="done.txt", content="ok\n", expected="marker")],
        acceptance_checks=[AcceptanceCheck(name="exists", argv=["test", "-f", "done.txt"])],
    )
    backend = ScriptedBackend([phantom, grounded])
    result = _engine(tmp_path, backend).execute(mode="implement", goal="prove the phantom guard", caller="human")

    assert result["ok"] is True
    assert result["task"]["attempt"] == 2
    replan_prompt = [p for p in backend.prompts if "structured coding plan" in p][-1]
    assert "ghost.sh" in replan_prompt and "do not exist" in replan_prompt
    assert not any(r["action_id"] for r in result["task"]["results"] if r["status"] == "failed"), (
        "the phantom command never reached the executor"
    )


# -- A, B, A is not a new approach ------------------------------------------------------------
def test_replanning_notices_an_approach_from_two_attempts_ago(tmp_path: Path) -> None:
    def plan(expected: str) -> ModelPlan:
        return ModelPlan(
            summary=expected,
            actions=[Action(kind=ActionKind.INSPECT, argv=["true"], expected=expected)],
            acceptance_checks=[AcceptanceCheck(name="never", argv=["false"])],
        )

    backend = ScriptedBackend([plan("A"), plan("B"), plan("A"), plan("B"), plan("A")])
    result = _engine(tmp_path, backend).execute(mode="implement", goal="alternate between two walls", caller="human")

    assert result["ok"] is False
    assert result["task"]["attempt"] == 2, "the third plan repeats the first and is never run"
    assert "repeated the same approach" in result["task"]["failure"]


# -- gear up before the work ---------------------------------------------------------------------
def test_gear_up_installs_a_missing_tool_through_the_executor(tmp_path: Path, monkeypatch) -> None:
    marker = tmp_path / "installed.flag"
    need = ToolNeed("flag tool", ("flagtool",), "prove the gear-up", (f"touch {marker.name}",))
    calls: list[str] = []

    def fake_assess(goal, workspace=None, commands=(), *, brain=None):
        calls.append("assess")
        present = marker.exists()
        return ToolReadiness(
            query=goal,
            needs=(need,),
            available={"flagtool": "/usr/bin/flagtool"} if present else {},
            missing=() if present else ("flag tool",),
        )

    monkeypatch.setattr("xander_agent.engine.assess", fake_assess)
    plan = ModelPlan(
        summary="done",
        actions=[Action(kind=ActionKind.CREATE, path="out.txt", content="x\n", expected="output")],
        acceptance_checks=[AcceptanceCheck(name="exists", argv=["test", "-f", "out.txt"])],
    )
    events: list[XanderEvent] = []
    engine = _engine(tmp_path, ScriptedBackend([plan]), event_sink=events.append)

    result = engine.execute(mode="implement", goal="needs a tool first", caller="human", setup_policy="allow")

    assert result["ok"] is True
    assert marker.exists(), "the install suggestion ran before the first plan"
    assert any(e["kind"] == "gear_up" and e["installed"] == ["flag tool"] for e in result["task"]["evidence"])
    assert any(e.type == "voice" and "upgraded myself" in e.message for e in events)
    assert any(e.type == "research" and e.message.startswith("geared up:") for e in events)
    assert calls.count("assess") >= 2, "readiness is re-checked after the install"


def test_gear_up_asks_the_operator_when_policy_is_ask_and_someone_is_there(tmp_path: Path, monkeypatch) -> None:
    need = ToolNeed("thing", ("thing",), "purpose", ("touch got-it",))

    def fake_assess(goal, workspace=None, commands=(), *, brain=None):
        present = (tmp_path / "got-it").exists()
        return ToolReadiness(query=goal, needs=(need,), available={}, missing=() if present else ("thing",))

    monkeypatch.setattr("xander_agent.engine.assess", fake_assess)
    asked: list[str] = []
    plan = ModelPlan(
        summary="done",
        actions=[Action(kind=ActionKind.CREATE, path="out.txt", content="x\n", expected="output")],
        acceptance_checks=[AcceptanceCheck(name="exists", argv=["test", "-f", "out.txt"])],
    )
    engine = _engine(
        tmp_path,
        ScriptedBackend([plan]),
        approve=lambda action, reason: asked.append(" ".join(action.argv)) or True,
    )

    result = engine.execute(mode="implement", goal="needs a tool first", caller="human", setup_policy="ask")

    # `touch got-it` is contained, so no approval is needed for it; the point
    # is that "ask" with a human present proceeds instead of pausing the mission.
    assert result["ok"] is True
    assert (tmp_path / "got-it").exists()


def test_gear_up_pauses_headless_when_policy_is_ask(tmp_path: Path, monkeypatch) -> None:
    need = ToolNeed("thing", ("thing",), "purpose", ("touch got-it",))

    def fake_assess(goal, workspace=None, commands=(), *, brain=None):
        return ToolReadiness(query=goal, needs=(need,), available={}, missing=("thing",))

    monkeypatch.setattr("xander_agent.engine.assess", fake_assess)
    engine = _engine(tmp_path, ScriptedBackend([]))

    result = engine.execute(mode="implement", goal="needs a tool first", caller="human", setup_policy="ask")

    assert result["status"] == "waiting_approval"
    assert not (tmp_path / "got-it").exists()


# -- typing mid-run must not kill the mission --------------------------------------------------
def test_steering_events_validate() -> None:
    event = XanderEvent(run_id="t", sequence=1, type="steering", message="you said: use bash")
    assert event.type == "steering"


# -- dependencies named by path, not id --------------------------------------------------------
def test_dependencies_named_by_path_resolve_to_the_creating_step() -> None:
    plan = ModelPlan(
        summary="wasm",
        actions=[
            Action(id="one", kind=ActionKind.CREATE, path="add.wat", content="(module)", expected="the module"),
            Action(id="two", kind=ActionKind.COMMAND, argv=["wat2wasm", "add.wat"], depends_on=["add.wat"]),
            Action(id="three", kind=ActionKind.COMMAND, argv=["node", "run.js"], depends_on=["the module", "ghost"]),
        ],
    )
    Engine._resolve_dependencies(plan)
    assert plan.actions[1].depends_on == ["one"]
    assert plan.actions[2].depends_on == ["one"], "unknown labels are dropped, known ones map to ids"


def test_commands_the_order_names_become_checks_the_planner_cannot_drop() -> None:
    goal = (
        "Make a cargo project, build it with cargo build, run cargo test, then run it on sample.txt. "
        "Also run python3 snake.py --demo and run node run.js."
    )
    assert Engine._goal_commands(goal) == [["cargo", "test"], ["python3", "snake.py", "--demo"], ["node", "run.js"]]
    plan = ModelPlan(
        summary="x",
        actions=[Action(kind=ActionKind.CREATE, path="a.txt", content="a\n")],
        acceptance_checks=[AcceptanceCheck(name="build", argv=["cargo", "build"])],
    )
    added = Engine._synthesize_checks(plan, goal)
    assert [c.argv for c in added] == [["cargo", "test"], ["python3", "snake.py", "--demo"], ["node", "run.js"]]
    assert all(c.required for c in added)


def test_unified_patch_marks_missing_final_newlines_the_way_git_does(tmp_path: Path) -> None:
    import subprocess

    marker = "\\ No newline at end of file\n"
    assert Engine._unified_patch("a\nb\n", "a\nc", "f.txt").endswith("+c\n" + marker)

    (tmp_path / "f.txt").write_text("a\nb", encoding="utf-8")  # the file on disk lacks one
    patch = Engine._unified_patch("a\nb", "a\nc\n", "f.txt")
    assert marker in patch
    (tmp_path / "p.patch").write_text(patch, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    check = subprocess.run(["git", "apply", "--check", "p.patch"], cwd=tmp_path, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_reading_github_is_not_privileged() -> None:
    from xander_agent.models import Risk
    from xander_agent.policy import _argv_risk

    assert _argv_risk(["gh", "search", "repos", "local llm agent", "--limit", "5"]) == Risk.LOW
    assert _argv_risk(["gh", "repo", "view", "owner/name"]) == Risk.LOW
    assert _argv_risk(["gh", "api", "repos/o/n"]) == Risk.LOW
    assert _argv_risk(["gh", "api", "-X", "POST", "repos/o/n/issues"]) == Risk.HIGH
    assert _argv_risk(["gh", "pr", "create"]) == Risk.HIGH
    assert _argv_risk(["gh", "repo", "delete", "x"]) == Risk.HIGH


def test_a_check_on_a_built_binary_finds_it_under_target(tmp_path: Path) -> None:
    (tmp_path / "target" / "debug").mkdir(parents=True)
    (tmp_path / "target" / "debug" / "wordfreq").write_text("", encoding="utf-8")
    engine = _engine(tmp_path, ScriptedBackend([]))
    assert engine._ground_check_argv(["./wordfreq", "sample.txt"]) == ["target/debug/wordfreq", "sample.txt"]
    assert engine._ground_check_argv(["./missing"]) == ["./missing"]
    assert engine._ground_check_argv(["cargo", "test"]) == ["cargo", "test"]


def test_a_goal_named_output_that_is_only_an_intention_fails_the_judgment(tmp_path: Path) -> None:
    engine = _engine(tmp_path, ScriptedBackend([]))
    from xander_agent.models import XanderRequest

    task = engine.task_store.create(XanderRequest(mode="implement", workspace=tmp_path, goal="write the findings to repos.md"))
    (tmp_path / "repos.md").write_text("I'll search for 8 repositories across the required categories.", encoding="utf-8")
    assert "placeholder" in engine._stub_output_failure(task)
    (tmp_path / "repos.md").write_text("## a/b\nhttps://example.com/a/b\nA tool.\nTake the loop.\n\n## c/d\n...\n", encoding="utf-8")
    assert "placeholder" in engine._stub_output_failure(task), "an ellipsis stub is still a stub"
    (tmp_path / "repos.md").write_text("## a/b\nhttps://example.com/a/b\nA tool.\nTake the loop.\n", encoding="utf-8")
    assert engine._stub_output_failure(task) == ""
