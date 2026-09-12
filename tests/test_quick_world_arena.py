"""Small orders, XanderWorld projects, and the arena runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from xander_agent import arena
from xander_agent.models import ActionKind
from xander_agent.quick import parse_quick_order
from xander_agent.world import create_project, ensure_folder_map


@pytest.mark.parametrize(
    ("line", "kind", "path"),
    [
        ("Create an empty file called done.txt in this folder.", "create_file", "done.txt"),
        ("touch done.txt", "create_file", "done.txt"),
        ("make a new file named notes.md inside docs", "create_file", "docs/notes.md"),
        ("make a folder called src", "create_dir", "src"),
        ('write "hello" to greeting.txt', "write_file", "greeting.txt"),
    ],
)
def test_one_line_orders_parse_offline(line: str, kind: str, path: str) -> None:
    order = parse_quick_order(line)
    assert order is not None and order.kind == kind
    assert order.paths[0] == path
    assert order.reply.endswith("What now?")


@pytest.mark.parametrize(
    "line",
    [
        "Quick fucking asking so much",
        "create a file called a.txt and then delete it",
        "create a file called ../../escape.txt",
        "group the files in this folder into subfolders",
        "write a python script fizzbuzz.py that prints fizzbuzz and run it",
    ],
)
def test_anything_else_falls_through_to_the_planner(line: str) -> None:
    assert parse_quick_order(line) is None


def test_rename_and_delete_need_the_file_to_exist(tmp_path: Path) -> None:
    assert parse_quick_order("rename old.txt to new.txt", tmp_path) is None
    (tmp_path / "old.txt").write_text("x")
    order = parse_quick_order("rename old.txt to new.txt", tmp_path)
    assert order is not None and order.actions[0].argv == ["mv", "old.txt", "new.txt"]
    order = parse_quick_order("delete the file old.txt", tmp_path)
    assert order is not None and order.actions[0].argv == ["rm", "old.txt"]
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")
    assert parse_quick_order("delete the folder full", tmp_path) is None  # not a one-liner


def test_quick_run_only_accepts_a_plain_quoted_command() -> None:
    order = parse_quick_order("run `ls -la`")
    assert order is not None and order.actions[0].kind == ActionKind.COMMAND
    assert parse_quick_order("run `ls | grep x`") is None


def test_world_projects_are_numbered_and_mapped(tmp_path: Path) -> None:
    first = create_project("Codewars client", "sign in and solve kata", tmp_path)
    second = create_project("Second thing", root=tmp_path)
    assert first.name == "000 - Codewars client" and second.name == "001 - Second thing"
    assert "sign in and solve kata" in (first / ".folder").read_text()
    assert (tmp_path / ".folder").exists()


def test_folder_map_is_written_once_and_never_clobbered(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text('"""Entry point."""\nprint(1)\n')
    written = ensure_folder_map(tmp_path, "a test project")
    assert written == tmp_path / ".folder"
    body = written.read_text()
    assert "src/" in body and "Entry point." in body
    written.write_text("mine\n")
    assert ensure_folder_map(tmp_path) is None
    assert written.read_text() == "mine\n"


def test_arena_runs_a_scenario_and_judges_deterministically(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_invoke(mode: str, **kwargs) -> dict:
        (kwargs["workspace"] / "done.txt").write_text("")
        return {"ok": True, "status": "completed", "task_id": "t", "task": {"status": "completed", "evidence": []}}

    monkeypatch.setattr("xander_agent.cli.invoke_engine", fake_invoke)
    result = arena.run_scenario(arena.SCENARIOS_BY_NAME["touch-done"], backend=None)
    assert result.passed and result.judge["score"] == 10 and result.judge["source"] == "deterministic"


def test_arena_marks_an_approval_ask_as_waste(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_invoke(mode: str, **kwargs) -> dict:
        kwargs["approve"](object(), "privileged")
        (kwargs["workspace"] / "done.txt").write_text("")
        return {"ok": True, "status": "completed", "task_id": "t", "task": {"status": "completed"}}

    monkeypatch.setattr("xander_agent.cli.invoke_engine", fake_invoke)
    result = arena.run_scenario(arena.SCENARIOS_BY_NAME["touch-done"], backend=None)
    assert not result.passed
    assert any("approval" in why for why in result.failures)


def test_arena_pick_samples_by_tag_and_seed() -> None:
    names = [s.name for s in arena.pick(None, 2, ["one-liner"], seed=1)]
    assert len(names) == 2 and all("one-liner" in arena.SCENARIOS_BY_NAME[n].tags for n in names)
    assert names == [s.name for s in arena.pick(None, 2, ["one-liner"], seed=1)]
