import asyncio
from pathlib import Path

from textual.widgets import Input

from xander_agent.tui import XanderApp
from xander_agent.workboard import WorkboardStore


def test_workboard_persists_todos_learning_and_contest_comments(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WorkboardStore(root=tmp_path / "boards")
    board = store.load(workspace)

    first = store.add_todo(board, "  inspect  the real output ")
    assert first is not None
    store.set_learning(board, "container behavior", ["runtime docs", "runtime docs", "command output"])
    assert store.add_comment(board, "stop before changing the retry policy")
    assert store.add_observed_lesson(board, "exit codes beat intended behavior")
    store.set_todo_state(board, first.id, "active")

    loaded = store.load(workspace)
    assert loaded.todos[0].text == "inspect the real output"
    assert loaded.todos[0].state == "active"
    assert loaded.learning_sources == ["runtime docs", "command output"]
    assert loaded.contest_comments == ["stop before changing the retry policy"]
    assert loaded.observed_lessons == ["exit codes beat intended behavior"]


def test_workboard_stores_goals_once_and_lists_only_open_ones(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WorkboardStore(root=tmp_path / "boards")
    board = store.load(workspace)

    first = store.add_goal(board, "  ship a  verified APK ")
    assert first is not None and first.text == "ship a verified APK"
    assert store.add_goal(board, "Ship A Verified APK") is first  # repeated open goal is not duplicated
    second = store.add_goal(board, "solve a 3 kyu kata")
    assert store.add_goal(board, "   ") is None
    assert [goal.text for goal in store.open_goals(board)] == ["ship a verified APK", "solve a 3 kyu kata"]

    assert store.set_goal_state(board, first.id, "done", "apk built and installed")
    assert not store.set_goal_state(board, "missing", "done")

    loaded = store.load(workspace)
    assert [goal.text for goal in store.open_goals(loaded)] == ["solve a 3 kyu kata"]
    assert loaded.goals[0].state == "done"
    assert loaded.goals[0].evidence == "apk built and installed"
    assert loaded.goals[1].id == second.id


def test_workboard_without_goals_field_still_loads(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = WorkboardStore(root=tmp_path / "boards")
    board = store.load(workspace)
    store.add_todo(board, "older record")
    path = store.path(workspace)
    data = path.read_text(encoding="utf-8").replace('  "goals": [],\n', "")
    path.write_text(data, encoding="utf-8")
    assert '"goals"' not in data
    loaded = store.load(workspace)
    assert loaded.goals == []
    assert loaded.todos[0].text == "older record"


def test_todo_sequence_runs_one_item_at_a_time_and_records_lessons(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XANDER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("XANDER_CACHE_DIR", str(tmp_path / "cache"))
    calls: list[str] = []

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(120, 48)) as pilot:
            await pilot.pause()
            app._run_goal = lambda goal, mode: calls.append(goal)
            field = app.query_one("#todo-input", Input)
            for value in ("first step", "second step"):
                field.value = value
                app._add_todo()
            app._run_todo_sequence()
            assert calls == ["first step"]
            assert [item.state for item in app._workboard.todos] == ["active", "todo"]

            app._finish({"ok": True, "status": "completed", "task_id": "t1", "task": {"lesson": "first observed"}})
            assert calls == ["first step", "second step"]
            assert [item.state for item in app._workboard.todos] == ["done", "active"]

            app._finish({"ok": True, "status": "completed", "task_id": "t2", "task": {"lesson": "second observed"}})
            assert [item.state for item in app._workboard.todos] == ["done", "done"]
            assert app._todo_sequence_active is False
            assert app._workboard.observed_lessons == ["first observed", "second observed"]

    asyncio.run(scenario())


def test_contest_shortcut_stops_work_and_leaves_todo_pending(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XANDER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("XANDER_CACHE_DIR", str(tmp_path / "cache"))

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(120, 48)) as pilot:
            await pilot.pause()
            field = app.query_one("#todo-input", Input)
            field.value = "challenge the new logic"
            app._add_todo()
            app._run_goal = lambda goal, mode: None
            app._run_todo_sequence()
            app._engine_busy = True
            app.action_contest()
            assert app.task_state == "needs-attention"
            assert app._todo_sequence_active is False
            assert app._workboard.todos[0].state == "todo"
            contest = app.query_one("#contest-input", Input)
            contest.value = "keep the old route until the new check passes"
            app._record_contest_comment()
            assert app._workboard.contest_comments == ["keep the old route until the new check passes"]

    asyncio.run(scenario())
