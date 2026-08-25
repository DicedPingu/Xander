import asyncio
from pathlib import Path

import pytest
from textual.widgets import Input, RichLog, Static, TabPane

from xander_agent import MANTRA_PHASES
from xander_agent.tui import XanderApp


def test_tui_mounts_headlessly_and_exposes_the_complete_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, variant="ambusher", caller="human")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            phase_strip = str(app.query_one("#phase-strip", Static).render())
            for phase in MANTRA_PHASES:
                assert phase in phase_strip
            assert len(app.query(TabPane)) == 10
            context = str(app.query_one("#context-bar", Static).render())
            assert "variant:ambusher" in context

            goal = app.query_one("#goal-input", Input)
            goal.value = "temporary goal"
            await pilot.press("ctrl+n")
            await pilot.pause()
            assert goal.value == ""
            assert app.task_state == "idle"

    asyncio.run(scenario())


def test_tui_narrates_events_into_channel_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XANDER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("XANDER_CACHE_DIR", str(tmp_path / "cache"))

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, variant="default", caller="human")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            assert app.query_one("#run-log", RichLog) is not None
            assert app.query_one("#tests-log", RichLog) is not None

            app._append_event({"type": "phase", "phase": "research", "message": "hunting local truth"})
            app._append_event(
                {
                    "type": "test",
                    "phase": "test",
                    "message": "ok",
                    "data": {"name": "pytest -q", "result": {"returncode": 0, "status": "ok"}},
                }
            )
            await pilot.pause()
            assert app.phase_index == 4

    asyncio.run(scenario())

    # RichLog rendering is deferred until display; the plain-file twin is the
    # deterministic record that narration actually happened.
    session_log = tmp_path / "state" / "logs" / "xander.log"
    content = session_log.read_text(encoding="utf-8")
    assert "[PHASE] research:hunting local truth" in content
    assert "[TEST]" in content and "rc=0" in content


def test_tui_queues_orders_and_answers_questions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XANDER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("XANDER_CACHE_DIR", str(tmp_path / "cache"))

    calls: list[tuple[str, dict]] = []

    def fake_invoke(mode: str, **kwargs) -> dict:
        calls.append((mode, kwargs))
        return {"ok": True, "status": "completed", "task_id": "t-fake", "task": {}, "handoff": {}}

    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)

    async def scenario() -> None:
        app = XanderApp(
            workspace=tmp_path,
            variant="default",
            caller="human",
            autonomy="supervised",
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            goal_input = app.query_one("#goal-input", Input)

            # orders submitted while the engine is busy join the queue
            app.task_state = "running"
            app.on_input_submitted(Input.Submitted(goal_input, "second order"))
            assert app._order_queue == ["second order"]

            # finishing the current run auto-starts the queued order
            app._finish({"ok": True, "status": "completed", "task_id": "t0", "task": {}, "handoff": {}})
            for _ in range(100):
                await asyncio.sleep(0.05)
                if any(kwargs.get("goal") == "second order" for _, kwargs in calls) and app.task_state == "complete":
                    break
            assert app._order_queue == []
            goal_calls = [kwargs for _, kwargs in calls if kwargs.get("goal") == "second order"]
            assert goal_calls and goal_calls[0]["autonomy"] == "supervised"

            # a waiting_approval result raises a question instead of finishing
            app._finish(
                {
                    "ok": False,
                    "status": "waiting_approval",
                    "task_id": "t-q",
                    "task": {
                        "id": "t-q",
                        "status": "waiting_approval",
                        "plan": {
                            "options": [
                                {"id": "alpha", "title": "Alpha", "summary": "first way"},
                                {"id": "beta", "title": "Beta", "summary": "second way"},
                            ]
                        },
                    },
                    "handoff": {},
                }
            )
            assert app.task_state == "question"
            assert app._pending_choice == {"task_id": "t-q", "options": ["alpha", "beta"]}

            # answering by number resumes the task with the mapped option id
            app.on_input_submitted(Input.Submitted(goal_input, "2"))
            for _ in range(100):
                await asyncio.sleep(0.05)
                if any(mode == "resume" for mode, _ in calls):
                    break
            resume_calls = [kwargs for mode, kwargs in calls if mode == "resume"]
            assert resume_calls and resume_calls[0]["task_id"] == "t-q"
            assert resume_calls[0]["selected_options"] == ["beta"]
            assert resume_calls[0]["autonomy"] == "supervised"
            assert app._pending_choice is None

            # ctrl+n dismisses a stale question so the next goal is not
            # swallowed as an answer
            app._pending_choice = {"task_id": "t-stale", "options": ["a"]}
            app.action_new()
            assert app._pending_choice is None

    asyncio.run(scenario())


def test_tui_task_refresh_filters_foreign_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xander_agent.models import XanderRequest
    from xander_agent.tasks import TaskStore

    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XANDER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("XANDER_CACHE_DIR", str(tmp_path / "cache"))
    own_workspace = tmp_path / "own"
    foreign_workspace = tmp_path / "foreign"
    own_workspace.mkdir()
    foreign_workspace.mkdir()
    store = TaskStore(root=tmp_path / "tasks")
    own = store.create(XanderRequest(mode="inspect", workspace=own_workspace, goal="own task"))
    foreign = store.create(
        XanderRequest(mode="inspect", workspace=foreign_workspace, goal="foreign task")
    )
    monkeypatch.setattr("xander_agent.tasks.TaskStore", lambda: store)
    captured: dict[str, list[str]] = {}
    app = XanderApp(workspace=own_workspace)
    monkeypatch.setattr(
        app,
        "_fill_log",
        lambda selector, lines: captured.__setitem__(selector, lines),
    )

    app._refresh_tasks()

    rendered = "\n".join(captured["#tasks-log"])
    assert own.id in rendered and "own task" in rendered
    assert foreign.id not in rendered and "foreign task" not in rendered
