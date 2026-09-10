import asyncio
from pathlib import Path

import pytest
from textual.containers import VerticalScroll
from textual.widgets import ContentSwitcher, Input, RichLog, Select, Static

from xander_agent.models import Action, ActionKind
from xander_agent.tui import XanderApp


def test_tui_mounts_headlessly_and_exposes_direct_work_surfaces(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, variant="ambusher", caller="human")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            context = str(app.query_one("#status-bar", Static).render())
            assert "XANDER" in context
            assert "idle" in context
            assert app.query_one("#focus-summary", Static)
            assert app.query_one("#loop-log", Static)
            assert app.query_one("#view-switcher", ContentSwitcher)
            assert not app.query("#phase-strip")
            assert not app.query("TabbedContent")

            goal = app.query_one("#goal-input", Input)
            goal.value = "temporary goal"
            await pilot.press("alt+n")
            await pilot.pause()
            assert goal.value == ""
            assert app.task_state == "idle"

            assert not [binding for binding in app.active_bindings.values() if binding.binding.show]

    asyncio.run(scenario())


def test_compact_tui_keeps_one_sidebar_scroll_and_a_visible_composer(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            rail = app.query_one("#loop-rail", VerticalScroll)
            assert rail.region.width >= 30
            assert not rail.query(RichLog)
            assert app.query_one("#loop-log", Static).region.height >= 3
            assert app.query_one("#goal-input", Input).region.height == 3
            assert app.query_one("#focus-summary", Static).region.height >= 5

    asyncio.run(scenario())


def test_tui_narrates_events_into_channel_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XANDER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("XANDER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("XANDER_LOG_DIR", str(tmp_path / "logs"))

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
            focus = str(app.query_one("#focus-summary", Static).render())
            assert "NOW" in focus
            assert "pytest -q" in focus
            assert "DECISION" in focus
            assert "Proof: ok" in focus

    asyncio.run(scenario())

    # RichLog rendering is deferred until display; the plain-file twin is the
    # deterministic record that narration actually happened.
    session_log = tmp_path / "logs" / "xander.log"
    content = session_log.read_text(encoding="utf-8")
    assert "[PHASE] research:hunting local truth" in content
    assert "[PROOF OK] pytest -q" in content


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

            app._dispatch("first order", "implement")
            assert app.query_one(ContentSwitcher).current == "activity-view"
            for worker in [worker for worker in app.workers if worker.group == "xander-task"]:
                worker.cancel()
            app._engine_busy = False
            app.task_state = "running"

            # a bare phrase is not an order: it is kept as a draft, nothing queues
            app.on_input_submitted(Input.Submitted(goal_input, "second order"))
            assert app._order_queue == []
            assert app._draft == "second order"
            # /work authorizes the kept draft; while the engine is busy it joins the queue
            app.on_input_submitted(Input.Submitted(goal_input, "/work"))
            assert app._order_queue == [{"goal": "second order", "mode": "implement"}]
            assert app._draft == ""

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


def test_tui_can_answer_a_high_risk_action_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_invoke(mode: str, **kwargs) -> dict:
        approved = kwargs["approve"](
            Action(kind=ActionKind.COMMAND, argv=["python3", "-c", "print('ok')"], expected="run the check"),
            "interpreter execution needs approval",
        )
        calls.append(bool(approved))
        return {
            "ok": approved,
            "status": "completed" if approved else "unverified",
            "task_id": "approval-task",
            "task": {},
            "handoff": {},
        }

    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, caller="human", autonomy="supervised")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._dispatch("run the approved check", "implement")
            for _ in range(100):
                await pilot.pause(0.01)
                if app._pending_approval is not None:
                    break
            assert app.task_state == "approval"
            assert app._pending_approval is not None
            app.on_input_submitted(Input.Submitted(app.query_one("#goal-input", Input), "yes"))
            for _ in range(100):
                await pilot.pause(0.01)
                if calls:
                    break
            assert calls == [True]
            assert app._pending_approval is None

    asyncio.run(scenario())


def test_tui_routes_self_work_back_to_xanders_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    external = tmp_path / "external"
    target = tmp_path / "xander"
    external.mkdir()
    target.mkdir()

    def fake_invoke(mode: str, **kwargs) -> dict:
        calls.append({"mode": mode, **kwargs})
        return {"ok": True, "status": "completed", "task_id": "self-task", "task": {}, "handoff": {}}

    monkeypatch.setattr("xander_agent.tui._xander_workspace", lambda: target)
    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)

    async def scenario() -> None:
        app = XanderApp(workspace=external, caller="human")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            goal = app.query_one("#goal-input", Input)
            app.on_input_submitted(Input.Submitted(goal, "research your soul and improve yourself"))
            for _ in range(100):
                await pilot.pause(0.01)
                if calls:
                    break

    asyncio.run(scenario())
    assert calls
    assert calls[0]["workspace"] == target
    assert calls[0]["goal"] == "research your soul and improve yourself"


def test_tui_live_controls_pass_into_the_next_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    def fake_invoke(mode: str, **kwargs) -> dict:
        calls.append((mode, kwargs))
        return {"ok": True, "status": "completed", "task_id": "t-form", "task": {}, "handoff": {}}

    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, caller="human")
        async with app.run_test(size=(120, 42)) as pilot:
            await pilot.pause()
            app.query_one("#mission-mode", Select).value = "plan"
            app.query_one("#mission-autonomy", Select).value = "supervised"
            app.query_one("#mission-time", Input).value = "7"
            app.query_one("#mission-proof", Input).value = "pytest -q"
            app.query_one("#mission-allowed", Input).value = "README.md, docs"
            app.query_one("#mission-constraints", Input).value = "keep it concise, no new deps"
            app.query_one("#goal-input", Input).value = "make the result explainable"
            app.action_run()
            for _ in range(20):
                await pilot.pause(0.05)
                if calls:
                    break

    asyncio.run(scenario())
    assert calls
    mode, request = calls[0]
    assert mode == "plan"
    assert request["autonomy"] == "supervised"
    assert request["timeout"] == 420
    assert request["acceptance_checks"] == ["pytest -q"]
    assert request["allowed_paths"] == ["README.md", "docs"]
    assert request["constraints"][:2] == ["keep it concise", "no new deps"]
    assert any(item.startswith("Treat command exit codes") for item in request["constraints"])
    assert any(item.startswith("When a check fails") for item in request["constraints"])
    assert request["setup_policy"] == "ask"


def test_questions_are_conversation_not_missions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XANDER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("XANDER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("XANDER_LOG_DIR", str(tmp_path / "logs"))
    calls: list[tuple[str, str, list[dict[str, str]]]] = []

    def fake_talk(workspace: Path, variant: str, message: str, history: list[dict[str, str]]) -> dict:
        calls.append((variant, message, history))
        return {"text": f"answer to {message}", "role": "critic", "stats": {"model": "test-model"}}

    monkeypatch.setattr("xander_agent.tui.talk", fake_talk)
    monkeypatch.setattr(
        "xander_agent.tui.invoke_engine",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chat created a task")),
    )

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, variant="default")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            field = app.query_one("#goal-input", Input)
            app.on_input_submitted(Input.Submitted(field, "What does this workspace contain?"))
            for _ in range(50):
                await pilot.pause(0.05)
                if not app._chat_busy:
                    break
            app.on_input_submitted(Input.Submitted(field, "And what should I inspect first?"))
            for _ in range(50):
                await pilot.pause(0.05)
                if not app._chat_busy:
                    break
            assert app.task_state == "idle"
            assert [turn["role"] for turn in app._conversation] == ["user", "assistant", "user", "assistant"]

    asyncio.run(scenario())
    assert [message for _, message, _ in calls] == [
        "What does this workspace contain?",
        "And what should I inspect first?",
    ]
    assert calls[1][2][-1]["content"] == "answer to What does this workspace contain?"
    transcript = (tmp_path / "logs" / "xander.log").read_text(encoding="utf-8")
    assert "[CHAT] you: What does this workspace contain?" in transcript
    assert "[CHAT] default: answer to What does this workspace contain?" in transcript


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


def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XANDER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("XANDER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("XANDER_LOG_DIR", str(tmp_path / "logs"))


def test_composer_resolves_every_slash_line_and_never_runs_unknown_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "xander_agent.tui.invoke_engine",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a slash line reached the engine")),
    )
    monkeypatch.setattr(
        "xander_agent.tui.talk",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("a slash line reached the model")),
    )

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, caller="human")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            field = app.query_one("#goal-input", Input)

            # /mode and /cd used to be swallowed as "unknown" by the composer;
            # they must route like the session commands they are.
            app.on_input_submitted(Input.Submitted(field, "/mode plan"))
            assert app.mode == "plan"
            assert app.autonomy == "proposal-only"
            sub = tmp_path / "sub"
            sub.mkdir()
            app.on_input_submitted(Input.Submitted(field, "/cd sub"))
            assert app.workspace == sub.resolve()

            for line in ("/rm -rf /", "/bin/sh -c 'echo hi'", "/reserch python", "/"):
                app.on_input_submitted(Input.Submitted(field, line))
                assert field.value == ""
                assert app.task_state == "idle"
            assert app._order_queue == []

            # a pending approval never receives a slash line as its answer
            from threading import Event

            app._pending_approval = {"event": Event(), "approved": False}
            app.task_state = "approval"
            app.on_input_submitted(Input.Submitted(field, "/values"))
            assert app._pending_approval is not None
            app._pending_approval = None
            app.task_state = "idle"

            app.on_input_submitted(Input.Submitted(field, "/help"))
            await pilot.pause()

    asyncio.run(scenario())
    transcript = (tmp_path / "logs" / "xander.log").read_text(encoding="utf-8")
    assert "unknown command explained, not run: /rm" in transcript
    assert "unknown command explained, not run: /bin/sh" in transcript
    assert "unknown command explained, not run: /reserch" in transcript


def test_discussion_first_routing_keeps_the_draft_until_work_is_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(tmp_path, monkeypatch)
    engine_calls: list[tuple[str, dict]] = []
    chat_calls: list[str] = []

    def fake_invoke(mode: str, **kwargs) -> dict:
        engine_calls.append((mode, kwargs))
        return {"ok": True, "status": "completed", "task_id": "t-work", "task": {}, "handoff": {}}

    def fake_talk(workspace: Path, variant: str, message: str, history: list[dict[str, str]]) -> dict:
        chat_calls.append(message)
        return {"text": f"thoughts on {message}", "role": "critic", "stats": {"model": "test-model"}}

    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)
    monkeypatch.setattr("xander_agent.tui.talk", fake_talk)

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, caller="human", autonomy="supervised")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            field = app.query_one("#goal-input", Input)

            # scene-setting opens a discussion: model yes, engine no, draft kept
            app.on_input_submitted(Input.Submitted(field, "This project is going to be about a Codewars client"))
            for _ in range(50):
                await pilot.pause(0.05)
                if not app._chat_busy:
                    break
            assert chat_calls == ["This project is going to be about a Codewars client"]
            assert engine_calls == []
            assert app._draft == "This project is going to be about a Codewars client"
            assert app.task_state == "idle"

            # an unclear phrase in build mode is held, not run
            app.on_input_submitted(Input.Submitted(field, "Codewars client for Android"))
            assert engine_calls == []
            assert app._draft == "Codewars client for Android"
            assert app.task_state == "idle"

            # /work with no argument authorizes the kept draft
            app.on_input_submitted(Input.Submitted(field, "/work"))
            for _ in range(50):
                await pilot.pause(0.05)
                if engine_calls and app.task_state == "complete":
                    break
            assert [(mode, kwargs["goal"]) for mode, kwargs in engine_calls] == [
                ("implement", "Codewars client for Android")
            ]
            assert engine_calls[0][1]["autonomy"] == "supervised"
            assert app._draft == ""

            # an imperative opening authorizes work directly
            app.on_input_submitted(Input.Submitted(field, "Create a README for the client"))
            for _ in range(50):
                await pilot.pause(0.05)
                if len(engine_calls) == 2 and app.task_state == "complete":
                    break
            assert engine_calls[1][1]["goal"] == "Create a README for the client"

    asyncio.run(scenario())
    transcript = (tmp_path / "logs" / "xander.log").read_text(encoding="utf-8")
    assert "draft kept (not run): Codewars client for Android" in transcript


def test_unclear_lines_in_plan_mode_are_prepared_not_implemented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolate(tmp_path, monkeypatch)
    engine_calls: list[tuple[str, dict]] = []

    def fake_invoke(mode: str, **kwargs) -> dict:
        engine_calls.append((mode, kwargs))
        return {"ok": True, "status": "completed", "task_id": "t-plan", "task": {}, "handoff": {}}

    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, caller="human", mode="plan")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            field = app.query_one("#goal-input", Input)
            app.on_input_submitted(Input.Submitted(field, "Codewars client for Android"))
            for _ in range(50):
                await pilot.pause(0.05)
                if engine_calls:
                    break

    asyncio.run(scenario())
    assert [mode for mode, _ in engine_calls] == ["plan"]
    assert engine_calls[0][1]["goal"] == "Codewars client for Android"


def test_goal_addtodo_and_research_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(tmp_path, monkeypatch)
    engine_calls: list[tuple[str, dict]] = []

    def fake_invoke(mode: str, **kwargs) -> dict:
        engine_calls.append((mode, kwargs))
        return {"ok": True, "status": "completed", "task_id": "t-research", "task": {}, "handoff": {}}

    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, caller="human")
        async with app.run_test(size=(120, 48)) as pilot:
            await pilot.pause()
            field = app.query_one("#goal-input", Input)

            app.on_input_submitted(Input.Submitted(field, "/goal ship a verified APK"))
            app.on_input_submitted(Input.Submitted(field, "/goal solve a 3 kyu kata"))
            app.on_input_submitted(Input.Submitted(field, "/addtodo write the profile README"))
            assert engine_calls == []
            assert [goal.text for goal in app._workboard.goals] == ["ship a verified APK", "solve a 3 kyu kata"]
            assert [todo.text for todo in app._workboard.todos] == ["write the profile README"]
            # goals survive a reload of the workspace record
            reloaded = app._workboard_store.load(tmp_path)
            assert [goal.text for goal in reloaded.goals] == ["ship a verified APK", "solve a 3 kyu kata"]
            panel = str(app.query_one("#todo-log", Static).render())
            assert "Goals" in panel and "solve a 3 kyu kata" in panel

            # /research runs read-only research regardless of the build mode wheel
            app.on_input_submitted(Input.Submitted(field, "/research https://docs.python.org/3/library/importlib.html"))
            for _ in range(50):
                await pilot.pause(0.05)
                if engine_calls and app.task_state == "complete":
                    break
            assert [(mode, kwargs["goal"]) for mode, kwargs in engine_calls] == [
                ("research", "https://docs.python.org/3/library/importlib.html")
            ]

            # /work with no draft falls back to the newest open goal
            app.on_input_submitted(Input.Submitted(field, "/work"))
            for _ in range(50):
                await pilot.pause(0.05)
                if len(engine_calls) == 2 and app.task_state == "complete":
                    break
            assert engine_calls[1][0] == "implement"
            assert engine_calls[1][1]["goal"] == "solve a 3 kyu kata"

    asyncio.run(scenario())
    transcript = (tmp_path / "logs" / "xander.log").read_text(encoding="utf-8")
    assert "goal stored: ship a verified APK" in transcript
