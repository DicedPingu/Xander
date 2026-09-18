"""The single-view terminal: one feed, one composer, no tabs.

Every scenario drives the app the way an operator does — typing a line —
and reads the feed the way an operator does: as text.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from xander_agent.models import Action, ActionKind
from xander_agent.power import PowerStatus
from xander_agent.tui import Composer, XanderApp


def _feed_text(app: XanderApp) -> str:
    feed = app.query_one("#feed", VerticalScroll)
    return "\n".join(str(child.render()) for child in feed.children if isinstance(child, Static))


def _type(app: XanderApp, text: str) -> None:
    composer = app.query_one("#composer", Composer)
    composer.value = text
    app.on_input_submitted(Input.Submitted(composer, text))


def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XANDER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("XANDER_CACHE_DIR", str(tmp_path / "cache"))


def test_there_are_no_tabs_and_the_composer_has_focus(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, power_reader=lambda: PowerStatus(87, "charging", "test"))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app.query_one("#composer", Composer).has_focus
            assert not app.query("TabbedContent") and not app.query("ContentSwitcher")
            status = str(app.query_one("#status", Static).render())
            assert "XANDER" in status and str(tmp_path.name) in status and "87%" in status
            # Feed entries are Static widgets, which Textual lets the mouse select and ctrl+c copy.
            assert all(child.allow_select for child in app.query_one("#feed", VerticalScroll).children)

    asyncio.run(scenario())


def test_every_typed_line_is_echoed_and_history_walks_with_arrows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("xander_agent.tui.invoke_engine", lambda mode, **kw: {"ok": True, "status": "completed", "task": {}})

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            _type(app, "/help")
            _type(app, "/values")
            await pilot.pause()
            text = _feed_text(app)
            assert "you › /help" in text and "you › /values" in text
            composer = app.query_one("#composer", Composer)
            composer.focus()
            await pilot.press("up")
            assert composer.value == "/values"
            await pilot.press("up")
            assert composer.value == "/help"
            await pilot.press("down")
            assert composer.value == "/values"
            await pilot.press("down")
            assert composer.value == ""

    asyncio.run(scenario())


def test_a_small_order_runs_and_xander_replies_in_one_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    calls: list[dict] = []

    def fake_invoke(mode: str, **kwargs) -> dict:
        calls.append({"mode": mode, **kwargs})
        return {
            "ok": True,
            "status": "completed",
            "task_id": "quick-1",
            "task": {
                "id": "quick-1",
                "status": "completed",
                "plan": {"summary": "I created done.txt. What now?"},
                "evidence": [{"kind": "quick", "order": "create_file", "paths": ["done.txt"]}],
                "results": [{"status": "ok", "changed_paths": ["done.txt"]}],
            },
        }

    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, autonomy="full-auto")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            _type(app, "Create an empty file called done.txt in this folder.")
            for _ in range(200):
                await pilot.pause(0.01)
                if app.task_state == "complete":
                    break
            assert calls and calls[0]["mode"] == "implement"
            assert calls[0]["goal"].startswith("Create an empty file")
            assert callable(calls[0]["steering"])
            text = _feed_text(app)
            assert "Xander › I created done.txt. What now?" in text
            assert "ADAPTIVE LOOP" not in text and "Judge" not in text

    asyncio.run(scenario())


def test_lines_typed_mid_run_are_shown_and_reach_the_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    seen: list[dict] = []

    def fake_invoke(mode: str, **kwargs) -> dict:
        import time

        deadline = time.time() + 3
        while time.time() < deadline:
            report = kwargs["steering"]()
            if report["lines"]:
                seen.append(report)
                break
            time.sleep(0.02)
        return {"ok": True, "status": "completed", "task": {}}

    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            _type(app, "build the thing")
            for _ in range(50):
                await pilot.pause(0.01)
                if app.task_state == "running":
                    break
            _type(app, "use tabs not spaces")
            for _ in range(300):
                await pilot.pause(0.01)
                if seen:
                    break
            assert seen and seen[0]["constraints"] == ["use tabs not spaces"]
            text = _feed_text(app)
            assert "you › use tabs not spaces" in text
            assert "heard" in text

    asyncio.run(scenario())


def test_approval_is_asked_in_the_feed_and_answered_by_typing_yes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    outcomes: list[bool] = []

    def fake_invoke(mode: str, **kwargs) -> dict:
        approved = kwargs["approve"](
            Action(kind=ActionKind.COMMAND, argv=["sudo", "apt", "install", "x"], expected="install"),
            "privileged action",
        )
        outcomes.append(bool(approved))
        return {"ok": approved, "status": "completed" if approved else "unverified", "task": {}}

    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            _type(app, "install x")
            for _ in range(200):
                await pilot.pause(0.01)
                if app._pending_approval is not None:
                    break
            assert app.task_state == "approval"
            assert "may I run" in _feed_text(app) and "sudo apt install x" in _feed_text(app)
            _type(app, "yes")
            for _ in range(200):
                await pilot.pause(0.01)
                if outcomes:
                    break
            assert outcomes == [True]

    asyncio.run(scenario())


def test_questions_orders_and_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    calls: list[tuple[str, dict]] = []

    def fake_invoke(mode: str, **kwargs) -> dict:
        calls.append((mode, kwargs))
        return {"ok": True, "status": "completed", "task_id": "t", "task": {}}

    monkeypatch.setattr("xander_agent.tui.invoke_engine", fake_invoke)

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path, autonomy="supervised")
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.task_state = "running"
            app._engine_busy = True
            _type(app, "second order")  # a bare phrase mid-run: heard, not queued
            assert app._order_queue == []
            _type(app, "create the second thing")  # an order mid-run: queued
            assert app._order_queue == [{"goal": "create the second thing", "mode": "implement"}]
            app._engine_busy = False
            app._finish({"ok": True, "status": "completed", "task_id": "t0", "task": {}})
            for _ in range(200):
                await pilot.pause(0.01)
                if any(kw.get("goal") == "create the second thing" for _, kw in calls):
                    break
            assert app._order_queue == []

            app._finish(
                {
                    "ok": False,
                    "status": "waiting_approval",
                    "task_id": "t-q",
                    "task": {
                        "id": "t-q",
                        "status": "waiting_approval",
                        "plan": {"options": [{"id": "alpha", "title": "Alpha"}, {"id": "beta", "title": "Beta"}]},
                    },
                }
            )
            assert app.task_state == "question"
            assert "which way?" in _feed_text(app)
            _type(app, "2")
            for _ in range(200):
                await pilot.pause(0.01)
                if any(mode == "resume" for mode, _ in calls):
                    break
            resume = [kw for mode, kw in calls if mode == "resume"]
            assert resume and resume[0]["selected_options"] == ["beta"]

    asyncio.run(scenario())


def test_slash_lines_never_run_and_conversation_is_not_a_mission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    engine_calls: list[str] = []
    chats: list[str] = []
    monkeypatch.setattr(
        "xander_agent.tui.invoke_engine", lambda mode, **kw: engine_calls.append(kw.get("goal", "")) or {"ok": True}
    )
    monkeypatch.setattr(
        "xander_agent.tui.talk", lambda ws, variant, message, history: chats.append(message) or {"text": "sure."}
    )

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            _type(app, "/research")  # known, but missing its argument
            _type(app, "/rm -rf /")  # unknown: explained, never a shell
            _type(app, "Quick fucking asking so much")
            _type(app, "what do you think of this folder?")
            for _ in range(100):
                await pilot.pause(0.01)
                if len(chats) >= 2:
                    break
            text = _feed_text(app)
            assert "not a Xander command" in text
            assert engine_calls == []
            assert chats == ["Quick fucking asking so much", "what do you think of this folder?"]
            assert "Xander › sure." in text
            _type(app, "/activity")
            assert "not a Xander command" in _feed_text(app).split("you › /activity")[-1]

    asyncio.run(scenario())


def test_self_work_moves_into_xanders_own_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    external, home = tmp_path / "external", tmp_path / "xander"
    external.mkdir()
    home.mkdir()
    calls: list[dict] = []
    monkeypatch.setattr("xander_agent.tui._xander_workspace", lambda: home)
    monkeypatch.setattr("xander_agent.tui.invoke_engine", lambda mode, **kw: calls.append(kw) or {"ok": True})

    async def scenario() -> None:
        app = XanderApp(workspace=external)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            _type(app, "improve yourself")
            for _ in range(200):
                await pilot.pause(0.01)
                if calls:
                    break
            assert calls[0]["workspace"] == home

    asyncio.run(scenario())


def test_power_zero_stops_everything_once(tmp_path: Path) -> None:
    shutdowns: list[str] = []

    async def scenario() -> None:
        app = XanderApp(
            workspace=tmp_path,
            power_reader=lambda: PowerStatus(0, "discharging", "test"),
            power_shutdown=lambda: shutdowns.append("poweroff"),
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._poll_power()
            app._poll_power()
            assert app.task_state == "power-zero"

    asyncio.run(scenario())
    assert shutdowns == ["poweroff"]


def test_stats_command_prints_the_scoreboard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "xander_agent.tui.stats_payload", lambda *a, **kw: {"tasks": 3, "success_rate": 1.0}
    )
    monkeypatch.setattr(
        "xander_agent.tui.render_lines", lambda payload: [f"[bold]scoreboard[/] · {payload['tasks']} task(s)"]
    )

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            _type(app, "/stats")
            await pilot.pause()
            assert "scoreboard · 3 task(s)" in _feed_text(app)

    asyncio.run(scenario())


def test_set_verbose_toggles_and_shows_in_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolate(monkeypatch, tmp_path)

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app._verbose is False
            _type(app, "/set verbose on")
            await pilot.pause()
            assert app._verbose is True
            _type(app, "/values")
            await pilot.pause()
            assert "verbose on" in _feed_text(app)
            _type(app, "/set verbose off")
            await pilot.pause()
            assert app._verbose is False

    asyncio.run(scenario())


def test_model_indicator_appears_after_a_generation_event(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app._current_model == ""
            app._append_event(
                {
                    "type": "delegation",
                    "message": "model completed coder work",
                    "data": {
                        "agent": "Xander",
                        "role": "coder",
                        "operation": "model generation",
                        "status": "completed",
                        "model": "qwen3.8-9b-heretic:latest",
                    },
                }
            )
            await pilot.pause()
            assert app._current_model == "qwen3.8-9b-heretic:latest"
            status = str(app.query_one("#status", Static).render())
            assert "qwen3.8-9b-heretic:latest" in status

    asyncio.run(scenario())


def test_verbose_adds_a_detail_line_for_model_generation(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            event = {
                "type": "delegation",
                "message": "model completed coder work",
                "data": {
                    "agent": "Xander",
                    "role": "coder",
                    "operation": "model generation",
                    "status": "completed",
                    "model": "qwen3.8-9b-heretic:latest",
                    "tokens": 128,
                    "tokens_per_second": 12.9,
                },
            }
            app._append_event(event)
            await pilot.pause()
            before = _feed_text(app)
            assert "coder ran on qwen3.8-9b-heretic:latest" not in before

            app._verbose = True
            app._append_event(event)
            await pilot.pause()
            after = _feed_text(app)
            assert "coder ran on qwen3.8-9b-heretic:latest" in after
            assert "12.9 tok/s" in after

    asyncio.run(scenario())
