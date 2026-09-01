import asyncio
from pathlib import Path

from textual.widgets import ContentSwitcher, Input

from xander_agent.tui import XanderApp


def test_alt_reaches_every_view_while_the_operator_is_mid_sentence(tmp_path: Path) -> None:
    """A focused Input eats plain letters, so single-key nav is unusable while
    typing. Alt chords are not consumed, and the draft must survive."""

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            field = app.query_one("#goal-input", Input)
            field.focus()
            await pilot.pause()
            for character in "learn httpx and":
                await pilot.press(character)
            await pilot.pause()

            switcher = app.query_one("#view-switcher", ContentSwitcher)
            for key, expected in (
                ("alt+3", "evidence-view"),
                ("alt+h", "history-view"),
                ("alt+1", "activity-view"),
                ("alt+c", "controls-view"),
                ("alt+x", "system-view"),
                ("alt+2", "controls-view"),
                ("alt+a", "activity-view"),
            ):
                await pilot.press(key)
                await pilot.pause()
                assert switcher.current == expected, f"{key} went to {switcher.current}"

            await pilot.press("alt+b")
            await pilot.pause()
            assert app.screen.has_class("loop-collapsed")

            await pilot.press("alt+l")
            await pilot.pause()
            assert field.has_focus
            # The half-written prompt was never touched by any of that.
            assert field.value == "learn httpx and"

    asyncio.run(scenario())


def _line(app: XanderApp, payload: dict) -> str | None:
    return app._human_event_line(payload, "")


def test_the_feed_drops_ceremony_and_keeps_facts(tmp_path: Path) -> None:
    """Every line should carry a fact. Status words about status are not facts."""

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            # "Master started/completed — plan review" said nothing, twice.
            assert _line(app, {"type": "delegation", "message": "plan review",
                               "data": {"agent": "Master", "status": "started"}}) is None
            assert _line(app, {"type": "delegation", "message": "plan review",
                               "data": {"agent": "Master", "status": "completed"}}) is None
            # A delegation that actually failed is worth a line.
            failed = _line(app, {"type": "delegation", "message": "plan review",
                                 "data": {"agent": "Master", "status": "failed", "error": "model timed out"}})
            assert failed and "model timed out" in failed

            route = _line(app, {"type": "delegation", "message": "routing plan generation",
                                "data": {"agent": "Xander", "status": "selected", "role": "planner",
                                         "reason": "architecture needs deeper reasoning"}})
            assert route and "planner" in route and "deeper reasoning" in route

            decision = _line(app, {"type": "plan", "message": "old summary",
                                   "data": {"decision": "reuse the existing Rust pipeline",
                                            "why": "Cargo.lock and generated glue already match",
                                            "actions": 2, "checks": 1}})
            assert decision and "reuse the existing Rust pipeline" in decision
            assert "why:" in decision and "Cargo.lock" in decision

            # A check emits twice; the first has no result and used to render
            # as "Proof <name> — <name>".
            start = _line(app, {"type": "test", "message": "Check the file exists", "data": {}})
            assert start is None

            passed = _line(app, {"type": "test", "message": "ok",
                                 "data": {"name": "Check the file exists",
                                          "result": {"status": "ok", "returncode": 0}}})
            assert passed and "Proof passed" in passed
            assert passed.count("Check the file exists") == 1  # not twice

            failed_check = _line(app, {"type": "test", "message": "failed",
                                       "data": {"name": "compile it",
                                                "result": {"status": "failed", "returncode": 2,
                                                           "stderr": "undefined reference to main"}}})
            assert failed_check and "exit 2" in failed_check
            # The reason it failed is the whole point of the line.
            assert "undefined reference to main" in failed_check

            # The loop rail already shows progress; the feed should not repeat it.
            assert _line(app, {"type": "guide", "message": "guide written",
                               "data": {"guide": {"progress": "0/5 complete", "current": "resolving"}}}) is None
            # Unless he is actually asking something.
            question = _line(app, {"type": "guide", "message": "guide written",
                                   "data": {"guide": {"progress": "0/5", "current": "x",
                                                      "questions": ["What counts as done?"]}}})
            assert question and "What counts as done?" in question

            # Real work still reports.
            changed = _line(app, {"type": "patch", "message": "wrote it",
                                  "data": {"result": {"status": "ok", "changed_paths": ["tictactoe.cpp"]}}})
            assert changed and "tictactoe.cpp" in changed

            completed = _line(app, {"type": "action", "message": "ran cargo metadata",
                                    "data": {"action": {"expected": "inspect Rust metadata"},
                                             "result": {"status": "ok"}}})
            assert completed and "Step done" in completed

    asyncio.run(scenario())
