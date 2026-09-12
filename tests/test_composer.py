"""`?` shows the shortcuts card; `/` completes commands, ranked by use."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Input, Static

from xander_agent.composer import CommandCompleter, shortcuts_card
from xander_agent.intents import COMMANDS
from xander_agent.tui import Composer, XanderApp


def test_the_card_lists_every_key_mode_and_registered_command() -> None:
    card = "\n".join(shortcuts_card())
    assert "ctrl+q" in card and "shift+tab" in card and "↑ ↓" in card
    for spec in COMMANDS:
        assert spec.usage in card, spec.usage
    assert "ask" in card and "yolo" in card
    assert "(also /how, /commands)" in card


def test_suggestions_match_prefix_and_aliases_and_rank_by_use(tmp_path: Path) -> None:
    store = tmp_path / "usage.json"
    completer = CommandCompleter(store)

    names = [item.text for item in completer.suggest("/h")]
    assert names[:2] == ["/help", "/history"], "registry order before any use"
    assert completer.suggest("/ho")[0].text == "/how", "an alias is offered as typed when only it matches"
    assert completer.suggest("/ho")[0].spec.name == "/help"
    assert completer.suggest("/miss")[0].text == "/missions", "an alias that is the only match is offered as typed"

    for _ in range(3):
        completer.record("/history")
    assert [item.text for item in completer.suggest("/h")][0] == "/history", "the used one comes first"

    reloaded = CommandCompleter(store)
    assert reloaded.counts == {"/history": 3}, "use survives a restart"
    reloaded.record("/nonsense")
    assert "/nonsense" not in reloaded.counts, "refused commands are not learned"
    assert completer.suggest("/history show") == [], "arguments end completion"
    assert completer.suggest("plain text") == []


def test_complete_and_hint() -> None:
    completer = CommandCompleter()
    assert completer.complete("/hi") == "/history"
    assert completer.complete("/c") == "/cd ", "a command that takes an argument gets a space"
    assert completer.complete("/zzz") == "/zzz"
    assert completer.hint("/hi").startswith("tab → /history")
    assert "/cd <path>" in completer.hint("/cd some")
    assert "not a command" in completer.hint("/zzz x")
    assert "no command starts with" in completer.hint("/zzz")
    assert completer.hint("hello") == ""


def _feed_text(app: XanderApp) -> str:
    feed = app.query_one("#feed", VerticalScroll)
    return "\n".join(str(child.render()) for child in feed.children if isinstance(child, Static))


def test_question_mark_opens_the_card_and_slash_shows_live_hints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))

    async def scenario() -> None:
        app = XanderApp(workspace=tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            composer = app.query_one("#composer", Composer)
            composer.focus()
            await pilot.press("question_mark")
            await pilot.pause()
            text = _feed_text(app)
            assert "Keys" in text and "/new-project" in text and "ctrl+q" in text
            assert composer.value == "", "the ? is consumed, not sent"
            assert "you › ?" not in text

            await pilot.press("slash", "h")
            await pilot.pause()
            hint = str(app.query_one("#hint", Static).render())
            assert "tab → /help" in hint and "/history" in hint
            await pilot.press("tab")
            await pilot.pause()
            assert composer.value == "/help", "tab completes the top suggestion"
            assert composer.has_focus, "tab did not move focus away"

            composer.value = ""
            await pilot.pause()
            hint = str(app.query_one("#hint", Static).render())
            assert "? keys" in hint and "/ commands" in hint, "the default hint returns"

            composer.value = "/history"
            app.on_input_submitted(Input.Submitted(composer, "/history"))
            await pilot.pause()
            assert composer.completer.counts.get("/history") == 1
            assert (tmp_path / "state" / "composer-usage.json").exists()

    asyncio.run(scenario())
