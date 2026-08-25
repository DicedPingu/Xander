"""Intent routing: operator lines pick the right door, feedback included."""

from __future__ import annotations

import pytest

from xander_agent.intents import parse_intent


@pytest.mark.parametrize(
    "line",
    [
        "I don't like walls of comments in generated code",
        "i really hate emoji in commit messages",
        "I like short summaries at the end",
        "I prefer uv over pip",
        "From now on keep patches minimal",
        "always run the tests before claiming success",
        "never touch files outside the workspace",
        "stop using semicolons in python examples",
        "please never add TODO comments",
        "less of the consultant filler",
    ],
)
def test_feedback_lines_are_feedback(line: str) -> None:
    intent = parse_intent(line)
    assert intent.kind == "feedback"
    assert intent.argument == line.strip()


@pytest.mark.parametrize(
    "line",
    [
        # A feedback-looking lead that clearly orders artifact work stays an order.
        "always create a test file for every new module in tests/",
        "never mind, just fix the failing test in tests/test_engine_replan.py",
        "create a website with the game of tic tac toe that is less than 30 KB",
        "make the background green",
    ],
)
def test_orders_stay_orders(line: str) -> None:
    assert parse_intent(line).kind == "order"


@pytest.mark.parametrize(
    "line",
    [
        "what could be offsetting the wasm bundle size?",
        "should I use flexbox or grid here",
        "explain how the executor snapshots the workspace",
    ],
)
def test_questions_are_advice(line: str) -> None:
    assert parse_intent(line).kind == "advice"


def test_session_commands_still_route() -> None:
    assert parse_intent("/help").kind == "help"
    assert parse_intent("/mode plan").kind == "mode"
    assert parse_intent("cd ~/SPQR").kind == "chdir"
