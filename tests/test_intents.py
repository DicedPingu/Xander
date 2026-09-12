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
        "hello Xander",
        "I wonder what this repository is for",
        "This looks unusual, right?",
    ],
)
def test_questions_are_advice(line: str) -> None:
    assert parse_intent(line).kind == "advice"


def test_session_commands_still_route() -> None:
    assert parse_intent("/help").kind == "help"
    assert parse_intent("/mode plan").kind == "mode"
    assert parse_intent("cd ~/SPQR").kind == "chdir"
    assert parse_intent("/cd ~/SPQR").kind == "chdir"
    assert parse_intent("/cd ~/SPQR").argument == "~/SPQR"


@pytest.mark.parametrize(
    ("line", "kind", "argument"),
    [
        ("/research https://docs.python.org/3/library/importlib.html", "research", "https://docs.python.org/3/library/importlib.html"),
        ("/research durable execution patterns", "research", "durable execution patterns"),
        ("/goal ship a verified APK", "goal", "ship a verified APK"),
        ("/goal", "goal", ""),
        ("/addtodo write the README", "todo", "write the README"),
        ("/discuss the sign-in flow", "discuss", "the sign-in flow"),
        ("/discuss", "discuss", ""),
        ("/work build the parser", "work", "build the parser"),
        ("/work", "work", ""),
        ("/talk hello there", "talk", "hello there"),
        ("/ask hello there", "talk", "hello there"),
        ("/HELP", "help", ""),
        ("/how", "help", ""),
        ("/commands", "help", ""),
    ],
)
def test_named_commands_resolve_with_their_argument(line: str, kind: str, argument: str) -> None:
    intent = parse_intent(line)
    assert intent.kind == kind
    assert intent.argument == argument


def test_only_work_carries_authorization() -> None:
    assert parse_intent("/work build the parser").authorized is True
    assert parse_intent("/discuss build the parser").authorized is False
    assert parse_intent("/goal build the parser").authorized is False
    assert parse_intent("/research build the parser").authorized is False


def test_interface_commands_resolve_as_commands() -> None:
    intent = parse_intent("/set mode plan")
    assert intent.kind == "command"
    assert intent.command == "/set"
    assert intent.argument == "mode plan"
    assert parse_intent("/proof").command == "/show"
    assert parse_intent("/todo write the README").kind == "command"


@pytest.mark.parametrize(
    "line",
    [
        "/rm -rf /",
        "/bin/sh -c 'echo hi'",
        "/usr/bin/python3 -c 'print(1)'",
        "/reserch python",
        "/",
        "/unknownthing",
    ],
)
def test_unknown_slash_lines_are_explained_never_run(line: str) -> None:
    intent = parse_intent(line)
    assert intent.kind == "unknown_command"
    assert intent.authorized is False
    assert "not run as shell" in intent.reason
    assert "/help" in intent.reason


def test_unknown_command_suggests_the_nearest_known_name() -> None:
    assert "/research" in parse_intent("/reserch python").reason
    assert "/discuss" in parse_intent("/discus the plan").reason


def test_cd_without_a_path_is_explained() -> None:
    intent = parse_intent("/cd")
    assert intent.kind == "unknown_command"
    assert "/cd <path>" in intent.reason


@pytest.mark.parametrize(
    "line",
    [
        "This project is going to be about a Codewars client for Android",
        "this project is about a chess engine",
        "The app will be a small habit tracker",
        "I'm thinking of adding a plugin system",
        "I've been thinking about how the retry loop should work",
        "I'd like to discuss the architecture before we start",
        "Let's think about the sign-in flow",
        "let's brainstorm names for the crawler clone",
        "The idea is a bounded web crawler with provenance",
        "idea: keep goals in a portable .Xander record",
        "What if we split the engine into layers",
        "Maybe we could store the draft on the workboard",
        "Imagine a version of this that runs offline",
    ],
)
def test_exploratory_framing_opens_a_discussion(line: str) -> None:
    intent = parse_intent(line)
    assert intent.kind == "discuss"
    assert intent.authorized is False
    assert intent.argument == line.strip()


@pytest.mark.parametrize(
    "line",
    [
        "Create a README for the crawler",
        "Do the migration to pydantic 2",
        "Make the composer keep the draft",
        "Build the Android client",
        "Implement /research in the TUI",
        "please fix the flaky test",
        "Xander, add a changelog entry",
        "ok, go ahead and write the parser",
        "run the test suite and report",
        "set up pre-commit",
        "Do it",
        "proceed",
    ],
)
def test_imperative_openings_authorize_work(line: str) -> None:
    intent = parse_intent(line)
    assert intent.kind == "order"
    assert intent.authorized is True


@pytest.mark.parametrize(
    "line",
    [
        "a tic tac toe website under 30 KB",
        "the retry loop",
        "Codewars Android app",
        "second order",
        "tests are red again",
    ],
)
def test_unclear_lines_are_kept_as_drafts_not_run(line: str) -> None:
    intent = parse_intent(line)
    assert intent.kind == "draft"
    assert intent.authorized is False
    assert intent.argument == line.strip()
    assert intent.reason


def test_empty_line_is_an_empty_draft() -> None:
    assert parse_intent("   ").kind == "draft"
    assert parse_intent("   ").argument == ""


def test_scene_setting_questions_are_still_discussions() -> None:
    # Both doors lead to conversation; discussion also keeps the line as the draft.
    assert parse_intent("What if we split the engine — would that help?").kind == "discuss"
    assert parse_intent("Should I use flexbox or grid here?").kind == "advice"


def test_scene_setting_with_a_clear_artifact_order_is_work() -> None:
    assert parse_intent("The idea is simple: create a test file for the parser module").kind == "order"


def test_command_help_lists_the_core_commands_first() -> None:
    from xander_agent.intents import command_help

    lines = command_help("core")
    usages = [line.partition(" — ")[0] for line in lines]
    assert usages[:3] == ["/help", "/mode [ask|plan|build|yolo]", "/cd <path>"]
    for expected in ("/discuss [topic]", "/work [goal]", "/research <URL or topic>", "/goal [goal]", "/addtodo <task>"):
        assert expected in usages
    assert all(" — " in line for line in command_help())


@pytest.mark.parametrize(
    "line",
    [
        "take yourself some time to research your soul and improve yourself",
        "upgrade yourself with better planning",
        "make yourself more capable",
        "fix Xander's retry loop",
    ],
)
def test_explicit_self_work_is_routed_to_xander(line: str) -> None:
    assert parse_intent(line).kind == "selfwork"
