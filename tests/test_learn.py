from pathlib import Path

import pytest

from xander_agent.learn import LearnStore, classify, constraints_for
from xander_agent.learnrun import goal_for, run_learning
from xander_agent.steering import SteeringInbox, apply, drain, interpret, parse_fast


@pytest.fixture
def store(tmp_path: Path) -> LearnStore:
    return LearnStore(root=tmp_path / "learning")


def test_targets_are_classified_offline_without_a_model() -> None:
    assert classify("https://example.com/guide")[:2] == ("url", "https://example.com/guide")
    assert classify("github.com/textualize/textual")[0] == "repo"
    assert classify("pypi:httpx") == ("package", "httpx", "pypi")
    assert classify("textual")[0] == "package"
    assert classify("skill:tdd-workflow")[:2] == ("skill", "tdd-workflow")
    assert classify("category:testing")[:2] == ("category", "testing")
    assert classify("how attention works")[0] == "topic"
    # An explicit prefix beats the shape guess.
    assert classify("topic:textual")[:2] == ("topic", "textual")


def test_operator_work_outranks_what_xander_discovered(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    store.add(queue, "found-later", origin="discovered")
    store.add(queue, "asked-for", origin="operator")
    assert queue.next_target().subject == "asked-for"

    # Re-adding a discovered target as an operator ask promotes it in place.
    store.add(queue, "found-later", origin="operator")
    promoted = next(item for item in queue.targets if item.subject == "found-later")
    assert promoted.origin == "operator" and promoted.priority <= 50


def test_queue_survives_a_reload(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    target = store.add(queue, "pypi:httpx")
    store.set_mission(queue, "learn the http stack")
    store.restrict(queue, "never install anything")

    reloaded = LearnStore(root=store.root).load(tmp_path)
    assert reloaded.mission == "learn the http stack"
    assert reloaded.restrictions == ["never install anything"]
    assert [item.id for item in reloaded.targets] == [target.id]


def test_fast_commands_need_no_model() -> None:
    assert parse_fast("!focus textual").kind == "focus"
    assert parse_fast("!drop react").kind == "drop"
    assert parse_fast("!never touch the network").kind == "restrict"
    # The prohibition must survive; storing the bare tail would invert it.
    assert parse_fast("!never install system packages").argument == "never install system packages"
    assert parse_fast("!dont use the network").argument == "never use the network"
    assert parse_fast("!stop").kind == "stop"
    assert parse_fast("!set depth 0").argument == "depth 0"
    # Plain prose is not a fast command, and a bare verb with no argument is not either.
    assert parse_fast("please look at httpx next") is None
    assert parse_fast("!focus") is None


def test_prose_falls_back_to_a_note_when_no_model_is_available() -> None:
    directive = interpret("have a look at the caching docs", backend=None)
    assert directive.kind == "note"
    assert "caching" in directive.argument


def test_steering_reprioritises_reshapes_and_restricts(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    store.add(queue, "httpx")
    store.add(queue, "textual")

    assert "brought forward" in apply(parse_fast("!focus textual"), queue, store)
    assert queue.next_target().subject == "textual"
    assert [item.subject for item in queue.pending()] == ["textual", "httpx"]

    assert "dropped" in apply(parse_fast("!drop httpx"), queue, store)
    assert next(i for i in queue.targets if i.subject == "httpx").state == "skipped"

    apply(parse_fast("!add pypi:rich, category:testing"), queue, store)
    assert {"rich", "testing"} <= {item.subject for item in queue.targets}

    apply(parse_fast("!never install system packages"), queue, store)
    assert queue.restrictions == ["never install system packages"]
    # Restrictions must reach the prompt every role reads.
    assert any("never install system packages" in line for line in constraints_for(queue))


def test_settings_change_live_and_reject_nonsense(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    assert "depth = 0" in apply(parse_fast("!set depth 0"), queue, store)
    assert queue.settings.depth == 0
    assert "allow_network = False" in apply(parse_fast("!set network off"), queue, store)
    assert queue.settings.allow_network is False
    assert "whole number" in apply(parse_fast("!set depth banana"), queue, store)
    assert queue.settings.depth == 0
    assert "unknown setting" in apply(parse_fast("!set wobble 3"), queue, store)


def test_posting_is_non_blocking_and_drain_applies_at_a_safe_point(
    tmp_path: Path, store: LearnStore
) -> None:
    queue = store.load(tmp_path)
    store.add(queue, "httpx")
    store.add(queue, "textual")

    inbox = SteeringInbox(tmp_path, root=tmp_path / "steering")
    inbox.post("!focus textual")
    inbox.post("!never use the network")

    # Nothing has changed yet: posting only records.
    assert queue.next_target().subject == "httpx"
    assert queue.restrictions == []

    handled = drain(tmp_path, queue, store, backend=None, inbox=inbox)
    assert len(handled) == 2
    assert queue.next_target().subject == "textual"
    assert queue.restrictions == ["never use the network"]

    # Applied notes are not replayed, and the history is kept for the operator.
    assert drain(tmp_path, queue, store, backend=None, inbox=inbox) == []
    assert len(inbox.history()) == 2
    assert all(note.applied_at for note in inbox.history())


def test_goals_are_concrete_per_kind(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    assert "httpx" in goal_for(store.add(queue, "pypi:httpx"))
    assert "repository" in goal_for(store.add(queue, "github.com/a/b"))
    assert "when NOT to load it" in goal_for(store.add(queue, "skill:tdd-workflow"))


def test_run_works_the_queue_and_listens_between_targets(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    store.add(queue, "first")
    store.add(queue, "second")
    store.add(queue, "third")

    inbox = SteeringInbox(tmp_path, root=tmp_path / "steering")
    seen: list[str] = []
    events: list[dict] = []

    def fake_invoke(mode: str, **kwargs: object) -> dict:
        goal = str(kwargs.get("goal", ""))
        seen.append(goal)
        # The operator types while the first target is running.
        if len(seen) == 1:
            inbox.post("!drop second")
        return {"ok": True, "status": "completed", "task_id": f"t{len(seen)}",
                "task": {"lesson": "a verified lesson"}, "handoff": {}}

    summary = run_learning(
        tmp_path,
        store=store,
        invoke=fake_invoke,
        event_sink=events.append,
        backend=None,
        max_targets=5,
        inbox=inbox,
    )

    # "second" was dropped mid-run, so only first and third were worked.
    worked = [row["target"] for row in summary["worked"]]
    assert worked == ["first", "third"]
    assert summary["stopped_because"] == "queue is empty"
    assert any(event["type"] == "steering" for event in events)
    assert any(event["type"] == "learn" for event in events)


def test_a_failing_target_is_blocked_not_silently_passed(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    store.add(queue, "only")

    def failing_invoke(mode: str, **kwargs: object) -> dict:
        return {"ok": False, "status": "failed", "task_id": "t1",
                "task": {"failure": "no usable source"}, "handoff": {}}

    summary = run_learning(tmp_path, store=store, invoke=failing_invoke, backend=None)
    assert summary["worked"][0]["ok"] is False
    reloaded = LearnStore(root=store.root).load(tmp_path)
    target = reloaded.targets[0]
    assert target.state == "blocked" and "no usable source" in target.evidence


def test_an_exploding_engine_blocks_the_target_and_keeps_going(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    store.add(queue, "explodes")
    store.add(queue, "survives")
    calls: list[str] = []

    def invoke(mode: str, **kwargs: object) -> dict:
        calls.append(str(kwargs.get("goal", "")))
        if len(calls) == 1:
            raise RuntimeError("backend fell over")
        return {"ok": True, "status": "completed", "task_id": "t2", "task": {}, "handoff": {}}

    summary = run_learning(tmp_path, store=store, invoke=invoke, backend=None)
    assert len(calls) == 2
    assert [row["target"] for row in summary["worked"]] == ["survives"]
    reloaded = LearnStore(root=store.root).load(tmp_path)
    assert next(i for i in reloaded.targets if i.subject == "explodes").state == "blocked"


def test_follow_ups_are_capped_by_depth_and_never_invented(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    queue.settings.depth = 2
    store.add(queue, "root")

    def invoke(mode: str, **kwargs: object) -> dict:
        return {
            "ok": True, "status": "completed", "task_id": "t1", "task": {},
            "handoff": {"research_sources": ["https://a.example", "https://b.example", "https://c.example"]},
        }

    run_learning(tmp_path, store=store, invoke=invoke, backend=None, max_targets=1)
    reloaded = LearnStore(root=store.root).load(tmp_path)
    discovered = [item for item in reloaded.targets if item.origin == "discovered"]
    assert len(discovered) == 2
    assert all(item.priority > 50 for item in discovered)


def test_depth_zero_means_learn_exactly_what_i_gave_you(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    queue.settings.depth = 0
    store.add(queue, "root")

    def invoke(mode: str, **kwargs: object) -> dict:
        return {"ok": True, "status": "completed", "task_id": "t1", "task": {},
                "handoff": {"research_sources": ["https://a.example"]}}

    run_learning(tmp_path, store=store, invoke=invoke, backend=None, max_targets=1)
    reloaded = LearnStore(root=store.root).load(tmp_path)
    assert [item.origin for item in reloaded.targets] == ["operator"]


def test_restrictions_and_mission_reach_every_role(tmp_path: Path, store: LearnStore) -> None:
    queue = store.load(tmp_path)
    store.set_mission(queue, "understand the http stack")
    store.restrict(queue, "never write to the workspace")
    lines = constraints_for(queue)
    assert any("understand the http stack" in line for line in lines)
    assert any("never write to the workspace" in line for line in lines)

    queue.settings.allow_network = False
    assert any("Network access is off" in line for line in constraints_for(queue))
