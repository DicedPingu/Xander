"""The scoreboard: deterministic aggregation from persisted task records."""

from __future__ import annotations

from pathlib import Path

from xander_agent.models import ActionResult, ActionStatus, TaskRecord, TaskStatus, XanderRequest
from xander_agent.stats import render_lines, sparkline, stats_payload
from xander_agent.tasks import TaskStore


def _seed(store: TaskStore, workspace: Path) -> None:
    request = XanderRequest(mode="implement", workspace=workspace, goal="paint it green")
    win = TaskRecord(id="99990101T000000Z-a-win", request=request)  # ids sort by timestamp
    win.status = TaskStatus.COMPLETED
    win.subject = "painting"
    win.attempt = 1
    win.results = [ActionResult(action_id="a1", status=ActionStatus.OK)]
    win.check_results = [ActionResult(action_id="c1", status=ActionStatus.OK)]
    win.evidence = [
        {"kind": "model", "role": "coder", "tokens": 300, "tokens_per_second": 40.0},
        {"kind": "master_verdict", "happy": True},
        {"kind": "lurker_brief", "chars": 500},
    ]
    store.save(win)

    request = XanderRequest(mode="implement", workspace=workspace, goal="shrink the bundle")
    loss = TaskRecord(id="99990101T000001Z-b-loss", request=request)  # newest record: the failed one
    loss.status = TaskStatus.FAILED
    loss.attempt = 3
    loss.results = [ActionResult(action_id="a2", status=ActionStatus.FAILED)]
    loss.check_results = [ActionResult(action_id="c2", status=ActionStatus.FAILED)]
    loss.evidence = [{"kind": "model", "role": "planner", "tokens": 100, "tokens_per_second": 20.0}]
    store.save(loss)


def test_stats_payload_aggregates_wins_tokens_and_squad_activity(tmp_path: Path) -> None:
    store = TaskStore(root=tmp_path / "tasks")
    _seed(store, tmp_path)

    payload = stats_payload(store)

    assert payload["tasks"] == 2
    assert payload["by_status"] == {"completed": 1, "failed": 1}
    assert payload["success_rate"] == 0.5
    assert payload["average_attempts"] == 2.0
    assert payload["actions"] == {"ok": 1, "failed": 1}
    assert payload["checks"] == {"passed": 1, "total": 2}
    assert payload["model"] == {"calls": 2, "tokens": 400, "average_tokens_per_second": 30.0}
    assert payload["squad"] == {"master_verdicts": 1, "master_happy": 1, "lurker_briefs": 1}
    assert payload["wins_by_variant"] == {"default": 1}
    assert "painting" in payload["top_subjects"]
    assert sum(payload["last_days"]) == 2, "both tasks were created today"
    assert len(payload["sparkline"]) == 14


def test_streak_counts_consecutive_wins_from_the_newest_task(tmp_path: Path) -> None:
    store = TaskStore(root=tmp_path / "tasks")
    _seed(store, tmp_path)  # newest record is the failed one
    assert stats_payload(store)["current_streak"] == 0

    request = XanderRequest(mode="implement", workspace=tmp_path, goal="win again")
    newer_win = TaskRecord(id="99990102T000000Z-win", request=request)  # force newest
    newer_win.status = TaskStatus.COMPLETED
    store.save(newer_win)
    assert stats_payload(store)["current_streak"] == 1


def test_sparkline_scales_to_the_peak() -> None:
    assert sparkline([]) == ""
    assert sparkline([0, 0, 4]) == "▁▁█"
    assert len(sparkline([1] * 14)) == 14


def test_render_lines_stay_markup_safe_and_informative(tmp_path: Path) -> None:
    store = TaskStore(root=tmp_path / "tasks")
    _seed(store, tmp_path)
    text = "\n".join(render_lines(stats_payload(store)))
    assert "50% verified" in text
    assert "400 tokens" in text
    assert "Master happy 1/1" in text
    assert "Lurker briefs 1" in text


def test_empty_store_renders_without_dividing_by_zero(tmp_path: Path) -> None:
    payload = stats_payload(TaskStore(root=tmp_path / "tasks"))
    assert payload["tasks"] == 0
    assert payload["success_rate"] == 0.0
    assert render_lines(payload), "an empty scoreboard still renders"
