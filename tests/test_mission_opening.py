from pathlib import Path

from xander_agent.models import ActionResult, ActionStatus, TaskStatus, XanderRequest
from xander_agent.mission import MissionStore
from xander_agent.opening import OPENING_STYLES, render_opening
from xander_agent.power import PowerStatus, PowerZeroGuard, read_power_status
from xander_agent.tasks import TaskStore


def test_mission_is_workspace_scoped_and_has_readable_timeline(tmp_path: Path) -> None:
    own = tmp_path / "own"
    other = tmp_path / "other"
    own.mkdir()
    other.mkdir()
    store = TaskStore(root=tmp_path / "tasks")
    task = store.create(XanderRequest(mode="implement", workspace=own, goal="make a home screen"))
    task.status = TaskStatus.COMPLETED
    task.evidence.append({"kind": "milestone", "message": "opening menu landed"})
    task.results.append(
        ActionResult(
            action_id="a1",
            status=ActionStatus.OK,
            changed_paths=["xander_agent/tui.py"],
            reason="updated the opening surface",
        )
    )
    store.save(task)

    missions = MissionStore(store).list(own)
    assert [mission.id for mission in missions] == [task.id]
    assert "verified evidence" in missions[0].result
    assert [item["kind"] for item in missions[0].timeline()] == ["milestone", "change", "result"]
    try:
        MissionStore(store).load(other, task.id)
    except ValueError as exc:
        assert "different workspace" in str(exc)
    else:
        raise AssertionError("foreign workspace could read a Mission")


def test_mission_delete_is_explicit_and_workspace_scoped(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = TaskStore(root=tmp_path / "tasks")
    task = store.create(XanderRequest(mode="inspect", workspace=workspace, goal="inspect"))

    MissionStore(store).delete(workspace, task.id)
    assert not store.path(task.id).exists()


def test_opening_styles_are_structurally_different() -> None:
    rendered = [
        render_opening(
            style,
            workspace="/tmp/project",
            branch="main",
            dirty_count=2,
            power="88% · charging",
            mission="make the home screen understandable",
            mission_state="idle",
            history_count=3,
            desktop="ready",
        )
        for style in OPENING_STYLES
    ]
    assert len(set(rendered)) == 5
    assert any("MISSION DESK" in item for item in rendered)
    assert any("COMPASS" in item for item in rendered)
    assert any("CHRONICLE" in item for item in rendered)
    assert any("WORKSHOP" in item for item in rendered)
    assert any("XANDER / RESIDENT" in item for item in rendered)


def test_power_zero_trips_once_and_unknown_is_not_zero(tmp_path: Path) -> None:
    battery = tmp_path / "BAT0"
    battery.mkdir()
    (battery / "capacity").write_text("0\n", encoding="utf-8")
    (battery / "status").write_text("Discharging\n", encoding="utf-8")
    assert read_power_status(tmp_path).capacity == 0

    calls: list[str] = []
    guard = PowerZeroGuard(
        reader=lambda: PowerStatus(0, "discharging", "test"),
        shutdown=lambda: calls.append("poweroff"),
    )
    guard.poll()
    guard.poll()
    assert calls == ["poweroff"]
    assert guard.tripped
    assert PowerStatus(None).label == "unknown"
