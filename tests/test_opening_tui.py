import asyncio
from pathlib import Path

from textual.widgets import ContentSwitcher, Input, Select, Static

from xander_agent.models import MissionGuide, TaskStatus, XanderRequest
from xander_agent.power import PowerStatus
from xander_agent.tasks import TaskStore
from xander_agent.tui import XanderApp


def test_tui_opens_on_the_composer_and_switches_useful_surfaces(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = XanderApp(
            workspace=tmp_path,
            variant="ambusher",
            autonomy="supervised",
            power_reader=lambda: PowerStatus(87, "charging", "test"),
        )
        async with app.run_test(size=(120, 42)) as pilot:
            await pilot.pause()
            opening = str(app.query_one("#mission-banner", Static).render())
            assert "Describe the result below" in opening
            assert app.query_one("#goal-input", Input).has_focus
            app.action_history()
            assert app.query_one(ContentSwitcher).current == "history-view"
            app.action_soul()
            assert app.query_one(ContentSwitcher).current == "system-view"
            app.action_tests()
            assert app.query_one(ContentSwitcher).current == "evidence-view"
            assert app.query_one("#mission-mode", Select).value == "implement"
            assert app.query_one("#mission-autonomy", Select).value == "supervised"

    asyncio.run(scenario())


def test_tui_power_zero_cancels_and_requests_shutdown_once(tmp_path: Path) -> None:
    shutdowns: list[str] = []

    async def scenario() -> None:
        app = XanderApp(
            workspace=tmp_path,
            power_reader=lambda: PowerStatus(0, "discharging", "test"),
            power_shutdown=lambda: shutdowns.append("poweroff"),
        )
        async with app.run_test(size=(120, 42)) as pilot:
            await pilot.pause()
            app._poll_power()
            app._poll_power()
            assert app.task_state == "power-zero"

    asyncio.run(scenario())
    assert shutdowns == ["poweroff"]


def test_mission_library_can_continue_and_delete_a_finished_mission(tmp_path: Path, monkeypatch) -> None:
    store = TaskStore(root=tmp_path / "tasks")
    task = store.create(XanderRequest(mode="implement", workspace=tmp_path, goal="improve the finished result"))
    task.status = TaskStatus.COMPLETED
    task.guide = MissionGuide(
        statement="Deliver a verified result for: improve the finished result",
        current="Mission result recorded.",
        progress="5/5 complete",
        result="goal verified",
    )
    store.save(task)
    monkeypatch.setattr("xander_agent.tasks.TaskStore", lambda: store)

    async def scenario() -> None:
        app = XanderApp(
            workspace=tmp_path,
            power_reader=lambda: PowerStatus(87, "charging", "test"),
        )
        continued: list[tuple[str, list[str]]] = []
        async with app.run_test(size=(120, 42)) as pilot:
            await pilot.pause()
            app._run_resume = lambda mission_id, selected: continued.append((mission_id, selected))
            app.action_history()
            await pilot.pause()
            library = app.query_one("#mission-library", Select)
            assert library.value == task.id
            assert "improve the finished result" in str(app.query_one("#mission-library-detail", Static).render())
            app._continue_selected_mission()
            assert continued == [(task.id, [])]
            app.task_state = "idle"
            app._engine_busy = False
            app._delete_selected_mission()
            app._delete_selected_mission()

    asyncio.run(scenario())
    assert not store.path(task.id).exists()
