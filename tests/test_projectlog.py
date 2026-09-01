from pathlib import Path

from xander_agent.engine import Engine
from xander_agent.models import XanderRequest
from xander_agent.projectlog import append_project_event, project_log_path
from xander_agent.tasks import TaskStore


def test_project_activity_has_one_human_log_inside_the_project(tmp_path: Path, monkeypatch) -> None:
    agent = tmp_path / "ASKAR" / "Xander"
    project = agent / "projects" / "demo"
    project.mkdir(parents=True)
    monkeypatch.setenv("XANDER_AGENT_DIR", str(agent))

    first = append_project_event(project, "task-1", "[PLAN] choose the smallest safe change")
    second = append_project_event(project, "task-1", "[TEST] checks passed")

    assert first == project / "logs" / "task-1.md"
    assert second == first
    content = first.read_text(encoding="utf-8")
    assert content.startswith("# Xander task task-1")
    assert "choose the smallest safe change" in content
    assert "checks passed" in content

    external = tmp_path / "opened-system-settings"
    external.mkdir()
    external_log = project_log_path(external, "task-2")
    assert external_log is not None
    assert external_log.is_relative_to(agent / "projects" / "external")
    append_project_event(external, "task-2", "[PLAN] inspect the explicitly opened settings workspace")
    assert external_log.is_file()
    assert external_log.parent.parent.joinpath("PROJECT.md").is_file()
    assert not (external / "logs").exists()


def test_engine_event_writes_the_project_log_without_a_tui(tmp_path: Path, monkeypatch) -> None:
    agent = tmp_path / "ASKAR" / "Xander"
    project = agent / "projects" / "cli-project"
    project.mkdir(parents=True)
    monkeypatch.setenv("XANDER_AGENT_DIR", str(agent))
    store = TaskStore(root=tmp_path / "state" / "tasks")
    task = store.create(XanderRequest(mode="inspect", workspace=project, goal="record this run"))
    engine = Engine.__new__(Engine)
    engine.workspace = project
    engine.task_store = store
    engine.event_sink = None
    engine._sequence = 0

    engine._emit(task, "delegation", "Monica returned a reference brief", {"agent": "Monica"})

    log = project / "logs" / f"{task.id}.md"
    assert log.is_file()
    assert "Monica returned a reference brief" in log.read_text(encoding="utf-8")
