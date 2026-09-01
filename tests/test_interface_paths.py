from pathlib import Path

from xander_agent import paths


def test_xdg_paths_are_lazy_and_private(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    assert paths.config_dir() == tmp_path / "config" / "xander"
    assert paths.tasks_dir() == tmp_path / "state" / "xander" / "tasks"
    assert paths.variants_dir() == tmp_path / "config" / "xander" / "variants"
    assert paths.migration_dir() == tmp_path / "state" / "xander" / "migrations"
    assert not paths.config_dir().exists()

    created = paths.ensure_runtime_dirs()
    assert all(path.is_dir() for path in created)
    assert paths.cache_dir() in created


def test_explicit_agent_root_keeps_runtime_inside_askar(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "ASKAR" / "Xander"
    monkeypatch.setenv("XANDER_AGENT_DIR", str(root))
    monkeypatch.delenv("XANDER_CONFIG_DIR", raising=False)
    monkeypatch.delenv("XANDER_STATE_DIR", raising=False)
    monkeypatch.delenv("XANDER_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    assert paths.agent_dir() == root
    assert paths.config_dir() == root / "config"
    assert paths.state_dir() == root / "state"
    assert paths.logs_dir() == root / "logs"
    assert paths.projects_dir() == root / "projects"
    assert paths.shared_dir() == root.parent / "shared"


def test_legacy_task_records_are_read_but_new_saves_use_current_root(tmp_path: Path, monkeypatch) -> None:
    from xander_agent.models import TaskRecord, TaskStatus, XanderRequest
    from xander_agent.tasks import TaskStore

    current = tmp_path / "askar" / "Xander" / "state" / "tasks"
    legacy = tmp_path / "home" / ".local" / "state" / "xander" / "tasks"
    current.mkdir(parents=True)
    legacy.mkdir(parents=True)
    monkeypatch.setenv("XANDER_STATE_DIR", str(current.parent))
    monkeypatch.setattr(paths, "legacy_tasks_dir", lambda: legacy)
    request = XanderRequest(mode="inspect", workspace=tmp_path, goal="recover the old plan")
    task = TaskRecord(id="legacy-task", request=request)
    (legacy / "legacy-task.json").write_text(task.model_dump_json(), encoding="utf-8")

    store = TaskStore()
    assert store.load("legacy-task").request.goal == "recover the old plan"
    task.status = TaskStatus.COMPLETED
    destination = store.save(task)
    assert destination.parent == current
    assert '"completed"' not in (legacy / "legacy-task.json").read_text(encoding="utf-8")
