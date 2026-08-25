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
