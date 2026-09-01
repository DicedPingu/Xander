"""Xander's private ASKAR-local runtime locations."""

from __future__ import annotations

import os
from pathlib import Path


def _home() -> Path:
    return Path(os.environ.get("HOME", "~")).expanduser()


def _xdg(env_name: str, fallback: Path) -> Path:
    configured = os.environ.get(env_name)
    return Path(configured).expanduser() if configured else fallback


def agent_dir() -> Path:
    override = os.environ.get("XANDER_AGENT_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[1]


def shared_dir() -> Path:
    return agent_dir().parent / "shared"


def shared_skills_dir() -> Path:
    return shared_dir() / "skills"


def shared_guides_dir() -> Path:
    return shared_dir() / "guides"


def logs_dir() -> Path:
    override = os.environ.get("XANDER_LOG_DIR")
    if override:
        return Path(override).expanduser()
    return agent_dir() / "logs"


def projects_dir() -> Path:
    return agent_dir() / "projects"


def config_dir() -> Path:
    """Return Xander's user configuration directory."""

    override = os.environ.get("XANDER_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    configured = os.environ.get("XDG_CONFIG_HOME")
    return Path(configured).expanduser() / "xander" if configured else agent_dir() / "config"


def state_dir() -> Path:
    """Return Xander's durable task and migration state directory."""

    override = os.environ.get("XANDER_STATE_DIR")
    if override:
        return Path(override).expanduser()
    configured = os.environ.get("XDG_STATE_HOME")
    return Path(configured).expanduser() / "xander" if configured else agent_dir() / "state"


def cache_dir() -> Path:
    """Return Xander's replaceable indexes and research cache directory."""

    override = os.environ.get("XANDER_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    configured = os.environ.get("XDG_CACHE_HOME")
    return Path(configured).expanduser() / "xander" if configured else agent_dir() / "cache"


def tasks_dir() -> Path:
    return state_dir() / "tasks"


def variants_dir() -> Path:
    return config_dir() / "variants"


def migration_dir() -> Path:
    return state_dir() / "migrations"


def legacy_state_dir() -> Path | None:
    if any(
        os.environ.get(name)
        for name in ("XANDER_AGENT_DIR", "XANDER_STATE_DIR", "XDG_STATE_HOME")
    ):
        return None
    return _home() / ".local" / "state" / "xander"


def legacy_tasks_dir() -> Path | None:
    root = legacy_state_dir()
    return root / "tasks" if root else None


def legacy_memory_dir() -> Path | None:
    root = legacy_state_dir()
    return root / "memory" if root else None


def legacy_workboards_dir() -> Path | None:
    root = legacy_state_dir()
    return root / "workboards" if root else None


def ensure_runtime_dirs() -> tuple[Path, ...]:
    """Create private runtime directories and return them in stable order."""

    directories = (
        config_dir(),
        state_dir(),
        cache_dir(),
        logs_dir(),
        projects_dir(),
        tasks_dir(),
        variants_dir(),
        migration_dir(),
    )
    for directory in directories:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directories
