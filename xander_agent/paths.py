"""XDG-compliant runtime locations for Xander.

Path accessors deliberately do not create directories. Call
``ensure_runtime_dirs`` at an application boundary that needs writable state.
"""

from __future__ import annotations

import os
from pathlib import Path


def _home() -> Path:
    return Path(os.environ.get("HOME", "~")).expanduser()


def _xdg(env_name: str, fallback: Path) -> Path:
    configured = os.environ.get(env_name)
    return Path(configured).expanduser() if configured else fallback


def config_dir() -> Path:
    """Return Xander's user configuration directory."""

    override = os.environ.get("XANDER_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return _xdg("XDG_CONFIG_HOME", _home() / ".config") / "xander"


def state_dir() -> Path:
    """Return Xander's durable task and migration state directory."""

    override = os.environ.get("XANDER_STATE_DIR")
    if override:
        return Path(override).expanduser()
    return _xdg("XDG_STATE_HOME", _home() / ".local" / "state") / "xander"


def cache_dir() -> Path:
    """Return Xander's replaceable indexes and research cache directory."""

    override = os.environ.get("XANDER_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return _xdg("XDG_CACHE_HOME", _home() / ".cache") / "xander"


def tasks_dir() -> Path:
    return state_dir() / "tasks"


def variants_dir() -> Path:
    return config_dir() / "variants"


def migration_dir() -> Path:
    return state_dir() / "migrations"


def ensure_runtime_dirs() -> tuple[Path, ...]:
    """Create private runtime directories and return them in stable order."""

    directories = (
        config_dir(),
        state_dir(),
        cache_dir(),
        tasks_dir(),
        variants_dir(),
        migration_dir(),
    )
    for directory in directories:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directories
