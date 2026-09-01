"""Explicit, local desktop observation with a graceful Linux fallback."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .paths import cache_dir


@dataclass(frozen=True)
class DesktopCapability:
    available: bool
    command: str = ""
    note: str = ""


def desktop_capability() -> DesktopCapability:
    for command in ("gnome-screenshot", "grim", "import", "scrot"):
        if shutil.which(command):
            return DesktopCapability(True, command, "capture is explicit and stays local")
    return DesktopCapability(False, note="install a screenshot backend for desktop observation")


def capture_desktop(destination: Path | None = None) -> Path:
    capability = desktop_capability()
    if not capability.available:
        raise RuntimeError(capability.note)
    output = destination or cache_dir() / "screenshots" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".png"
    )
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if capability.command == "grim":
        argv = [capability.command, str(output)]
    elif capability.command == "import":
        argv = [capability.command, "-window", "root", str(output)]
    elif capability.command == "scrot":
        argv = [capability.command, str(output)]
    else:
        argv = [capability.command, "-f", str(output)]
    subprocess.run(argv, check=True, timeout=15)
    return output
