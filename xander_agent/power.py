"""Battery observation and the one-way power-zero safety stop."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class PowerStatus:
    capacity: int | None
    state: str = "unknown"
    source: str = "unavailable"

    @property
    def label(self) -> str:
        if self.capacity is None:
            return "unknown"
        return f"{self.capacity}% · {self.state}"


def read_power_status(root: Path = Path("/sys/class/power_supply")) -> PowerStatus:
    """Read the first battery, without treating a desktop/no-battery as zero."""

    try:
        batteries = sorted(path for path in root.glob("BAT*") if path.is_dir())
    except OSError:
        return PowerStatus(None)
    for battery in batteries:
        try:
            raw_capacity = (battery / "capacity").read_text(encoding="utf-8").strip()
            capacity = max(0, min(100, int(raw_capacity)))
            state = (battery / "status").read_text(encoding="utf-8").strip().casefold()
        except (OSError, ValueError):
            continue
        return PowerStatus(capacity, state or "unknown", str(battery))
    return PowerStatus(None)


def request_poweroff() -> None:
    """Ask systemd to power off after the guard has stopped active work."""

    subprocess.run(["systemctl", "poweroff"], check=False, timeout=15)


class PowerZeroGuard:
    """Trip once when a real battery reports zero and stop further work."""

    def __init__(
        self,
        *,
        reader: Callable[[], PowerStatus] = read_power_status,
        shutdown: Callable[[], None] = request_poweroff,
    ) -> None:
        self.reader = reader
        self.shutdown = shutdown
        self.tripped = False

    def poll(self) -> PowerStatus:
        status = self.reader()
        self.trip(status)
        return status

    def trip(self, status: PowerStatus) -> None:
        if status.capacity == 0 and not self.tripped:
            self.tripped = True
            self.shutdown()
