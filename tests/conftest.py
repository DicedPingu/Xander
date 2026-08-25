"""Suite-wide guards: tests never leave the machine."""

from __future__ import annotations

import os

os.environ.setdefault("XANDER_OFFLINE", "1")
