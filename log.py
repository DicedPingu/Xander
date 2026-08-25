"""Structured, grep-able logging for Xander — same line format as Monica so a
single `grep event=` sweeps the whole ASKAR bench.

    [2026-07-10T04:40:00] [INFO] [xander] message | key=val key=val

Writes to logs/xander.log always; echoes to the console unless quiet; emits
JSONL to stdout when XANDER_JSON=1 (for machine consumers / other agents).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys

from config import LOG_DIR

_JSON = os.environ.get("XANDER_JSON") == "1"
_FILE = LOG_DIR / "xander.log"

# level -> plain glyph (console styling is handled by the UI, not here)
_GLYPH = {"INFO": "*", "OK": "✓", "WARN": "!", "ERROR": "✗", "STEP": "»", "EVENT": "·"}


def _ts() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _fields(fields: dict) -> str:
    if not fields:
        return ""
    parts = []
    for k, v in fields.items():
        s = str(v).replace("\n", " ")
        if " " in s:
            s = f'"{s}"'
        parts.append(f"{k}={s}")
    return " | " + " ".join(parts)


class Log:
    def __init__(self, worker: str = "xander", quiet: bool = False):
        self.worker = worker
        self.quiet = quiet or _JSON

    def _emit(self, level: str, msg: str, **fields) -> None:
        line = f"[{_ts()}] [{level}] [{self.worker}] {msg}{_fields(fields)}"
        try:
            with _FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        if _JSON:
            rec = {"ts": _ts(), "level": level, "worker": self.worker, "msg": msg}
            rec.update(fields)
            sys.stdout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            return
        if self.quiet:
            return
        print(f"[{_GLYPH.get(level, '*')}] {msg}{_fields(fields)}")

    def info(self, msg: str, **f) -> None:
        self._emit("INFO", msg, **f)

    def ok(self, msg: str, **f) -> None:
        self._emit("OK", msg, **f)

    def warn(self, msg: str, **f) -> None:
        self._emit("WARN", msg, **f)

    def error(self, msg: str, **f) -> None:
        self._emit("ERROR", msg, **f)

    def event(self, kind: str, **f) -> None:
        self._emit("EVENT", kind, event=kind, **f)
