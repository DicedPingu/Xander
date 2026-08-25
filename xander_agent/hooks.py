"""Operator hooks: declared broad, narrowed to the mission at hand.

A hook is written wide — a moment it attaches to and an optional match
pattern — and lives in ``hooks.json`` in Xander's config directory. When a
task starts, only the hooks whose pattern matches the goal are armed; the
rest stay on the shelf. That is the whole model: flexible at rest,
narrowed at muster.

Hooks shape the work, they never do the work: a ``constraint`` hook joins
the task's effective constraints, a ``note`` hook is spoken/logged at its
moment. No hook executes commands.

Moments: ``analyze``, ``plan``, ``test``, ``victory``, ``setback``, or
``*`` for every moment.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

MOMENTS = ("analyze", "plan", "test", "victory", "setback", "*")


def _hooks_path() -> Path:
    from .paths import config_dir

    return config_dir() / "hooks.json"


@dataclass(frozen=True)
class Hook:
    on: str = "*"
    match: str = ""  # empty means: always relevant
    constraint: str = ""
    note: str = ""

    def relevant(self, goal: str) -> bool:
        if not self.match:
            return True
        try:
            return bool(re.search(self.match, goal, re.IGNORECASE))
        except re.error:
            return self.match.casefold() in goal.casefold()


class HookBook:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _hooks_path()
        self.hooks: list[Hook] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, dict):
                continue
            hook = Hook(
                on=str(item.get("on", "*")) if str(item.get("on", "*")) in MOMENTS else "*",
                match=str(item.get("match", "")),
                constraint=str(item.get("constraint", "")).strip(),
                note=str(item.get("note", "")).strip(),
            )
            if hook.constraint or hook.note:
                self.hooks.append(hook)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"on": hook.on, "match": hook.match, "constraint": hook.constraint, "note": hook.note}
            for hook in self.hooks
        ]
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def add(self, *, on: str = "*", match: str = "", constraint: str = "", note: str = "") -> Hook:
        hook = Hook(on=on if on in MOMENTS else "*", match=match, constraint=constraint.strip(), note=note.strip())
        if not (hook.constraint or hook.note):
            raise ValueError("a hook needs a constraint or a note")
        self.hooks.append(hook)
        self.save()
        return hook

    # -- narrowing ------------------------------------------------------------
    def narrowed(self, goal: str) -> list[Hook]:
        """The broad book, cut down to what this mission is actually about."""

        return [hook for hook in self.hooks if hook.relevant(goal)]

    def constraints_for(self, goal: str) -> list[str]:
        return list(dict.fromkeys(hook.constraint for hook in self.narrowed(goal) if hook.constraint))

    def notes_for(self, goal: str, moment: str) -> list[str]:
        return [
            hook.note
            for hook in self.narrowed(goal)
            if hook.note and hook.on in {moment, "*"}
        ]
