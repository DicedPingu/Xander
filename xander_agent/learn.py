"""A standing queue of things Xander is learning, and the order he takes them.

The operator drops in anything — a URL, a package, a skill, a category, a bare
topic — and Xander turns each into a target he can actually work. He may add
targets himself while working (a doc links three others), which is why a target
records where it came from: operator intent outranks anything he discovered.

Classification is deliberately deterministic and offline. Deciding that
``https://x/y`` is a URL does not need a model, and spending a generation on it
is exactly the wasted work the operator asked to avoid.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import Field

from .models import StrictModel, utc_now

TargetKind = Literal["url", "package", "skill", "category", "repo", "topic"]
TargetState = Literal["todo", "active", "done", "blocked", "skipped"]

# Explicit prefixes always win over shape-guessing; an operator who writes
# `topic:requests` means the subject, not the PyPI package.
_PREFIXES: dict[str, TargetKind] = {
    "url": "url",
    "link": "url",
    "repo": "repo",
    "git": "repo",
    "pypi": "package",
    "npm": "package",
    "crate": "package",
    "gem": "package",
    "go": "package",
    "pkg": "package",
    "package": "package",
    "skill": "skill",
    "category": "category",
    "cat": "category",
    "topic": "topic",
}

_ECOSYSTEMS = {"pypi", "npm", "crate", "gem", "go"}
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_BARE_HOST_RE = re.compile(r"^(?:www\.|[a-z0-9-]+\.)+[a-z]{2,}(?:/|$)", re.IGNORECASE)
_PACKAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._@/-]{0,80}$", re.IGNORECASE)


class LearnTarget(StrictModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    raw: str
    subject: str
    kind: TargetKind = "topic"
    ecosystem: str = ""
    state: TargetState = "todo"
    # Lower sorts first. Operator-added work defaults ahead of anything Xander
    # discovered for himself, so his own curiosity never outranks the order.
    priority: int = 100
    origin: Literal["operator", "discovered"] = "operator"
    parent: str = ""
    evidence: str = ""
    lesson: str = ""
    attempts: int = 0
    created_at: str = Field(default_factory=utc_now)


class LearnSettings(StrictModel):
    # How many follow-up targets Xander may add from one target's findings.
    # 0 means "learn exactly what I gave you and nothing else".
    depth: int = Field(default=2, ge=0, le=10)
    max_targets: int = Field(default=60, ge=1, le=500)
    per_target_seconds: int = Field(default=600, ge=60, le=14_400)
    allow_network: bool = True
    # Write what he learns back as a durable skill card, not just a lesson line.
    author_skills: bool = True
    mode: str = "research"
    autonomy: str = "full-auto"


class LearnQueue(StrictModel):
    schema_: Literal["xander.learn/v1"] = Field(default="xander.learn/v1", alias="schema")
    workspace: str
    mission: str = ""
    targets: list[LearnTarget] = Field(default_factory=list)
    # Hard rules the operator added mid-flight. These reach every role through
    # the planner's constraint list, so a restriction changes how the planner,
    # coder and critic all behave on the very next call.
    restrictions: list[str] = Field(default_factory=list)
    settings: LearnSettings = Field(default_factory=LearnSettings)
    updated_at: str = Field(default_factory=utc_now)

    def pending(self) -> list[LearnTarget]:
        """Work order, not insertion order — so callers can show what is next."""

        waiting = [target for target in self.targets if target.state in {"todo", "active"}]
        return sorted(
            waiting,
            key=lambda target: (0 if target.state == "active" else 1, target.priority, self.targets.index(target)),
        )

    def next_target(self) -> LearnTarget | None:
        """Lowest priority number first, then insertion order — a stable queue."""

        active = [target for target in self.targets if target.state == "active"]
        if active:
            return active[0]
        waiting = [target for target in self.targets if target.state == "todo"]
        if not waiting:
            return None
        return min(waiting, key=lambda target: (target.priority, self.targets.index(target)))

    def progress(self) -> str:
        done = sum(1 for target in self.targets if target.state == "done")
        return f"{done}/{len(self.targets)} learned"


def classify(raw: str) -> tuple[TargetKind, str, str]:
    """Return ``(kind, subject, ecosystem)`` for one operator token. Offline."""

    text = " ".join(str(raw).split()).strip().strip("\"'`,")
    if not text:
        return "topic", "", ""
    head, separator, tail = text.partition(":")
    key = head.casefold().strip()
    if separator and key in _PREFIXES and tail.strip() and not _URL_RE.match(text):
        subject = tail.strip()
        kind = _PREFIXES[key]
        return kind, subject, key if key in _ECOSYSTEMS else ""
    if _URL_RE.match(text):
        kind: TargetKind = "repo" if "github.com" in text.casefold() or "gitlab.com" in text.casefold() else "url"
        return kind, text, ""
    if _BARE_HOST_RE.match(text):
        return ("repo" if "github.com" in text.casefold() else "url"), f"https://{text}", ""
    if " " not in text and _PACKAGE_RE.match(text):
        return "package", text, ""
    return "topic", text, ""


class LearnStore:
    """Workspace-scoped persistence, mirroring :mod:`xander_agent.workboard`."""

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from .paths import state_dir

            root = state_dir() / "learning"
        self.root = root

    def path(self, workspace: Path) -> Path:
        resolved = workspace.expanduser().resolve(strict=False)
        digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:24]
        return self.root / f"{digest}.json"

    def load(self, workspace: Path) -> LearnQueue:
        resolved = str(workspace.expanduser().resolve(strict=False))
        candidate = self.path(workspace)
        if candidate.is_file():
            try:
                queue = LearnQueue.model_validate_json(candidate.read_text(encoding="utf-8"))
            except Exception:
                queue = None
            if queue is not None and queue.workspace == resolved:
                return queue
        return LearnQueue(workspace=resolved)

    def save(self, queue: LearnQueue) -> Path:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        queue.updated_at = utc_now()
        destination = self.path(Path(queue.workspace))
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(queue.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
        return destination

    # -- mutations ----------------------------------------------------------

    def add(
        self,
        queue: LearnQueue,
        raw: str,
        *,
        origin: Literal["operator", "discovered"] = "operator",
        parent: str = "",
        priority: int | None = None,
    ) -> LearnTarget | None:
        kind, subject, ecosystem = classify(raw)
        if not subject:
            return None
        if len(queue.targets) >= queue.settings.max_targets:
            return None
        existing = next(
            (target for target in queue.targets if target.subject.casefold() == subject.casefold()),
            None,
        )
        if existing is not None:
            # Re-adding something already queued is a priority signal, not a duplicate.
            if origin == "operator" and existing.origin == "discovered":
                existing.origin = "operator"
                existing.priority = min(existing.priority, 50)
                self.save(queue)
            return existing
        if priority is None:
            priority = 50 if origin == "operator" else 150
        target = LearnTarget(
            raw=raw,
            subject=subject,
            kind=kind,
            ecosystem=ecosystem,
            origin=origin,
            parent=parent,
            priority=priority,
        )
        queue.targets.append(target)
        self.save(queue)
        return target

    def add_many(self, queue: LearnQueue, items: list[str], **kwargs: object) -> list[LearnTarget]:
        added: list[LearnTarget] = []
        for item in items:
            target = self.add(queue, item, **kwargs)  # type: ignore[arg-type]
            if target is not None:
                added.append(target)
        return added

    def set_state(
        self,
        queue: LearnQueue,
        target_id: str,
        state: TargetState,
        *,
        evidence: str = "",
        lesson: str = "",
    ) -> bool:
        for target in queue.targets:
            if target.id != target_id:
                continue
            target.state = state
            if evidence:
                target.evidence = " ".join(evidence.split())[:600]
            if lesson:
                target.lesson = " ".join(lesson.split())[:600]
            if state == "active":
                target.attempts += 1
            self.save(queue)
            return True
        return False

    def focus(self, queue: LearnQueue, text: str) -> list[LearnTarget]:
        """Pull everything matching ``text`` to the front without dropping work."""

        needle = text.casefold().strip()
        if not needle:
            return []
        moved = [
            target
            for target in queue.targets
            if target.state in {"todo", "active"}
            and (needle in target.subject.casefold() or needle in target.raw.casefold())
        ]
        for target in moved:
            target.priority = 1
        if moved:
            self.save(queue)
        return moved

    def drop(self, queue: LearnQueue, text: str) -> list[LearnTarget]:
        needle = text.casefold().strip()
        if not needle:
            return []
        dropped = [
            target
            for target in queue.targets
            if target.state in {"todo", "active"}
            and (needle in target.subject.casefold() or needle in target.raw.casefold())
        ]
        for target in dropped:
            target.state = "skipped"
            target.evidence = "dropped by the operator"
        if dropped:
            self.save(queue)
        return dropped

    def restrict(self, queue: LearnQueue, rule: str) -> bool:
        rule = " ".join(rule.split())
        if not rule or rule in queue.restrictions:
            return False
        queue.restrictions.append(rule)
        queue.restrictions = queue.restrictions[-40:]
        self.save(queue)
        return True

    def set_mission(self, queue: LearnQueue, mission: str) -> None:
        queue.mission = " ".join(mission.split())[:600]
        self.save(queue)


def constraints_for(queue: LearnQueue) -> list[str]:
    """The lines that carry operator intent into every model role's prompt."""

    lines = [
        "Learning run: produce durable, reusable understanding backed by evidence you actually gathered.",
        "Do not run tools, installs, or checks that cannot change what you conclude about this target.",
    ]
    if queue.mission:
        lines.append(f"Standing learning mission: {queue.mission}")
    if not queue.settings.allow_network:
        lines.append("Network access is off for this run; work only from local material.")
    lines.extend(f"Operator restriction: {rule}" for rule in queue.restrictions[-10:])
    return lines
