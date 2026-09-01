"""The back channel: the operator talks while Xander works.

Posting is an append to a file and nothing else — it never waits for a model, a
lock, or the engine, so typing mid-run cannot stall the run. Xander drains the
inbox only at *safe points* (between actions, between attempts, between learning
targets), never inside a running step, so a comment reshapes the next decision
instead of corrupting the one in flight.

Reading is two-tier on purpose. A `!` command is parsed deterministically at zero
cost and zero latency; only free prose falls through to the cheap `classifier`
role. Interpreting "skip this one" with an 8B planner would be exactly the wasted
work this module exists to avoid.

Directives land in the shared :class:`~xander_agent.learn.LearnQueue`, whose
mission and restrictions are folded into every prompt. That is the mechanism by
which one note changes how *every* role subsequently thinks, not just the loop.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from .learn import LearnQueue, LearnStore
from .models import StrictModel, utc_now

DirectiveKind = Literal["focus", "drop", "add", "mission", "restrict", "setting", "stop", "note"]
Urgency = Literal["now", "next"]

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# `!command argument` — the fast path. Aliases are generous because the operator
# is typing mid-run and should not have to remember one exact spelling.
_FAST: dict[str, tuple[DirectiveKind, Urgency]] = {
    "focus": ("focus", "now"),
    "first": ("focus", "now"),
    "now": ("focus", "now"),
    "drop": ("drop", "now"),
    "skip": ("drop", "now"),
    "forget": ("drop", "now"),
    "add": ("add", "next"),
    "learn": ("add", "next"),
    "also": ("add", "next"),
    "mission": ("mission", "now"),
    "goal": ("mission", "now"),
    "never": ("restrict", "now"),
    "dont": ("restrict", "now"),
    "restrict": ("restrict", "now"),
    "set": ("setting", "now"),
    "stop": ("stop", "now"),
    "next": ("stop", "now"),
    "note": ("note", "next"),
}


class SteeringDirective(StrictModel):
    kind: DirectiveKind = "note"
    argument: str = ""
    urgency: Urgency = "next"
    summary: str = ""


class SteeringNote(StrictModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    text: str
    received_at: str = Field(default_factory=utc_now)
    applied_at: str = ""
    outcome: str = ""
    directive: SteeringDirective | None = None


def parse_fast(text: str) -> SteeringDirective | None:
    """Deterministic `!` commands. No model, no network, no latency."""

    cleaned = _CONTROL_RE.sub(" ", str(text)).strip()
    if not cleaned.startswith("!"):
        return None
    head, _, tail = cleaned[1:].partition(" ")
    key = head.casefold().strip().rstrip(":")
    entry = _FAST.get(key)
    if entry is None:
        return None
    kind, urgency = entry
    argument = tail.strip()
    # "!never install X" must survive as a prohibition. Storing the bare tail
    # would invert the operator's intent into an instruction to do the thing.
    if key in {"never", "dont"} and argument:
        argument = f"never {argument}"
    if kind in {"focus", "drop", "add", "mission", "restrict", "setting"} and not argument:
        return None
    return SteeringDirective(
        kind=kind,
        argument=argument,
        urgency=urgency,
        summary=f"!{key} {argument}".strip(),
    )


_INTERPRET_PROMPT = """
Classify one message the operator typed while an autonomous learning run was working.
Return only the schema. Do not answer the message; route it.

MESSAGE: {text}

CURRENT MISSION: {mission}
PENDING TARGETS: {targets}

kind:
- focus     they want existing queued work done sooner; argument = what to bring forward
- drop      they want queued work skipped; argument = what to drop
- add       they want something new learned; argument = the thing, one item per comma
- mission   they are restating the overall objective; argument = the new mission
- restrict  a hard rule to obey from now on; argument = the rule
- setting   a knob change (depth, network, mode, authority); argument = "name value"
- stop      abandon the target in flight and move on; argument = ""
- note      context worth recording that changes no plan; argument = ""

urgency: "now" if the current step should be reshaped immediately, else "next".
summary: one short line an operator can read in a log.
""".strip()


def interpret(text: str, backend: Any | None = None, queue: LearnQueue | None = None) -> SteeringDirective:
    """Fast path first; the cheap classifier only for genuine prose."""

    fast = parse_fast(text)
    if fast is not None:
        return fast
    cleaned = " ".join(_CONTROL_RE.sub(" ", str(text)).split())[:800]
    if not cleaned:
        return SteeringDirective(kind="note", summary="(empty)")
    if backend is None or not getattr(backend, "available", lambda: False)():
        return SteeringDirective(kind="note", argument=cleaned, summary=cleaned[:120])
    pending = ", ".join(target.subject for target in (queue.pending() if queue else [])[:8]) or "none"
    prompt = _INTERPRET_PROMPT.format(
        text=cleaned,
        mission=(queue.mission if queue else "") or "none set",
        targets=pending,
    )
    try:
        raw = backend.generate(prompt, role="classifier", schema=SteeringDirective, think=False, timeout=60)
        directive = SteeringDirective.model_validate_json(raw)
    except Exception:
        # A steering note must never be lost because a model hiccuped.
        return SteeringDirective(kind="note", argument=cleaned, summary=cleaned[:120])
    if not directive.summary:
        directive.summary = cleaned[:120]
    return directive


class SteeringInbox:
    """Append-only JSON-lines channel, one file per workspace."""

    def __init__(self, workspace: Path, root: Path | None = None) -> None:
        self.workspace = workspace.expanduser().resolve(strict=False)
        if root is None:
            import hashlib

            from .paths import state_dir

            root = state_dir() / "steering"
        self.root = root
        import hashlib

        digest = hashlib.sha256(str(self.workspace).encode("utf-8")).hexdigest()[:24]
        self.path = self.root / f"{digest}.jsonl"

    def post(self, text: str) -> SteeringNote | None:
        """Record one comment. Cheap, non-blocking, safe to call mid-run."""

        cleaned = " ".join(_CONTROL_RE.sub(" ", str(text)).split())[:2_000]
        if not cleaned:
            return None
        note = SteeringNote(text=cleaned)
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(note.model_dump(mode="json"), ensure_ascii=False) + "\n")
            os.chmod(self.path, 0o600)
        except OSError:
            return None
        return note

    def pending(self) -> list[SteeringNote]:
        if not self.path.is_file():
            return []
        notes: list[SteeringNote] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    note = SteeringNote.model_validate_json(line)
                except Exception:
                    continue
                if not note.applied_at:
                    notes.append(note)
        except OSError:
            return []
        return notes

    def mark_applied(self, applied: list[SteeringNote]) -> None:
        """Rewrite the log with outcomes recorded, preserving the full history."""

        if not applied or not self.path.is_file():
            return
        outcomes = {note.id: note for note in applied}
        rows: list[str] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    note = SteeringNote.model_validate_json(line)
                except Exception:
                    rows.append(line)
                    continue
                replacement = outcomes.get(note.id)
                if replacement is not None:
                    note = replacement
                rows.append(json.dumps(note.model_dump(mode="json"), ensure_ascii=False))
            temporary = self.path.with_suffix(".jsonl.tmp")
            temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
            os.chmod(temporary, 0o600)
            temporary.replace(self.path)
        except OSError:
            return

    def history(self, limit: int = 40) -> list[SteeringNote]:
        if not self.path.is_file():
            return []
        notes: list[SteeringNote] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    notes.append(SteeringNote.model_validate_json(line))
                except Exception:
                    continue
        except OSError:
            return []
        return notes[-limit:]


_SETTING_ALIASES = {
    "depth": "depth",
    "max": "max_targets",
    "max_targets": "max_targets",
    "budget": "per_target_seconds",
    "seconds": "per_target_seconds",
    "per_target_seconds": "per_target_seconds",
    "network": "allow_network",
    "online": "allow_network",
    "skills": "author_skills",
    "author_skills": "author_skills",
    "mode": "mode",
    "authority": "autonomy",
    "autonomy": "autonomy",
}
_TRUTHY = {"on", "true", "yes", "1", "allow", "allowed"}
_FALSEY = {"off", "false", "no", "0", "deny", "denied"}


def apply(directive: SteeringDirective, queue: LearnQueue, store: LearnStore) -> str:
    """Mutate the queue. Returns a human line for the log; never raises."""

    argument = directive.argument.strip()
    try:
        if directive.kind == "focus":
            moved = store.focus(queue, argument)
            if not moved:
                return f"nothing queued matched '{argument}'"
            return "brought forward: " + ", ".join(target.subject for target in moved[:5])
        if directive.kind == "drop":
            dropped = store.drop(queue, argument)
            if not dropped:
                return f"nothing queued matched '{argument}'"
            return "dropped: " + ", ".join(target.subject for target in dropped[:5])
        if directive.kind == "add":
            items = [part.strip() for part in re.split(r"[,;\n]| and ", argument) if part.strip()]
            added = store.add_many(queue, items, origin="operator", priority=40)
            if not added:
                return "nothing new to add"
            return "queued: " + ", ".join(target.subject for target in added[:5])
        if directive.kind == "mission":
            store.set_mission(queue, argument)
            return f"mission is now: {queue.mission}"
        if directive.kind == "restrict":
            return f"restriction added: {argument}" if store.restrict(queue, argument) else "restriction already active"
        if directive.kind == "setting":
            return _apply_setting(queue, store, argument)
        if directive.kind == "stop":
            active = [target for target in queue.targets if target.state == "active"]
            for target in active:
                target.state = "skipped"
                target.evidence = "stopped by the operator mid-target"
            store.save(queue)
            if not active:
                return "nothing was in flight"
            return "stopped: " + ", ".join(target.subject for target in active)
        return "recorded"
    except Exception as exc:  # a bad note must not end the run
        return f"could not apply ({type(exc).__name__})"


def _apply_setting(queue: LearnQueue, store: LearnStore, argument: str) -> str:
    parts = argument.replace("=", " ").split()
    if len(parts) < 2:
        return "use: set <depth|max|budget|network|skills|mode|authority> <value>"
    name = _SETTING_ALIASES.get(parts[0].casefold())
    value = " ".join(parts[1:]).strip()
    if name is None:
        return f"unknown setting '{parts[0]}'"
    settings = queue.settings
    if name in {"allow_network", "author_skills"}:
        lowered = value.casefold()
        if lowered not in _TRUTHY and lowered not in _FALSEY:
            return f"{name} takes on/off, got '{value}'"
        setattr(settings, name, lowered in _TRUTHY)
    elif name in {"depth", "max_targets", "per_target_seconds"}:
        try:
            number = int(value)
        except ValueError:
            return f"{name} takes a whole number, got '{value}'"
        try:
            setattr(settings, name, number)
        except Exception:
            return f"{number} is outside the allowed range for {name}"
    else:
        setattr(settings, name, value)
    store.save(queue)
    return f"{name} = {getattr(settings, name)}"


def drain(
    workspace: Path,
    queue: LearnQueue,
    store: LearnStore,
    *,
    backend: Any | None = None,
    inbox: SteeringInbox | None = None,
) -> list[tuple[SteeringNote, str]]:
    """Read, interpret and apply every waiting note. Called at safe points only."""

    inbox = inbox or SteeringInbox(workspace)
    notes = inbox.pending()
    if not notes:
        return []
    handled: list[tuple[SteeringNote, str]] = []
    for note in notes:
        directive = interpret(note.text, backend=backend, queue=queue)
        outcome = apply(directive, queue, store)
        note.directive = directive
        note.outcome = outcome
        note.applied_at = datetime.now().isoformat(timespec="seconds")
        handled.append((note, outcome))
    inbox.mark_applied([note for note, _ in handled])
    return handled
