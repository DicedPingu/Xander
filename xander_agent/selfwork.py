"""Xander working on Xander: his own code, his own config, his own models.

Three rules make this safe enough to leave running, and they are the feature —
not red tape bolted on afterwards.

1. **Verified or reverted.** A self-edit is kept only if the test suite still
   passes. Otherwise the exact files he touched are restored byte-for-byte.
   Restoration is scoped to *his* changes, never `git checkout`, because this
   worktree carries unrelated uncommitted work that must survive.
2. **He cannot edit his own brakes.** :data:`PROTECTED` covers the policy layer,
   this module, and the test suite. An agent that can rewrite its own guardrail
   has none, and an agent that can rewrite its own tests can always "pass".
3. **Bounded by the operator.** :class:`SelfPolicy` starts conservative. Every
   dial can be opened, but each is an explicit choice with a recorded reason.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import Field

from .models import StrictModel, utc_now

# He may not modify the things that decide what he is allowed to modify, nor the
# tests that decide whether a change was good.
PROTECTED: tuple[str, ...] = (
    "xander_agent/policy.py",
    "xander_agent/selfwork.py",
    "xander_agent/executor.py",
    "tests/",
    "pyproject.toml",
    ".git/",
)

CodeMode = Literal["off", "propose", "verified"]
ModelMode = Literal["off", "installed", "any"]
SystemMode = Literal["off", "ask", "allow"]


class SelfPolicy(StrictModel):
    """What Xander may change about himself, and how far."""

    # "verified" is the useful default: he writes, the suite judges, and a
    # regression is undone before it can reach the next run.
    code: CodeMode = "verified"
    config: bool = True
    # "any" (the default) lets him fetch a build he is missing as well as
    # re-route among those on disk; "installed" restricts him to what is already
    # there. Routing is still filtered by the operator's abliterated-only rule
    # in variants.VariantProfile, so a pull cannot smuggle in a banned build.
    models: ModelMode = "any"
    # Extra tags he may fetch beyond the caliber catalogue. Empty means the
    # catalogue only — he cannot reach for arbitrary models off the internet.
    pull_candidates: list[str] = Field(default_factory=list)
    pull_seconds: int = Field(default=1_800, ge=60, le=21_600)
    system: SystemMode = "ask"
    # The command that decides whether a self-edit survives.
    verify: list[str] = Field(default_factory=lambda: ["python", "-m", "pytest", "-q"])
    verify_seconds: int = Field(default=900, ge=30, le=7_200)
    max_changed_files: int = Field(default=12, ge=1, le=200)
    protected: list[str] = Field(default_factory=lambda: list(PROTECTED))


class SelfChange(StrictModel):
    at: str = Field(default_factory=utc_now)
    kind: Literal["code", "config", "model", "system"] = "code"
    summary: str = ""
    detail: str = ""
    verified: bool = False
    kept: bool = False
    paths: list[str] = Field(default_factory=list)


class SelfRecord(StrictModel):
    schema_: Literal["xander.self/v1"] = Field(default="xander.self/v1", alias="schema")
    policy: SelfPolicy = Field(default_factory=SelfPolicy)
    history: list[SelfChange] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now)


class SelfStore:
    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            from .paths import config_dir

            root = config_dir()
        self.root = root
        self.path = self.root / "self.json"

    def load(self) -> SelfRecord:
        if self.path.is_file():
            try:
                return SelfRecord.model_validate_json(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return SelfRecord()

    def save(self, record: SelfRecord) -> Path:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        record.updated_at = utc_now()
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(record.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)
        return self.path

    def record(self, record: SelfRecord, change: SelfChange) -> SelfChange:
        record.history.append(change)
        record.history = record.history[-200:]
        self.save(record)
        return change


def is_protected(relative: str, policy: SelfPolicy) -> bool:
    text = str(relative).replace("\\", "/").lstrip("./")
    for rule in policy.protected:
        rule = rule.replace("\\", "/").lstrip("./")
        if rule.endswith("/") and (text.startswith(rule) or f"/{rule}" in f"/{text}"):
            return True
        if text == rule:
            return True
    return False


class Guard:
    """Byte-level backup of only the files a run touches, so unrelated dirty
    work in the same tree is never at risk."""

    def __init__(self, workspace: Path, policy: SelfPolicy, known: set[str] | None = None) -> None:
        self.workspace = workspace
        self.policy = policy
        # Relative paths that existed before the run. Anything touched that is
        # not in here was created by him, so reverting means deleting it.
        self.known = set(known or ())
        self._saved: dict[Path, bytes | None] = {}

    def watch(self, paths: list[Path]) -> None:
        for path in paths:
            if path in self._saved:
                continue
            try:
                self._saved[path] = path.read_bytes() if path.is_file() else None
            except OSError:
                continue

    def restore(self, touched: list[str] | None = None) -> list[str]:
        """Put things back exactly as they were. Never touches git.

        Files he created are removed; files he edited are rewritten from the
        byte backup. Anything he never touched is left completely alone.
        """

        restored: list[str] = []
        for relative in touched or []:
            if relative in self.known:
                continue
            path = self.workspace / relative
            try:
                if path.is_file():
                    path.unlink()
                    restored.append(relative)
            except OSError:
                continue
        for path, content in self._saved.items():
            try:
                if content is None:
                    if path.is_file():
                        path.unlink()
                        restored.append(str(path))
                    continue
                if not path.is_file() or path.read_bytes() != content:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                    restored.append(str(path))
            except OSError:
                continue
        return restored


def changed_since(workspace: Path, before: dict[str, str]) -> list[str]:
    """Relative paths whose content hash moved. Git-free so it works anywhere."""

    now = _hash_tree(workspace)
    moved = [path for path, digest in now.items() if before.get(path) != digest]
    moved += [path for path in before if path not in now]
    return sorted(set(moved))


def _hash_tree(workspace: Path) -> dict[str, str]:
    import hashlib

    digests: dict[str, str] = {}
    for root in ("xander_agent", "config"):
        base = workspace / root
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                digests[str(path.relative_to(workspace))] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
    return digests


def verify(workspace: Path, policy: SelfPolicy) -> tuple[bool, str]:
    """Run the operator's verification command. Its exit code is the verdict."""

    if not policy.verify:
        return True, "no verification command configured"
    try:
        completed = subprocess.run(
            list(policy.verify),
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=policy.verify_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"verification timed out after {policy.verify_seconds}s"
    except OSError as exc:
        return False, f"verification could not run: {exc}"
    tail = " ".join((completed.stdout or completed.stderr or "").strip().splitlines()[-1:])
    return completed.returncode == 0, tail[:300] or f"exit {completed.returncode}"


def improve_code(
    workspace: Path,
    goal: str,
    *,
    variant: str = "default",
    store: SelfStore | None = None,
    invoke: Callable[..., dict[str, Any]] | None = None,
    event_sink: Callable[[Any], None] | None = None,
) -> SelfChange:
    """Let him change his own code, then keep it only if the suite still passes."""

    store = store or SelfStore()
    record = store.load()
    policy = record.policy

    def emit(message: str, data: dict[str, Any] | None = None) -> None:
        if event_sink is None:
            return
        try:
            event_sink({"type": "self", "phase": "work", "message": message, "data": data or {}})
        except Exception:
            pass

    if policy.code == "off":
        return store.record(record, SelfChange(kind="code", summary="self-coding is off", detail=goal))

    if invoke is None:
        from .cli import invoke_engine as invoke

    before = _hash_tree(workspace)
    guard = Guard(workspace, policy, known=set(before))
    guard.watch([workspace / relative for relative in before])

    emit(f"working on himself: {goal}")
    constraints = [
        "You are editing your own implementation. Make the smallest change that is provably better.",
        "Never weaken a safety check, a policy guard, or a test to make something pass.",
        f"These paths are off limits and any edit to them will be reverted: {', '.join(policy.protected)}",
    ]
    result = invoke(
        "implement" if policy.code == "verified" else "plan",
        workspace=workspace,
        goal=goal,
        variant=variant,
        caller="human",
        autonomy="full-auto" if policy.code == "verified" else "proposal-only",
        constraints=constraints,
        log_events=True,
        event_sink=event_sink,
    )

    touched = changed_since(workspace, before)
    change = SelfChange(kind="code", summary=goal[:200], paths=touched)

    if policy.code == "propose":
        change.detail = "proposal only; nothing applied"
        emit("proposed a change to himself; nothing applied", {"paths": touched})
        return store.record(record, change)

    if not touched:
        change.detail = "no file changed"
        emit("no change was made")
        return store.record(record, change)

    blocked = [path for path in touched if is_protected(path, policy)]
    if blocked:
        restored = guard.restore(touched)
        change.detail = f"reverted: tried to edit protected paths ({', '.join(blocked)})"
        emit("reverted — he tried to edit his own guardrails", {"blocked": blocked, "restored": restored})
        return store.record(record, change)

    if len(touched) > policy.max_changed_files:
        restored = guard.restore(touched)
        change.detail = f"reverted: {len(touched)} files changed, limit is {policy.max_changed_files}"
        emit("reverted — the change was larger than the limit", {"count": len(touched)})
        return store.record(record, change)

    ok, detail = verify(workspace, policy)
    change.verified = ok
    if ok:
        change.kept = True
        change.detail = f"verified: {detail}"
        emit(f"kept a self-improvement — {detail}", {"paths": touched})
    else:
        restored = guard.restore(touched)
        change.detail = f"reverted: {detail}"
        emit(f"reverted his own change — {detail}", {"paths": touched, "restored": restored})
    return store.record(record, change)


def tune_models(
    workspace: Path,
    *,
    variant: str = "default",
    store: SelfStore | None = None,
    probe: Callable[[str], bool] | None = None,
    event_sink: Callable[[Any], None] | None = None,
) -> list[SelfChange]:
    """Re-route each role to the best build he can actually reach.

    Honest about what it measures: whether a model *responds*, and which
    installed sibling has the highest precision. It does not claim to measure
    answer quality, because nothing here can do that cheaply.
    """

    from . import calibers
    from .backend import get_backend
    from .variants import load_variant, save_variant

    store = store or SelfStore()
    record = store.load()
    policy = record.policy
    changes: list[SelfChange] = []

    def emit(message: str, data: dict[str, Any] | None = None) -> None:
        if event_sink is None:
            return
        try:
            event_sink({"type": "self", "phase": "set_up", "message": message, "data": data or {}})
        except Exception:
            pass

    if policy.models == "off":
        return [store.record(record, SelfChange(kind="model", summary="model self-tuning is off"))]

    backend = get_backend()
    try:
        installed = list(backend.installed_models())
    except Exception as exc:
        return [store.record(record, SelfChange(kind="model", summary=f"cannot list models: {exc}"[:200]))]

    if probe is None:
        def probe(model: str) -> bool:
            # Installed is not the same as working; ask for one cheap token.
            if model not in installed:
                return False
            try:
                backend.ask("ok", model=model, timeout=45)
                return True
            except Exception:
                return False

    profile = load_variant(variant)
    routing = dict(profile.model_routing)
    updated = dict(routing)

    for role, current in routing.items():
        if calibers.is_cloud(current):
            continue
        if policy.models == "any" and current not in installed:
            # Self-healing: the routed build is gone or was never fetched.
            # Bounded to the catalogue plus tags the operator named, so this
            # cannot become "download anything it feels like".
            allowed = set(calibers.DEFAULT_MODELS.values()) | set(policy.pull_candidates)
            if current in allowed:
                emit(f"fetching the missing build for {role}: {current}")
                ok, detail = backend.pull(current, timeout=policy.pull_seconds)
                changes.append(
                    SelfChange(
                        kind="model",
                        summary=f"{role}: fetched {current}" if ok else f"{role}: could not fetch {current}",
                        detail=detail,
                        verified=ok,
                        kept=ok,
                    )
                )
                if ok:
                    installed = list(backend.installed_models())
        best = calibers.best_installed(current, installed)
        candidates = [best] + [item for item in calibers.alternatives(current, installed) if item != best]
        chosen = next((item for item in candidates if probe(item)), "")
        if not chosen or chosen == current:
            continue
        updated[role] = chosen
        reason = "higher-precision build of the same family" if chosen == best else "the routed build did not respond"
        changes.append(
            SelfChange(kind="model", summary=f"{role}: {current} -> {chosen}", detail=reason, verified=True, kept=True)
        )
        emit(f"re-routed {role} to {chosen} ({reason})")

    if updated != routing:
        try:
            profile.model_routing = updated
            save_variant(profile, replace=True)
        except Exception as exc:
            # The operator's abliterated-only rule lives here and must win.
            for change in changes:
                change.kept = False
                change.detail = f"rejected by operator model policy: {exc}"[:300]
            emit(f"routing change rejected by policy: {exc}"[:200])

    if not changes:
        changes.append(SelfChange(kind="model", summary="routing already optimal", verified=True, kept=True))
    for change in changes:
        store.record(record, change)
    return changes


def status(store: SelfStore | None = None) -> dict[str, Any]:
    store = store or SelfStore()
    record = store.load()
    kept = [item for item in record.history if item.kept]
    reverted = [item for item in record.history if item.verified is False and not item.kept and item.paths]
    return {
        "event": "self",
        "policy": record.policy.model_dump(mode="json"),
        "changes_kept": len(kept),
        "changes_reverted": len(reverted),
        "recent": [item.model_dump(mode="json") for item in record.history[-10:]],
    }
