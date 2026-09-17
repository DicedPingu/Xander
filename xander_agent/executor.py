from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from .models import Action, ActionKind, ActionResult, ActionStatus, WorkspaceSnapshot, utc_now
from .policy import (
    apply_edit_blocks,
    approval_reason,
    changed_paths,
    harden_argv,
    resolve_inside,
    safe_environment,
    sha256_file,
)


MAX_CAPTURE = 64_000
Approval = Callable[[Action, str], bool]


def _trim(text: str) -> str:
    if len(text) <= MAX_CAPTURE:
        return text
    return text[:MAX_CAPTURE] + "\n...[output truncated]"


def _redact(text: str) -> str:
    patterns = (
        re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s]+"),
        re.compile(r"(?i)((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s]+"),
        re.compile(r"\b(?:sk|ghp|github_pat|ctx7sk)-[A-Za-z0-9_-]{12,}\b"),
    )
    for pattern in patterns:
        text = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", text)
    return text


def _stop_process(process: subprocess.Popen[object]) -> None:
    if os.name == "posix":
        process_group = process.pid
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait(timeout=2)
            return
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=2)
        return
    if process.poll() is None:  # pragma: no cover - Xander currently targets Linux
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


class ActionExecutor:
    def __init__(
        self,
        workspace: Path,
        snapshot: WorkspaceSnapshot,
        *,
        autonomy: str = "full-auto",
        allowed_paths: list[str] | None = None,
        approve: Approval | None = None,
        default_timeout: int = 900,
    ) -> None:
        self.workspace = workspace.expanduser().resolve(strict=True)
        self.snapshot = snapshot
        self.autonomy = autonomy
        self.allowed_paths = allowed_paths or []
        self.approve = approve
        self.default_timeout = default_timeout

    def run(self, action: Action) -> ActionResult:
        started = utc_now()
        # Package commands run as they must (root, unattended) and the
        # operator is asked about that exact form, never a softer one.
        if action.argv:
            action.argv = harden_argv(action.argv)
        if action.pipeline:
            action.pipeline = [harden_argv(stage) for stage in action.pipeline]
        try:
            reason = approval_reason(
                action,
                self.workspace,
                self.snapshot,
                self.autonomy,
                self.allowed_paths,
            )
        except ValueError as exc:
            return self._result(action, ActionStatus.BLOCKED, started, reason=str(exc))
        if reason:
            if not self.approve or not self.approve(action, reason):
                return self._result(action, ActionStatus.BLOCKED, started, reason=reason)
        try:
            if action.kind in {ActionKind.INSPECT, ActionKind.COMMAND}:
                return self._command(action, started)
            if action.kind == ActionKind.PIPELINE:
                return self._pipeline(action, started)
            if action.kind == ActionKind.PATCH:
                return self._patch(action, started)
            if action.kind == ActionKind.CREATE:
                return self._create(action, started)
            if action.kind == ActionKind.EDIT:
                return self._edit(action, started)
            if action.kind == ActionKind.NOTE:
                return self._result(action, ActionStatus.OK, started, stdout=action.content or action.expected)
            return self._result(action, ActionStatus.FAILED, started, reason="unsupported action kind")
        except subprocess.TimeoutExpired as exc:
            stdout = _redact(_trim(_as_text(exc.stdout)))
            stderr = _redact(_trim(_as_text(exc.stderr)))
            return self._result(
                action,
                ActionStatus.FAILED,
                started,
                returncode=124,
                stdout=stdout,
                stderr=stderr,
                reason=f"timed out after {exc.timeout}s",
            )
        except Exception as exc:
            return self._result(action, ActionStatus.FAILED, started, reason=str(exc))

    def _cwd(self, action: Action) -> Path:
        return resolve_inside(self.workspace, action.cwd or ".")

    def _command(self, action: Action, started: str) -> ActionResult:
        if not action.argv:
            return self._result(action, ActionStatus.FAILED, started, reason="missing argv")
        process = subprocess.Popen(
            action.argv,
            cwd=self._cwd(action),
            env=safe_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = process.communicate(timeout=self.default_timeout)
        except subprocess.TimeoutExpired as exc:
            _stop_process(process)
            remaining_stdout, remaining_stderr = process.communicate(timeout=2)
            if exc.stdout is None:
                exc.stdout = remaining_stdout
            if exc.stderr is None:
                exc.stderr = remaining_stderr
            raise
        _stop_process(process)
        status = ActionStatus.OK if process.returncode == 0 else ActionStatus.FAILED
        return self._result(
            action,
            status,
            started,
            returncode=process.returncode,
            stdout=_redact(_trim(stdout)),
            stderr=_redact(_trim(stderr)),
        )

    def _pipeline(self, action: Action, started: str) -> ActionResult:
        if not action.pipeline or any(not stage for stage in action.pipeline):
            return self._result(action, ActionStatus.FAILED, started, reason="invalid pipeline")
        processes: list[subprocess.Popen[bytes]] = []
        error_streams = []
        previous = None
        cwd = self._cwd(action)
        try:
            for stage in action.pipeline:
                error_stream = tempfile.TemporaryFile()
                error_streams.append(error_stream)
                process = subprocess.Popen(
                    stage,
                    cwd=cwd,
                    env=safe_environment(),
                    stdin=previous.stdout if previous else None,
                    stdout=subprocess.PIPE,
                    stderr=error_stream,
                    shell=False,
                    start_new_session=os.name == "posix",
                )
                if previous and previous.stdout:
                    previous.stdout.close()
                processes.append(process)
                previous = process
            assert processes[-1].stdout is not None
            try:
                stdout, _ = processes[-1].communicate(timeout=self.default_timeout)
                for process in processes[:-1]:
                    process.wait(timeout=10)
            except subprocess.TimeoutExpired as exc:
                for process in processes:
                    _stop_process(process)
                remaining_stdout, _ = processes[-1].communicate(timeout=2)
                if exc.stdout is None:
                    exc.stdout = remaining_stdout
                exc.stderr = self._pipeline_errors(error_streams)
                raise
            returncodes = [process.returncode for process in processes]
            returncode = next((code for code in returncodes if code), 0)
            all_stderr = self._pipeline_errors(error_streams)
            status = ActionStatus.OK if returncode == 0 else ActionStatus.FAILED
            return self._result(
                action,
                status,
                started,
                returncode=returncode,
                stdout=_redact(_trim(stdout.decode("utf-8", "replace"))),
                stderr=_redact(_trim(all_stderr.decode("utf-8", "replace"))),
            )
        finally:
            for process in processes:
                _stop_process(process)
            for error_stream in error_streams:
                error_stream.close()

    @staticmethod
    def _pipeline_errors(error_streams: list[object]) -> bytes:
        captured = []
        for stream in error_streams:
            stream.seek(0)
            captured.append(stream.read())
        return b"\n".join(captured)

    def _validate_preimages(self, action: Action, paths: list[Path]) -> None:
        for path in paths:
            relative = str(path.relative_to(self.workspace))
            if relative not in action.preimage_hashes:
                continue
            expected = action.preimage_hashes[relative]
            actual = sha256_file(path)
            if expected != actual:
                raise ValueError(f"preimage changed for {relative}")

    def _patch(self, action: Action, started: str) -> ActionResult:
        if not action.patch.strip():
            return self._result(action, ActionStatus.FAILED, started, reason="empty patch")
        paths = changed_paths(action, self.workspace)
        if not paths:
            return self._result(action, ActionStatus.FAILED, started, reason="patch has no affected paths")
        self._validate_preimages(action, paths)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".patch", delete=False) as handle:
            handle.write(action.patch)
            patch_path = Path(handle.name)
        try:
            check = subprocess.run(
                ["git", "apply", "--check", "--whitespace=nowarn", str(patch_path)],
                cwd=self.workspace,
                env=safe_environment(),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if check.returncode != 0:
                return self._result(
                    action,
                    ActionStatus.FAILED,
                    started,
                    returncode=check.returncode,
                    stderr=_redact(_trim(check.stderr)),
                    reason="patch check failed",
                )
            apply = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch_path)],
                cwd=self.workspace,
                env=safe_environment(),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            status = ActionStatus.OK if apply.returncode == 0 else ActionStatus.FAILED
            return self._result(
                action,
                status,
                started,
                returncode=apply.returncode,
                stdout=_redact(_trim(apply.stdout)),
                stderr=_redact(_trim(apply.stderr)),
                changed_paths=[str(path.relative_to(self.workspace)) for path in paths] if status == ActionStatus.OK else [],
            )
        finally:
            patch_path.unlink(missing_ok=True)

    def _create(self, action: Action, started: str) -> ActionResult:
        if not action.path:
            return self._result(action, ActionStatus.FAILED, started, reason="missing create path")
        destination = resolve_inside(self.workspace, action.path)
        if destination.exists() or destination.is_symlink():
            return self._result(action, ActionStatus.FAILED, started, reason="create refuses to overwrite an existing path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8", newline="") as handle:
            handle.write(action.content)
        return self._result(
            action,
            ActionStatus.OK,
            started,
            returncode=0,
            changed_paths=[str(destination.relative_to(self.workspace))],
        )

    def _edit(self, action: Action, started: str) -> ActionResult:
        if not action.path:
            return self._result(action, ActionStatus.FAILED, started, reason="missing edit path")
        if not action.edits:
            return self._result(action, ActionStatus.FAILED, started, reason="edit action has no search/replace blocks")
        destination = resolve_inside(self.workspace, action.path)
        if not destination.is_file():
            return self._result(action, ActionStatus.FAILED, started, reason=f"edit target does not exist: {action.path}")
        self._validate_preimages(action, [destination])
        before = destination.read_text(encoding="utf-8")
        try:
            after = apply_edit_blocks(before, action.edits)
        except ValueError as exc:
            return self._result(action, ActionStatus.FAILED, started, reason=str(exc))
        if after == before:
            return self._result(action, ActionStatus.FAILED, started, reason="edit made no change")
        destination.write_text(after, encoding="utf-8")
        return self._result(
            action,
            ActionStatus.OK,
            started,
            returncode=0,
            changed_paths=[str(destination.relative_to(self.workspace))],
        )

    @staticmethod
    def _result(
        action: Action,
        status: ActionStatus,
        started: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        changed_paths: list[str] | None = None,
        reason: str = "",
    ) -> ActionResult:
        return ActionResult(
            action_id=action.id,
            action_hash=hashlib.sha256(
                json.dumps(
                    {
                        key: value
                        for key, value in action.model_dump(mode="json").items()
                        if key != "id"
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            status=status,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            changed_paths=changed_paths or [],
            started_at=started,
            finished_at=utc_now(),
            reason=reason,
        )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value
