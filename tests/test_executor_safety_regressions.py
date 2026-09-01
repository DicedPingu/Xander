import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from xander_agent.executor import ActionExecutor
from xander_agent.models import Action, ActionKind, ActionStatus, Risk
from xander_agent.policy import approval_reason, classify_risk, is_setup_action, snapshot_workspace


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["apt-get", "-y", "install", "example"], id="apt-option-before-install"),
        pytest.param(["gh", "--repo", "owner/project", "pr", "create"], id="gh-global-option"),
        pytest.param(["uv", "run", "--", "python", "-c", "print('not run')"], id="uv-python-wrapper"),
    ],
)
def test_wrapped_or_reordered_mutations_are_high_risk_and_require_approval(
    tmp_path: Path, argv: list[str]
) -> None:
    snapshot = snapshot_workspace(tmp_path)
    action = Action(kind=ActionKind.COMMAND, argv=argv, expected="must require approval")

    assert classify_risk(action) == Risk.HIGH
    assert approval_reason(action, tmp_path, snapshot, "full-auto", []) is not None
    assert ActionExecutor(tmp_path, snapshot).run(action).status == ActionStatus.BLOCKED


@pytest.mark.parametrize(
    "argv",
    [
        ["sudo", "apt-get", "update"],
        ["python3", "-m", "pip", "install", "example"],
        ["cargo", "install", "tool"],
        ["npm", "upgrade", "example"],
    ],
)
def test_common_package_manager_setup_is_detected_without_allowing_removal(argv: list[str]) -> None:
    action = Action(kind=ActionKind.COMMAND, argv=argv, expected="prepare a toolchain")

    assert classify_risk(action) == Risk.HIGH
    assert is_setup_action(action)


@pytest.mark.parametrize(
    "argv",
    [["apt-get", "remove", "example"], ["pip", "uninstall", "example"], ["cargo", "uninstall", "tool"]],
)
def test_package_removal_is_not_covered_by_setup_allowance(argv: list[str]) -> None:
    assert not is_setup_action(Action(kind=ActionKind.COMMAND, argv=argv, expected="remove a package"))


def test_public_git_clone_is_a_local_workspace_mutation_not_an_external_write(
    tmp_path: Path,
) -> None:
    snapshot = snapshot_workspace(tmp_path)
    action = Action(
        kind=ActionKind.COMMAND,
        argv=["git", "clone", "https://example.com/public.git"],
        expected="clone a public repository into the workspace",
    )

    assert classify_risk(action) == Risk.MEDIUM
    assert approval_reason(action, tmp_path, snapshot, "full-auto", []) is None


def test_container_execution_requires_isolated_runtime_review(tmp_path: Path) -> None:
    action = Action(
        kind=ActionKind.COMMAND,
        argv=["docker", "run", "--rm", "image", "pytest"],
        expected="run tests in a container",
    )

    reason = approval_reason(action, tmp_path, snapshot_workspace(tmp_path), "full-auto", [])

    assert classify_risk(action) == Risk.MEDIUM
    assert reason is not None
    assert "isolated runtime flags" in reason


def test_privileged_container_is_rejected_as_escape_boundary(tmp_path: Path) -> None:
    action = Action(
        kind=ActionKind.COMMAND,
        argv=["docker", "run", "--privileged", "image", "sh"],
        expected="must not weaken host isolation",
    )

    reason = approval_reason(action, tmp_path, snapshot_workspace(tmp_path), "full-auto", [])

    assert reason is not None
    assert "host namespace" in reason


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "clone", "https://example.com/public.git", "../outside"],
        ["git", "clone", "https://example.com/public.git", "/tmp/outside"],
        [
            "git",
            "clone",
            "--separate-git-dir=/tmp/outside-git",
            "https://example.com/public.git",
            "inside",
        ],
        ["git", "-C", "../outside", "clone", "https://example.com/public.git"],
    ],
)
def test_git_clone_cannot_escape_or_use_path_writing_options(
    tmp_path: Path, argv: list[str]
) -> None:
    action = Action(kind=ActionKind.COMMAND, argv=argv, expected="unsafe clone")

    assert approval_reason(
        action,
        tmp_path,
        snapshot_workspace(tmp_path),
        "full-auto",
        [],
    ) is not None


@pytest.mark.parametrize(
    "action",
    [
        Action(
            kind=ActionKind.COMMAND,
            argv=["env", "git", "clone", "https://example.com/public.git", "../outside"],
            expected="wrapped unsafe clone",
        ),
        Action(
            kind=ActionKind.PIPELINE,
            pipeline=[
                ["printf", "unused"],
                ["git", "clone", "https://example.com/public.git", "../outside"],
            ],
            expected="pipelined unsafe clone",
        ),
    ],
)
def test_wrapped_and_pipelined_git_clone_cannot_bypass_containment(
    tmp_path: Path, action: Action
) -> None:
    assert approval_reason(
        action,
        tmp_path,
        snapshot_workspace(tmp_path),
        "full-auto",
        [],
    ) is not None


def test_proposal_inspect_blocks_python_writes_but_allows_safe_inspection(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    executor = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="proposal")
    destination = tmp_path / "must-not-exist.txt"

    write = executor.run(
        Action(
            kind=ActionKind.INSPECT,
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path; Path('must-not-exist.txt').write_text('mutated')",
            ],
            expected="a mislabeled mutating inspection",
        )
    )
    safe_results = [
        executor.run(
            Action(kind=ActionKind.INSPECT, argv=["rg", "--version"], expected="inspect ripgrep")
        ),
        executor.run(
            Action(kind=ActionKind.INSPECT, argv=["git", "status", "--short"], expected="inspect git status")
        ),
    ]

    assert write.status == ActionStatus.BLOCKED
    assert not destination.exists()
    assert [result.status for result in safe_results] == [ActionStatus.OK, ActionStatus.OK]


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _reap_process(pid: int) -> None:
    if _process_exists(pid):
        os.kill(pid, signal.SIGKILL)
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass


def test_pipeline_timeout_terminates_and_reaps_every_child(tmp_path: Path) -> None:
    pid_paths = [tmp_path / "first.pid", tmp_path / "second.pid"]
    sleeper = (
        "from pathlib import Path; import os, sys, time; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    action = Action(
        kind=ActionKind.PIPELINE,
        pipeline=[
            [sys.executable, "-c", sleeper, str(pid_paths[0])],
            [sys.executable, "-c", sleeper, str(pid_paths[1])],
        ],
        expected="exercise pipeline timeout cleanup",
    )
    executor = ActionExecutor(
        tmp_path,
        snapshot_workspace(tmp_path),
        approve=lambda _action, _reason: True,
        default_timeout=1,
    )
    pids: list[int] = []

    try:
        result = executor.run(action)
        assert all(path.exists() for path in pid_paths)
        pids = [int(path.read_text(encoding="utf-8")) for path in pid_paths]
        deadline = time.monotonic() + 2
        while any(_process_exists(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.05)

        assert result.status == ActionStatus.FAILED
        assert result.returncode == 124
        assert not [pid for pid in pids if _process_exists(pid)]
    finally:
        for pid in pids:
            _reap_process(pid)


def test_command_timeout_cleans_up_a_child_after_its_session_leader_exits(tmp_path: Path) -> None:
    pid_path = tmp_path / "detached-child.pid"
    spawn_and_exit = (
        "from pathlib import Path; import subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "Path(sys.argv[1]).write_text(str(child.pid))"
    )
    action = Action(
        kind=ActionKind.COMMAND,
        argv=[sys.executable, "-c", spawn_and_exit, str(pid_path)],
        expected="bound a child that inherits the session pipes",
    )
    executor = ActionExecutor(
        tmp_path,
        snapshot_workspace(tmp_path),
        approve=lambda _action, _reason: True,
        default_timeout=1,
    )
    child_pid: int | None = None

    started = time.monotonic()
    try:
        result = executor.run(action)
        elapsed = time.monotonic() - started
        assert pid_path.exists()
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while _process_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)

        assert result.status == ActionStatus.FAILED
        assert result.returncode == 124
        assert elapsed < 5
        assert not _process_exists(child_pid)
    finally:
        if child_pid is not None:
            _reap_process(child_pid)
