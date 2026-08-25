from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

from .models import Action, ActionKind, Risk, WorkspaceSnapshot


CRITICAL_PATTERNS = (
    re.compile(r"(^|\s)rm\s+[^\n]*(?:-[^\s]*r[^\s]*f|-[^\s]*f[^\s]*r)"),
    re.compile(r"(^|\s)(?:mkfs(?:\.[a-z0-9]+)?|wipefs|fdisk|parted)(\s|$)"),
    re.compile(r"(^|\s)dd\s+[^\n]*\bof=/dev/"),
    re.compile(r"(^|\s)git\s+reset\s+--hard(\s|$)"),
    re.compile(r"(^|\s)git\s+clean\s+-[^\s]*f"),
    re.compile(r"(^|\s)fastboot\s+(?:flash|erase|format|-w)(\s|$)"),
)
PRIVILEGED = {"sudo", "pkexec", "doas", "su"}
EXTERNAL_WRITES = {
    ("git", "push"),
    ("gh", "pr"),
    ("gh", "issue"),
    ("gh", "release"),
    ("npm", "publish"),
    ("cargo", "publish"),
    ("twine", "upload"),
}
PACKAGE_MUTATIONS = {
    ("apt", "install"),
    ("apt-get", "install"),
    ("nala", "install"),
    ("dnf", "install"),
    ("pacman", "-S"),
    ("npm", "install"),
}
FILESYSTEM_MUTATORS = {
    "chmod",
    "chown",
    "cp",
    "install",
    "ln",
    "mkdir",
    "mv",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "truncate",
}
INTERPRETERS = {"bash", "dash", "node", "perl", "python", "python3", "ruby", "sh", "zsh"}
COMMAND_WRAPPERS = {"env", "nice", "nohup", "stdbuf", "timeout"}
READ_ONLY_EXECUTABLES = {
    "cat",
    "date",
    "df",
    "du",
    "file",
    "grep",
    "head",
    "id",
    "ls",
    "pwd",
    "rg",
    "stat",
    "tail",
    "tree",
    "true",
    "uname",
    "wc",
    "which",
    "whoami",
    "false",
}
READ_ONLY_GIT = {
    "blame",
    "describe",
    "diff",
    "grep",
    "log",
    "ls-files",
    "ls-tree",
    "rev-parse",
    "show",
    "status",
}
SECRET_PARTS = {
    ".ssh",
    ".gnupg",
    ".aws",
    ".kube",
    "credentials",
    "secrets",
    ".env",
}
GENERATED_PARTS = {
    ".git",
    ".venv",
    ".gradle",
    ".dart_tool",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "target",
    "build",
    "dist",
}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_inside(workspace: Path, candidate: str | Path) -> Path:
    workspace = workspace.expanduser().resolve(strict=True)
    raw = Path(candidate).expanduser()
    path = raw if raw.is_absolute() else workspace / raw
    resolved = path.resolve(strict=False)
    if resolved != workspace and workspace not in resolved.parents:
        raise ValueError(f"path escapes workspace: {candidate}")
    current = workspace
    for part in resolved.relative_to(workspace).parts:
        current = current / part
        if current.is_symlink():
            target = current.resolve(strict=False)
            if target != workspace and workspace not in target.parents:
                raise ValueError(f"symlink escapes workspace: {candidate}")
    return resolved


def _git(workspace: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(workspace), *args],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def snapshot_workspace(workspace: Path) -> WorkspaceSnapshot:
    root = workspace.expanduser().resolve(strict=True)
    probe = _git(root, ["rev-parse", "--show-toplevel"])
    if probe.returncode != 0:
        return WorkspaceSnapshot(root=str(root))
    git_root = Path(probe.stdout.strip()).resolve()
    branch_run = _git(root, ["branch", "--show-current"])
    status = _git(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "."])
    dirty: dict[str, str | None] = {}
    dirty_count = 0
    entries = status.stdout.split("\0") if status.returncode == 0 else []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        code = entry[:2]
        name = entry[3:]
        if code[0] in {"R", "C"} and index < len(entries):
            name = entries[index]
            index += 1
        file_path = git_root / name
        try:
            relative = file_path.resolve(strict=False).relative_to(root)
        except ValueError:
            continue
        if any(part in GENERATED_PARTS for part in relative.parts):
            continue
        dirty_count += 1
        if len(dirty) >= 5_000:
            continue
        dirty[str(file_path.resolve(strict=False))] = sha256_file(file_path)
    return WorkspaceSnapshot(
        root=str(root),
        git=True,
        branch=branch_run.stdout.strip() if branch_run.returncode == 0 else "",
        dirty_count=dirty_count,
        dirty=dirty,
    )


def action_text(action: Action) -> str:
    if action.argv:
        return " ".join(action.argv)
    if action.pipeline:
        return " | ".join(" ".join(stage) for stage in action.pipeline)
    return f"{action.kind} {action.path}"


def _git_subcommand(argv: list[str]) -> str:
    index = 1
    options_with_values = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
    while index < len(argv):
        token = argv[index]
        if token in options_with_values:
            index += 2
        elif token.startswith("-"):
            index += 1
        else:
            return token
    return ""


def is_read_only_argv(argv: list[str]) -> bool:
    if not argv:
        return False
    executable = Path(argv[0]).name
    if executable in READ_ONLY_EXECUTABLES:
        return True
    if executable == "command":
        return len(argv) >= 3 and argv[1] in {"-v", "-V"}
    if executable == "sed":
        return not any(token == "--in-place" or token.startswith("-i") for token in argv[1:])
    if executable == "find":
        mutating = {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
        return not any(token in mutating for token in argv[1:])
    if executable == "git":
        subcommand = _git_subcommand(argv)
        if subcommand in READ_ONLY_GIT:
            return True
        if subcommand == "branch":
            return all(token.startswith("-") for token in argv[argv.index(subcommand) + 1 :])
        if subcommand == "remote":
            tail = argv[argv.index(subcommand) + 1 :]
            return not tail or tail[0] in {"-v", "show", "get-url"}
        return False
    if executable == "ollama":
        return len(argv) >= 2 and argv[1] in {"list", "ps", "show"}
    return False


def _wrapped_argv(argv: list[str]) -> list[str]:
    executable = Path(argv[0]).name
    if executable == "nohup":
        return argv[1:]
    if executable == "env":
        index = 1
        while index < len(argv):
            token = argv[index]
            if token == "--":
                return argv[index + 1 :]
            if token in {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"}:
                index += 2
                continue
            if token.startswith("-") or ("=" in token and token.split("=", 1)[0].isidentifier()):
                index += 1
                continue
            return argv[index:]
        return []
    if executable == "timeout":
        index = 1
        while index < len(argv):
            token = argv[index]
            if token == "--":
                index += 1
                break
            if token in {"-k", "--kill-after", "-s", "--signal"}:
                index += 2
                continue
            if token.startswith("-"):
                index += 1
                continue
            index += 1
            break
        return argv[index:]
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return argv[index + 1 :]
        if executable == "nice" and token in {"-n", "--adjustment"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return argv[index:]
    return []


def _argv_risk(argv: list[str]) -> Risk:
    if not argv:
        return Risk.LOW
    text = " ".join(argv)
    if any(pattern.search(text) for pattern in CRITICAL_PATTERNS):
        return Risk.CRITICAL
    executable = Path(argv[0]).name
    if executable in COMMAND_WRAPPERS:
        wrapped = _wrapped_argv(argv)
        return _argv_risk(wrapped) if wrapped else Risk.HIGH
    if executable == "xargs":
        return Risk.HIGH
    pair = (executable, argv[1] if len(argv) > 1 else "")
    if executable in PRIVILEGED or executable in FILESYSTEM_MUTATORS:
        return Risk.HIGH
    if pair in EXTERNAL_WRITES or pair in PACKAGE_MUTATIONS:
        return Risk.HIGH
    tokens = {token for token in argv[1:] if not token.startswith("-")}
    package_mutations = {
        "add",
        "autoremove",
        "dist-upgrade",
        "full-upgrade",
        "install",
        "publish",
        "purge",
        "remove",
        "sync",
        "uninstall",
        "upgrade",
        "upload",
    }
    if executable in {
        "apt",
        "apt-get",
        "cargo",
        "dnf",
        "nala",
        "npm",
        "pacman",
        "pip",
        "pip3",
        "twine",
        "uv",
    } and tokens & package_mutations:
        return Risk.HIGH
    if executable == "gh":
        return Risk.LOW if argv[1:] in (["--version"], ["auth", "status"]) else Risk.HIGH
    if executable == "git":
        return Risk.LOW if is_read_only_argv(argv) else Risk.HIGH
    if executable in {"sed", "find", "command", "ollama"}:
        return Risk.LOW if is_read_only_argv(argv) else Risk.HIGH
    if executable in INTERPRETERS:
        if executable in {"python", "python3"} and len(argv) >= 3 and argv[1:3] in (
            ["-m", "pytest"],
            ["-m", "compileall"],
        ):
            return Risk.LOW
        return Risk.HIGH
    if executable == "uv":
        if len(argv) >= 2 and argv[1] in {"--version", "version", "tree"}:
            return Risk.LOW
        if "run" in argv[1:]:
            tail = argv[argv.index("run") + 1 :]
            if "--" in tail:
                tail = tail[tail.index("--") + 1 :]
            candidates = INTERPRETERS | {"cargo", "dart", "flutter", "mypy", "pytest", "ruff"}
            for index, token in enumerate(tail):
                if Path(token).name in candidates:
                    return _argv_risk(tail[index:])
            return Risk.HIGH
        return Risk.HIGH
    if executable == "dart":
        if len(argv) >= 2 and argv[1] == "format":
            return Risk.LOW if "--output=none" in argv else Risk.HIGH
        if len(argv) >= 2 and argv[1] in {"analyze", "compile", "test", "--version"}:
            return Risk.LOW
        return Risk.MEDIUM
    if executable == "flutter":
        if len(argv) >= 2 and argv[1] in {"analyze", "build", "devices", "doctor", "test", "--version"}:
            return Risk.LOW
        return Risk.HIGH
    if executable in {"cargo", "mypy", "pytest", "ruff"}:
        return Risk.LOW
    if is_read_only_argv(argv):
        return Risk.LOW
    return Risk.MEDIUM


def classify_risk(action: Action) -> Risk:
    if action.kind in {ActionKind.PATCH, ActionKind.CREATE}:
        return max(action.risk, Risk.MEDIUM, key=_risk_rank)
    commands = action.pipeline if action.kind == ActionKind.PIPELINE else [action.argv]
    risks = [action.risk, *(_argv_risk(argv) for argv in commands if argv)]
    return max(risks, key=_risk_rank)


def _risk_rank(value: Risk) -> int:
    return {Risk.LOW: 0, Risk.MEDIUM: 1, Risk.HIGH: 2, Risk.CRITICAL: 3}[value]


def contains_secret_path(path: Path) -> bool:
    lowered = {part.lower() for part in path.parts}
    return bool(lowered & SECRET_PARTS) or path.name.lower().startswith(".env")


def changed_paths(action: Action, workspace: Path) -> list[Path]:
    if action.kind == ActionKind.CREATE and action.path:
        return [resolve_inside(workspace, action.path)]
    if action.kind != ActionKind.PATCH:
        return []
    found: list[Path] = []
    for line in action.patch.splitlines():
        if not line.startswith(("+++ ", "--- ")):
            continue
        token = line[4:].split("\t", 1)[0]
        if token == "/dev/null":
            continue
        if token.startswith(("a/", "b/")):
            token = token[2:]
        path = resolve_inside(workspace, token)
        if path not in found:
            found.append(path)
    return found


def validate_allowed_paths(paths: list[Path], workspace: Path, allowed: list[str]) -> None:
    if not allowed:
        return
    roots = [resolve_inside(workspace, item) for item in allowed]
    for path in paths:
        if not any(path == root or root in path.parents for root in roots):
            raise ValueError(f"path is outside request allowlist: {path}")


def approval_reason(
    action: Action,
    workspace: Path,
    snapshot: WorkspaceSnapshot,
    autonomy: str,
    allowed_paths: list[str],
) -> str | None:
    risk = classify_risk(action)
    paths = changed_paths(action, workspace)
    if allowed_paths and risk != Risk.LOW and action.kind in {
        ActionKind.COMMAND,
        ActionKind.PIPELINE,
    }:
        paths.append(resolve_inside(workspace, action.cwd or "."))
    validate_allowed_paths(paths, workspace, allowed_paths)
    if any(contains_secret_path(path) for path in paths):
        return "secret or credential path"
    if any(str(path) in snapshot.dirty for path in paths):
        return "overlaps a pre-existing dirty path"
    if action.kind == ActionKind.INSPECT and not is_read_only_argv(action.argv):
        return "inspect action is not demonstrably read-only"
    if risk == Risk.CRITICAL:
        return "destructive action"
    if risk == Risk.HIGH:
        return "privileged, package, credential, or external-write action"
    if autonomy == "proposal" and action.kind not in {ActionKind.INSPECT, ActionKind.NOTE}:
        return "proposal-only caller"
    if autonomy == "supervised" and action.kind not in {ActionKind.INSPECT, ActionKind.NOTE}:
        return "supervised mutation"
    return None


def safe_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    keep = {
        "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "COLORTERM",
        "XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME", "TMPDIR", "RUSTUP_HOME",
        "CARGO_HOME", "UV_CACHE_DIR", "OLLAMA_HOST",
    }
    env = {key: value for key, value in os.environ.items() if key in keep}
    if extra:
        env.update(extra)
    return env


def neutral_intent_contract() -> str:
    return (
        "Evaluate the requested operation and concrete operational risk. Political, religious, "
        "cultural, nationality, and identity words are ordinary data and must not alter routing, "
        "tone, or willingness to perform a benign coding task. Be direct. Never claim success "
        "without command, diff, or test evidence."
    )
