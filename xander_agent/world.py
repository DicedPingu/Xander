"""XanderWorld: where Xander's own projects live, and the `.folder` map he
leaves in every place he works.

Projects are numbered folders under ``SPQR/XanderWorld``: ``000 - first
thing``, ``001 - next thing``. Each starts with a ``.folder`` — a short,
human-readable map of what the folder is and what is in it. The same map is
written once into any workspace he finishes a mission in, and never
overwritten: the operator may have edited it.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .models import utc_now

_NUMBERED = re.compile(r"^(?P<number>\d{3}) - (?P<name>.+)$")
_SKIP = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".vexp"}


def world_dir() -> Path:
    return Path.home() / "SPQR" / "XanderWorld"


def _next_number(root: Path) -> int:
    highest = -1
    if root.is_dir():
        for child in root.iterdir():
            match = _NUMBERED.match(child.name)
            if child.is_dir() and match:
                highest = max(highest, int(match.group("number")))
    return highest + 1


def _slug(name: str) -> str:
    cleaned = " ".join(re.sub(r"[^\w\s.-]", "", name).split())
    return cleaned[:60] or "untitled"


def create_project(name: str, goal: str = "", root: Path | None = None) -> Path:
    """Make ``NNN - name`` under XanderWorld with a starting ``.folder``."""

    root = (root or world_dir()).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if not (root / ".folder").exists():
        (root / ".folder").write_text(
            "XANDERWORLD — where Xander creates his own projects.\n"
            "One folder per project, numbered in the order they were started: `000 - name`, `001 - name`…\n"
            "Each has a .folder saying what it is. Xander opens one with /cd, or starts one with /new-project.\n",
            encoding="utf-8",
        )
    title = _slug(name)
    project = root / f"{_next_number(root):03d} - {title}"
    project.mkdir()
    lines = [
        f"{title.upper()} — a Xander project, started {datetime.now().strftime('%Y-%m-%d')}.",
        f"Goal: {goal.strip() or '(say what it is for — /goal <goal>)'}",
        "",
        ".folder           this map — keep it current when the shape of the project changes",
        "",
        "Nothing else yet.",
    ]
    (project / ".folder").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return project


def _describe(path: Path) -> str:
    if path.is_dir():
        try:
            count = sum(1 for _ in path.iterdir())
        except OSError:
            count = 0
        return f"{path.name}/  ({count} item{'s' if count != 1 else ''})"
    size = path.stat().st_size if path.exists() else 0
    unit = f"{size} B" if size < 1024 else f"{size / 1024:.0f} KB" if size < 1024**2 else f"{size / 1024**2:.1f} MB"
    first = ""
    if path.suffix in {".py", ".md", ".txt", ".sh", ".toml", ".json", ".yaml", ".yml"} and size < 200_000:
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:8]:
                text = line.strip().strip('"#/*- ').strip()
                if text and not text.startswith(("import ", "from ", "#!", "[", "{")):
                    first = text[:70]
                    break
        except OSError:
            first = ""
    return f"{path.name}  ({unit})" + (f"  — {first}" if first else "")


def folder_map_text(workspace: Path, purpose: str = "") -> str:
    workspace = workspace.expanduser().resolve()
    entries = sorted(
        (p for p in workspace.iterdir() if p.name not in _SKIP and p.name != ".folder"),
        key=lambda p: (not p.is_dir(), p.name.lower()),
    )
    lines = [
        f"{workspace.name.upper()} — {purpose.strip() or 'a folder Xander worked in'}.",
        f"Map written by Xander on {datetime.now().strftime('%Y-%m-%d')}; edit freely, he will not overwrite it.",
        "",
    ]
    for entry in entries[:80]:
        lines.append(_describe(entry))
        if entry.is_dir() and not entry.name.startswith("."):
            try:
                children = sorted(entry.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except OSError:
                children = []
            for child in children[:12]:
                if child.name in _SKIP:
                    continue
                lines.append("  " + _describe(child))
            if len(children) > 12:
                lines.append(f"  … {len(children) - 12} more")
    if len(entries) > 80:
        lines.append(f"… {len(entries) - 80} more")
    if not entries:
        lines.append("(empty)")
    return "\n".join(lines) + "\n"


def ensure_folder_map(workspace: Path, purpose: str = "") -> Path | None:
    """Write ``.folder`` into a workspace once. Existing maps are the operator's."""

    workspace = workspace.expanduser().resolve()
    target = workspace / ".folder"
    if target.exists():
        return None
    try:
        target.write_text(folder_map_text(workspace, purpose), encoding="utf-8")
    except OSError:
        return None
    return target


def project_record(project: Path, goal: str) -> None:
    """Append the goal to the project's .folder so the map says why it exists."""

    target = project / ".folder"
    if not target.exists():
        return
    body = target.read_text(encoding="utf-8")
    if goal and goal not in body:
        body = body.replace("Goal: (say what it is for — /goal <goal>)", f"Goal: {goal}", 1)
        body += f"\n{utc_now()[:10]}: goal — {goal}\n"
        target.write_text(body, encoding="utf-8")
