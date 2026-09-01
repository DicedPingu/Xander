from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

_TASK_ID_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def project_record_dir(workspace: Path) -> Path | None:
    try:
        from .paths import projects_dir

        root = projects_dir().expanduser().resolve(strict=False)
        target = workspace.expanduser().resolve(strict=False)
    except OSError:
        return None
    if target == root:
        return None
    try:
        target.relative_to(root)
        return target
    except ValueError:
        label = re.sub(r"[^A-Za-z0-9._-]+", "-", target.name).strip("-._")[:48] or "workspace"
        digest = hashlib.sha256(str(target).encode("utf-8")).hexdigest()[:10]
        return root / "external" / f"{label}-{digest}"


def project_log_path(workspace: Path, task_id: str) -> Path | None:
    if not _TASK_ID_RE.fullmatch(task_id):
        return None
    record_dir = project_record_dir(workspace)
    if record_dir is None:
        return None
    return record_dir / "logs" / f"{task_id}.md"


def append_project_event(workspace: Path, task_id: str, body: str) -> Path | None:
    destination = project_log_path(workspace, task_id)
    if destination is None:
        return None
    clean = _CONTROL_RE.sub(" ", str(body))
    clean = " ".join(clean.split())[:1_200]
    if not clean:
        return None
    stamp = datetime.now().isoformat(timespec="seconds")
    try:
        record_dir = destination.parent.parent
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        project_file = record_dir / "PROJECT.md"
        if not project_file.exists():
            workspace_text = str(workspace.expanduser().resolve(strict=False)).replace("`", "'")
            workspace_text = _CONTROL_RE.sub(" ", workspace_text)
            try:
                with project_file.open("x", encoding="utf-8") as handle:
                    handle.write(
                        "# Xander workspace record\n\n"
                        f"Workspace: {workspace_text}\n\n"
                        "Requested source or configuration changes stay in that explicitly "
                        "opened workspace; agent records and detailed logs stay here.\n"
                    )
                os.chmod(project_file, 0o600)
            except FileExistsError:
                pass
        if not destination.exists():
            destination.write_text(
                f"# Xander task {task_id}\n\n"
                f"Workspace: `{workspace.expanduser().resolve(strict=False)}`\n\n"
                "## Activity\n",
                encoding="utf-8",
            )
            os.chmod(destination, 0o600)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(f"- `{stamp}` {clean}\n")
        return destination
    except OSError:
        return None
