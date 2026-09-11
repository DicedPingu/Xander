"""Small concrete orders that need no mission ceremony.

"Create an empty file called done.txt" is one action and one sentence of
reply. It must not become a five-step adaptive loop, a research phase, a
delegation roster, a judge, and a lesson. This module recognizes such
orders offline, turns them into the same guarded ``Action`` objects the
engine already executes (so policy still applies), and supplies the one
line Xander says afterwards: "I created done.txt. What now?"

Everything here is deterministic. Anything it does not recognize falls
through to the ordinary planner untouched.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .models import Action, ActionKind

_LEAD = r"^\s*(?:(?:please|xander|ok(?:ay)?|now|then|just|go\s+ahead\s+and|can\s+you|could\s+you),?\s+)*"
_NAME = r"[\"'`]?(?P<name>[\w][\w.\- ]*?[\w.]|[\w])[\"'`]?"
_NAME2 = r"[\"'`]?(?P<name2>[\w][\w.\- ]*?[\w.]|[\w])[\"'`]?"
_WHERE = r"(?:\s+(?:in|inside|into|under|at)\s+(?:this|the\s+current|the)\s+(?:folder|directory|dir|workspace))?"
_SUBDIR = r"(?:\s+(?:in|inside|into|under)\s+[\"'`]?(?P<dir>[\w][\w./\-]*)[\"'`]?)?"
_TAIL = r"\s*[.!]*\s*$"

_CREATE_FILE = re.compile(
    _LEAD
    + r"(?:create|make|touch|add|new)\s+(?:an?\s+|the\s+)?(?:(?P<empty>empty|blank|new)\s+)?"
    + r"(?:text\s+)?file\s+(?:called|named|name)?\s*"
    + _NAME
    + _SUBDIR
    + _WHERE
    + r"(?:\s+(?:with|containing)\s+(?:the\s+)?(?:content|contents|text)?\s*[:]?\s*[\"'`](?P<content>.*?)[\"'`])?"
    + _TAIL,
    re.IGNORECASE | re.DOTALL,
)
_TOUCH = re.compile(_LEAD + r"touch\s+" + _NAME + _TAIL, re.IGNORECASE)
_CREATE_DIR = re.compile(
    _LEAD
    + r"(?:create|make|add|mkdir|new)\s+(?:an?\s+|the\s+)?(?:new\s+|empty\s+)?"
    + r"(?:folder|directory|dir|subfolder|sub-folder|subdirectory)\s+(?:called|named|name)?\s*"
    + _NAME
    + _SUBDIR
    + _WHERE
    + _TAIL,
    re.IGNORECASE,
)
_WRITE_TEXT = re.compile(
    _LEAD
    + r"(?:write|put|save)\s+[\"'`](?P<content>.*?)[\"'`]\s+(?:to|into|in)\s+(?:a\s+|the\s+)?(?:new\s+)?(?:file\s+)?(?:called|named)?\s*"
    + _NAME
    + _TAIL,
    re.IGNORECASE | re.DOTALL,
)
_DELETE = re.compile(
    _LEAD
    + r"(?:delete|remove|rm)\s+(?:the\s+)?(?:file|folder|directory|dir)?\s*(?:called|named)?\s*"
    + _NAME
    + _WHERE
    + _TAIL,
    re.IGNORECASE,
)
_RENAME = re.compile(
    _LEAD
    + r"(?:rename|move|mv)\s+(?:the\s+)?(?:file|folder|directory|dir)?\s*(?:called|named)?\s*"
    + _NAME
    + r"\s+(?:to|into|as|->)\s+"
    + _NAME2
    + _TAIL,
    re.IGNORECASE,
)
_RUN = re.compile(
    _LEAD + r"(?:run|execute|exec)\s*[:]?\s*[`\"'](?P<command>[^`\"']+)[`\"']" + _TAIL,
    re.IGNORECASE,
)


@dataclass
class QuickOrder:
    """A recognized small order: what to do and what to say afterwards."""

    kind: str
    actions: list[Action]
    reply: str
    paths: list[str] = field(default_factory=list)


def _clean(name: str) -> str:
    return name.strip().strip("\"'`").strip()


def _join(directory: str | None, name: str) -> str:
    name = _clean(name)
    if directory:
        return str(Path(_clean(directory)) / name)
    return name


def _safe_relative(path: str) -> bool:
    """Only plain workspace-relative names; absolute paths and `..` fall
    through to the full planner where the policy explains itself."""
    candidate = Path(path)
    return not candidate.is_absolute() and ".." not in candidate.parts and not path.startswith("~")


def parse_quick_order(goal: str, workspace: Path | None = None) -> QuickOrder | None:
    """Return a ``QuickOrder`` for one small, unambiguous file order, else ``None``.

    Multi-part sentences ("… and then …", "…, then …") are never quick:
    they deserve a plan. So are orders whose target escapes the workspace.
    """

    text = " ".join(goal.split())
    if not text or re.search(r"\b(?:and\s+then|then|after\s+that|also|as\s+well\s+as)\b", text, re.IGNORECASE):
        return None
    if len(text.split()) > 24:
        return None

    match = _CREATE_FILE.match(text) or _TOUCH.match(text)
    if match:
        groups = match.groupdict()
        path = _join(groups.get("dir"), groups["name"])
        content = groups.get("content") or ""
        if not _safe_relative(path):
            return None
        return QuickOrder(
            kind="create_file",
            actions=[
                Action(
                    kind=ActionKind.CREATE,
                    path=path,
                    content=content,
                    expected=f"{path} exists",
                )
            ],
            reply=f"I created {path}. What now?",
            paths=[path],
        )

    match = _WRITE_TEXT.match(text)
    if match:
        path = _clean(match.group("name"))
        if not _safe_relative(path):
            return None
        content = match.group("content")
        return QuickOrder(
            kind="write_file",
            actions=[Action(kind=ActionKind.CREATE, path=path, content=content, expected=f"{path} written")],
            reply=f"I wrote that to {path}. What now?",
            paths=[path],
        )

    match = _CREATE_DIR.match(text)
    if match:
        groups = match.groupdict()
        path = _join(groups.get("dir"), groups["name"])
        if not _safe_relative(path):
            return None
        return QuickOrder(
            kind="create_dir",
            actions=[Action(kind=ActionKind.COMMAND, argv=["mkdir", "-p", path], expected=f"{path}/ exists")],
            reply=f"I created the folder {path}/. What now?",
            paths=[path],
        )

    match = _RENAME.match(text)
    if match:
        source = _clean(match.group("name"))
        target = _clean(match.group("name2"))
        if not (_safe_relative(source) and _safe_relative(target)):
            return None
        if workspace is not None and not (workspace / source).exists():
            return None
        return QuickOrder(
            kind="rename",
            actions=[Action(kind=ActionKind.COMMAND, argv=["mv", source, target], expected=f"{target} exists")],
            reply=f"I renamed {source} to {target}. What now?",
            paths=[source, target],
        )

    match = _DELETE.match(text)
    if match:
        path = _clean(match.group("name"))
        if not _safe_relative(path):
            return None
        if workspace is not None:
            target = workspace / path
            if not target.exists():
                return None
            if target.is_dir() and any(target.iterdir()):
                # A non-empty folder is not a one-liner; let the planner spell it out.
                return None
        argv = ["rmdir", path] if workspace is not None and (workspace / path).is_dir() else ["rm", path]
        return QuickOrder(
            kind="delete",
            actions=[Action(kind=ActionKind.COMMAND, argv=argv, expected=f"{path} is gone")],
            reply=f"I deleted {path}. What now?",
            paths=[path],
        )

    match = _RUN.match(text)
    if match:
        try:
            argv = shlex.split(match.group("command"))
        except ValueError:
            return None
        if not argv or any(token in {"|", "&&", "||", ";", ">", ">>", "<"} for token in argv):
            return None
        return QuickOrder(
            kind="run",
            actions=[Action(kind=ActionKind.COMMAND, argv=argv, expected=f"`{' '.join(argv)}` ran")],
            reply=f"I ran `{' '.join(argv)}`. What now?",
        )
    return None
