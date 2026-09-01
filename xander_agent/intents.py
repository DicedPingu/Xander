"""Deterministic intent routing that runs before any model call.

Session commands (workspace changes, mode switches) and clearly advisory
questions are recognized here and never become structured coding plans.
That mismatch is exactly what sank two real orders: "change the selected
folder …" spent four model attempts failing to become a coding plan, and
"just write suggestions …" could never satisfy implement-mode acceptance
checks. Text the operator types is data; this module decides which door
it goes through, and it must stay cheap, offline, and predictable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# The operator-facing mode wheel (shift+tab in the TUI).
MODES: tuple[str, ...] = ("ask", "plan", "build", "yolo")

MODE_HELP: dict[str, str] = {
    "ask": "answer and advise in prose — no files change",
    "plan": "research and propose — nothing is applied",
    "build": "implement; risky actions ask you first",
    "yolo": "implement; workspace-level risk is auto-approved",
}

# Engine request parameters implied by each mode: (engine_mode, autonomy).
# None autonomy defers to the variant profile.
MODE_REQUESTS: dict[str, tuple[str, str | None]] = {
    "ask": ("answer", None),
    "plan": ("plan", "proposal-only"),
    "build": ("implement", None),
    "yolo": ("implement", "full-auto"),
}


@dataclass(frozen=True)
class Intent:
    kind: str  # "chdir" | "mode" | "advice" | "order" | "help" | "feedback" | "selfwork"
    argument: str = ""
    create: bool = False


_CD_COMMAND = re.compile(r"^\s*/?cd\s+(?P<path>\S.*?)\s*$", re.IGNORECASE)
_CD_PHRASE = re.compile(
    r"\b(?:change|switch|set|move|rebind|point)\b.{0,60}?"
    r"\b(?:work(?:ing)?\s*(?:folder|dir(?:ectory)?)|workspace|selected\s+folder|current\s+folder)\b"
    r".{0,60}?\b(?:to|into|at)\b\s*(?P<path>.+?)[.!\s]*$",
    re.IGNORECASE | re.DOTALL,
)
_CALLED = re.compile(r"\b(?:called|named)\s+[\"'`]?(?P<name>[\w.\\/-]+)", re.IGNORECASE)
_CREATE_HINT = re.compile(r"\b(?:called|named|new|create|make)\b", re.IGNORECASE)
_MODE_COMMAND = re.compile(r"^\s*/mode(?:\s+(?P<mode>[a-z-]+))?\s*$", re.IGNORECASE)
_HELP_COMMAND = re.compile(r"^\s*/(?:help|how)\s*$", re.IGNORECASE)

# Standing likes/dislikes about how Xander works or writes. These are
# preferences to remember, not work orders — even when typed mid-run.
_FEEDBACK_LEAD = re.compile(
    r"^\s*(?:i\s+(?:really\s+)?(?:do\s*n[o']t\s+like|dislike|hate|like|love|prefer)|"
    r"from\s+now\s+on|in\s+the\s+future|going\s+forward|"
    r"always|never|stop\s+(?:doing|using|writing|adding)|"
    r"please\s+(?:always|never|stop)|less\s+of|more\s+of)\b",
    re.IGNORECASE,
)
_ADVICE_LEAD = re.compile(
    r"^\s*(?:give me|list|suggest|recommend|brainstorm|what|which|how|why|when|where|"
    r"explain|describe|summarize|compare|review|tell me|should|would|could|do you|can you|is it|are there)\b",
    re.IGNORECASE,
)
_ADVICE_ANY = re.compile(
    r"\b(?:just\s+(?:write|give|list)\s+(?:the\s+)?suggestions?|suggestions?\s+only|"
    r"ideas?\s+only|only\s+ideas?|alternatives|no\s+code\s+changes?|"
    r"don'?t\s+(?:change|edit|touch|modify)\s+(?:any\s+)?(?:files?|code)|"
    r"for\s+now,?\s+just\s+(?:talk|answer|explain))\b",
    re.IGNORECASE,
)
_CONVERSATION_LINE = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|good morning|good evening|"
    r"let'?s talk|can we talk|i wonder|i'?m curious|what do you think)\b",
    re.IGNORECASE,
)
_SELF_WORK = re.compile(
    r"\b(?:improve|upgrade|fix|clean|optimi[sz]e|refactor)\s+yourself\b|"
    r"\bresearch\s+your\s+soul\b|\bmake\s+yourself\s+more\s+capable\b|"
    r"\b(?:improve|upgrade|fix|refactor)\s+xander(?:'s)?\b",
    re.IGNORECASE,
)
# Verbs that flip an advisory-sounding line back into real work when they
# clearly target an artifact ("how about you implement the parser in lib/").
_MUTATION_ORDER = re.compile(
    r"\b(?:implement|create|write|build|fix|refactor|apply|patch|install|scaffold|"
    r"add|delete|remove|rename|generate|migrate|wire)\b"
    r".{0,80}?\b(?:file|folder|directory|test|module|class|function|script|project|repo|package|patch)\b",
    re.IGNORECASE,
)


def parse_intent(text: str) -> Intent:
    """Classify one operator line. Falls through to ``order`` on any doubt."""

    stripped = text.strip()
    if not stripped:
        return Intent(kind="order", argument="")
    if _HELP_COMMAND.match(stripped):
        return Intent(kind="help")
    mode_match = _MODE_COMMAND.match(stripped)
    if mode_match:
        return Intent(kind="mode", argument=(mode_match.group("mode") or "").lower())
    cd_match = _CD_COMMAND.match(stripped)
    if cd_match:
        return Intent(kind="chdir", argument=cd_match.group("path"), create=False)
    phrase_match = _CD_PHRASE.search(stripped)
    if phrase_match:
        return Intent(
            kind="chdir",
            argument=phrase_match.group("path"),
            create=bool(_CREATE_HINT.search(stripped)),
        )
    if _FEEDBACK_LEAD.match(stripped) and not _MUTATION_ORDER.search(stripped):
        return Intent(kind="feedback", argument=stripped)
    if _SELF_WORK.search(stripped):
        return Intent(kind="selfwork", argument=stripped)
    advisory = bool(
        stripped.endswith("?")
        or _ADVICE_LEAD.match(stripped)
        or _ADVICE_ANY.search(stripped)
        or _CONVERSATION_LINE.match(stripped)
    )
    if advisory and not _MUTATION_ORDER.search(stripped):
        return Intent(kind="advice", argument=stripped)
    return Intent(kind="order", argument=stripped)


def resolve_target(raw: str, workspace: Path) -> Path:
    """Turn a chdir argument — possibly a prose fragment — into one path.

    "subfolder inside the Xander dir called testing" resolves to
    ``workspace / "testing"``: a trailing called/named name wins over the
    prose around it, because the name is the only part the operator chose.
    """

    candidate = raw.strip().strip("\"'`")
    named = _CALLED.search(candidate)
    if named and (" " in candidate or candidate.lower().startswith(("a ", "the "))):
        candidate = named.group("name")
    elif " " in candidate:
        # Prose without a called/named marker: keep the last path-looking token.
        tokens = [token for token in re.findall(r"[\w.~\\/-]+", candidate) if "/" in token or token.startswith("~")]
        candidate = tokens[-1] if tokens else candidate.split()[-1]
    path = Path(candidate).expanduser()
    return path if path.is_absolute() else (workspace / path)
