"""Deterministic intent routing that runs before any model call.

Session commands (workspace changes, mode switches) and clearly advisory
questions are recognized here and never become structured coding plans.
That mismatch is exactly what sank two real orders: "change the selected
folder …" spent four model attempts failing to become a coding plan, and
"just write suggestions …" could never satisfy implement-mode acceptance
checks. Text the operator types is data; this module decides which door
it goes through, and it must stay cheap, offline, and predictable.

Two rules from the working agreement shape the doors:

* Anything beginning with ``/`` is a command. A known command is resolved
  here; an unknown one is explained, never handed to a shell or a model.
* Preparation must not silently become implementation. "This project is
  going to be about …" opens a discussion; "Create …" or "Do …" authorizes
  work; a line that is neither is held as a draft until the operator picks
  ``/discuss`` or ``/work``. The draft is retained, so the choice is one word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
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
class CommandSpec:
    """One slash command the composer understands.

    ``scope`` is ``core`` for commands with meaning outside the TUI (they
    get their own intent kind) and ``tui`` for view/lifecycle shortcuts
    that only the full-screen interface can act on.
    """

    name: str
    usage: str
    summary: str
    kind: str
    scope: str = "core"
    aliases: tuple[str, ...] = ()
    takes_argument: bool = False


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("/help", "/help", "list commands and keys", "help", aliases=("/how", "/commands")),
    CommandSpec("/mode", "/mode [ask|plan|build|yolo]", "show or switch the mode wheel", "mode", takes_argument=True),
    CommandSpec("/cd", "/cd <path>", "change the workspace folder", "chdir", takes_argument=True),
    CommandSpec(
        "/discuss",
        "/discuss [topic]",
        "talk it through — analysis, ideas, plans; nothing runs",
        "discuss",
        takes_argument=True,
    ),
    CommandSpec(
        "/work",
        "/work [goal]",
        "authorize work on the goal, the kept draft, or the newest open goal",
        "work",
        takes_argument=True,
    ),
    CommandSpec(
        "/research",
        "/research <URL or topic>",
        "read sources and report what was learned; the workspace is not modified",
        "research",
        takes_argument=True,
    ),
    CommandSpec(
        "/goal",
        "/goal [goal]",
        "store a goal for this workspace, or list the stored goals",
        "goal",
        takes_argument=True,
    ),
    CommandSpec("/addtodo", "/addtodo <task>", "pin a TODO item for this workspace", "todo", takes_argument=True),
    CommandSpec(
        "/new-project",
        "/new-project <name> [-- goal]",
        "start a numbered project folder in SPQR/XanderWorld and work there",
        "new_project",
        aliases=("/project", "/new"),
        takes_argument=True,
    ),
    CommandSpec(
        "/talk",
        "/talk <message>",
        "speak with the selected clone",
        "talk",
        aliases=("/ask",),
        takes_argument=True,
    ),
    # Interface-only shortcuts: everything prints into the one feed.
    CommandSpec("/history", "/history", "recent missions in this workspace", "command", scope="tui", aliases=("/missions",)),
    CommandSpec(
        "/show", "/show [id]", "one mission's evidence (default: the last)", "command", scope="tui",
        aliases=("/evidence", "/proof"), takes_argument=True,
    ),
    CommandSpec(
        "/todo", "/todo [task|run]", "list TODOs, pin one, or `run` them one by one", "command", scope="tui",
        aliases=("/todos",), takes_argument=True,
    ),
    CommandSpec("/pause", "/pause", "pause the running mission", "command", scope="tui"),
    CommandSpec("/resume", "/resume", "resume the paused mission", "command", scope="tui"),
    CommandSpec("/stop", "/stop", "contest the running mission", "command", scope="tui"),
    CommandSpec("/cancel", "/cancel", "cancel the running mission", "command", scope="tui"),
    CommandSpec("/fresh", "/fresh", "clear pending decisions and start fresh", "command", scope="tui", aliases=("/reset",)),
    CommandSpec("/clear", "/clear", "clear the feed", "command", scope="tui"),
    CommandSpec("/desktop", "/desktop", "take a local desktop screenshot", "command", scope="tui"),
    CommandSpec(
        "/set",
        "/set mode|authority|setup|variant|verbose <value>",
        "change one live value",
        "command",
        scope="tui",
        takes_argument=True,
    ),
    CommandSpec("/values", "/values", "show the live values", "command", scope="tui"),
    CommandSpec("/stats", "/stats", "the scoreboard — tasks, success rate, models, squad", "command", scope="tui"),
)

_COMMAND_INDEX: dict[str, CommandSpec] = {}
for _spec in COMMANDS:
    _COMMAND_INDEX[_spec.name] = _spec
    for _alias in _spec.aliases:
        _COMMAND_INDEX[_alias] = _spec


def command_spec(name: str) -> CommandSpec | None:
    return _COMMAND_INDEX.get(name.strip().casefold())


def command_help(scope: str | None = None) -> list[str]:
    """Readable ``usage — summary`` lines, core commands first."""

    return [
        f"{spec.usage} — {spec.summary}"
        for spec in COMMANDS
        if scope is None or spec.scope == scope
    ]


@dataclass(frozen=True)
class Intent:
    kind: str
    argument: str = ""
    create: bool = False
    command: str = ""
    authorized: bool = False
    reason: str = ""


_CD_COMMAND = re.compile(r"^\s*/?cd\s+(?P<path>\S.*?)\s*$", re.IGNORECASE)
# Only a direct "change/set the workspace to X" is a workspace change. The
# older, looser shape ("move … folder … into …") also matched "move the
# files in the current folder into subfolders" and tried to cd into
# "subfolders" — a real order mis-read as navigation.
_CD_PHRASE = re.compile(
    r"^\s*(?:(?:please|xander|ok(?:ay)?|now),?\s+)*"
    r"(?:change|switch|set|rebind|point|open)\s+(?:the\s+|my\s+|your\s+)?"
    r"(?:work(?:ing)?\s*(?:folder|dir(?:ectory)?)|workspace|selected\s+folder|current\s+folder|project\s+folder)"
    r"\s+(?:to|into|at)\s+(?P<path>\S.*?)[.!\s]*$",
    re.IGNORECASE | re.DOTALL,
)
# Venting, complaints, and remarks about Xander himself are conversation,
# never a script to write. "Quick fucking asking so much" became a Python
# file that printed the sentence; it should have been a reply.
_VENT = re.compile(
    r"^\s*(?:(?:quick|please|just|ffs|omg|ugh|wtf|jesus|god|man|dude),?\s+)*"
    r"(?:stop|quit|enough|why\s+(?:do|are|did)\s+you|you\s+(?:are|keep|always|never)|"
    r"that(?:'s| is| was)\s+(?:not|wrong|stupid|dumb|useless|bad)|"
    r"(?:this|that)\s+(?:is|was)\s+(?:not\s+)?what\s+i|no[,.!]|nope|wrong|"
    r"fucking|fuck|shit|damn|bullshit|retard|idiot|stupid|dumb|useless|"
    r"asking\s+(?:so\s+much|too\s+much|too\s+many))\b",
    re.IGNORECASE,
)
_CALLED = re.compile(r"\b(?:called|named)\s+[\"'`]?(?P<name>[\w.\\/-]+)", re.IGNORECASE)
_CREATE_HINT = re.compile(r"\b(?:called|named|new|create|make)\b", re.IGNORECASE)
_SLASH_LINE = re.compile(r"^\s*(?P<name>/\S*)(?:\s+(?P<argument>.*?))?\s*$", re.DOTALL)

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
# Exploratory framing: the operator is setting the scene, not giving an
# order. These open a discussion and keep the line as a draft.
_EXPLORATORY_LEAD = re.compile(
    r"^\s*(?:"
    r"(?:this|the|my|our)\s+(?:new\s+)?(?:project|app|tool|idea|plan|repo|repository|service|site|game)"
    r"\s+(?:is\s+going\s+to\s+be|will\s+be|is|would\s+be|should\s+be)\s+(?:about|for|a|an)\b|"
    r"i(?:'m|\s+am|'ve\s+been|\s+have\s+been|\s+was)\s+(?:thinking|considering|wondering|planning|imagining)\b|"
    r"i(?:'d|\s+would)\s+like\s+to\s+(?:discuss|talk|think|explore|brainstorm)\b|"
    r"i\s+want\s+to\s+(?:discuss|talk|think|explore|brainstorm)\b|"
    r"let'?s\s+(?:think|discuss|explore|brainstorm|consider|figure\s+out|plan|imagine)\b|"
    r"(?:the|my|our)\s+(?:idea|plan|thought|goal|vision|dream)\s+(?:is|was|would\s+be)\b|"
    r"idea\s*:|thought\s*:|"
    r"what\s+if\b|maybe\s+we\b|perhaps\s+we\b|we\s+could\b|we\s+might\b|"
    r"i\s+was\s+thinking\b|thinking\s+(?:about|of)\b|"
    r"imagine\b|picture\s+this\b|here'?s\s+the\s+(?:idea|context|situation|background)\b"
    r")",
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
# An imperative opening authorizes work: "Create …", "Do …", "Make …".
# Politeness and a name in front do not change that.
_AUTHORIZING_LEAD = re.compile(
    r"^\s*(?:(?:please|xander|ok(?:ay)?|now|then|go\s+ahead\s+and|just),?\s+)*"
    r"(?:create|do|make|build|implement|write|fix|add|run|install|refactor|remove|delete|rename|"
    r"generate|migrate|wire|update|change|set\s+up|setup|scaffold|patch|apply|convert|port|deploy|"
    r"test|replace|move|extract|split|merge|bump|upgrade|configure|enable|disable|turn|"
    r"execute|start\s+(?:working|building|implementing)|go\s+ahead|proceed|ship|finish|complete|"
    r"clean\s+up|optimi[sz]e|rewrite|reorgani[sz]e|restructure|document|translate|render|compile|"
    r"group|sort|organi[sz]e|tidy|arrange|watch|monitor|browse|navigate|visit|open|copy|download|"
    r"clone|touch|mkdir|put|save|print|echo|count|search|find|look\s+up|go\s+to|log\s+in|sign\s+in)\b",
    re.IGNORECASE,
)


def resolve_command(text: str) -> Intent | None:
    """Resolve a ``/``-prefixed line, or return ``None`` when it is not one.

    Every slash line resolves to *something*: a known command becomes its
    intent, an unknown one becomes ``unknown_command`` with an explanation
    (and the nearest known name when there is one). Slash lines never fall
    through to shell or model routing.
    """

    match = _SLASH_LINE.match(text)
    if not match:
        return None
    name = match.group("name").casefold()
    argument = (match.group("argument") or "").strip()
    spec = _COMMAND_INDEX.get(name)
    if spec is None:
        suggestions = get_close_matches(name, list(_COMMAND_INDEX), n=1, cutoff=0.6)
        hint = f" Did you mean {suggestions[0]}?" if suggestions else ""
        return Intent(
            kind="unknown_command",
            argument=argument,
            command=name,
            reason=f"{name} is not a Xander command; it was not run as shell.{hint} Type /help for the list.",
        )
    if spec.kind == "chdir":
        if not argument:
            return Intent(kind="unknown_command", command=spec.name, reason=f"Use {spec.usage}.")
        return Intent(kind="chdir", argument=argument, command=spec.name)
    if spec.kind == "mode":
        return Intent(kind="mode", argument=argument.casefold(), command=spec.name)
    authorized = spec.kind == "work"
    return Intent(kind=spec.kind, argument=argument, command=spec.name, authorized=authorized)


def parse_intent(text: str) -> Intent:
    """Classify one operator line.

    Order of doors: slash commands, workspace changes, standing feedback,
    explicit self-work, questions/advice, exploratory discussion, authorized
    orders. What is left is a ``draft``: kept, not run.
    """

    stripped = text.strip()
    if not stripped:
        return Intent(kind="draft", argument="", reason="nothing was typed")
    command = resolve_command(stripped)
    if command is not None:
        return command
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
    mutation = bool(_MUTATION_ORDER.search(stripped))
    if _FEEDBACK_LEAD.match(stripped) and not mutation:
        return Intent(kind="feedback", argument=stripped)
    if _VENT.search(stripped) and not mutation and not _AUTHORIZING_LEAD.match(stripped):
        return Intent(kind="advice", argument=stripped, reason="this reads as a remark to me, so I answer it")
    if _SELF_WORK.search(stripped):
        return Intent(kind="selfwork", argument=stripped, authorized=True)
    if _EXPLORATORY_LEAD.match(stripped) and not mutation:
        # Scene-setting beats the generic question lead: "what if we …" is a
        # discussion whose line is worth keeping as the draft.
        return Intent(
            kind="discuss",
            argument=stripped,
            reason="this reads as setting the scene, so it opens a discussion; /work authorizes it",
        )
    advisory = bool(
        stripped.endswith("?")
        or _ADVICE_LEAD.match(stripped)
        or _ADVICE_ANY.search(stripped)
        or _CONVERSATION_LINE.match(stripped)
    )
    if advisory and not mutation:
        return Intent(kind="advice", argument=stripped)
    if _AUTHORIZING_LEAD.match(stripped) or mutation:
        return Intent(kind="order", argument=stripped, authorized=True, reason="imperative opening")
    return Intent(
        kind="draft",
        argument=stripped,
        reason="no imperative opening and no discussion framing; kept as a draft",
    )


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
