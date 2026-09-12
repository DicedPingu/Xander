"""What the composer knows while you type: `?` for the shortcuts card,
`/` for command completion.

The command list is the live registry in ``intents.COMMANDS`` plus every
name the operator has actually used, so a command added later shows up
without anyone editing a list here. Suggestions are ranked by how often
each has been used in this installation (a small JSON count file under
the state dir), then by registry order. Everything is offline and cheap:
it runs on every keystroke.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .intents import COMMANDS, MODE_HELP, MODES, CommandSpec, command_spec


@dataclass(frozen=True)
class Suggestion:
    text: str  # what tab inserts, e.g. "/history"
    spec: CommandSpec


KEYS: tuple[tuple[str, str], ...] = (
    ("enter", "send the line"),
    ("?", "this card"),
    ("/", "commands · tab completes the first suggestion"),
    ("↑ ↓", "walk your history"),
    ("shift+tab", "mode wheel: " + " → ".join(MODES)),
    ("esc", "back to the composer"),
    ("ctrl+k", "clear the feed"),
    ("ctrl+c", "copy the selected text"),
    ("ctrl+q", "quit"),
    ("f1", "help"),
)


def shortcuts_card(extra_commands: list[str] | None = None) -> list[str]:
    """The lines shown when `?` is typed alone: keys, modes, commands."""

    width = max(len(key) for key, _ in KEYS)
    lines = ["Keys"]
    lines += [f"  {key.ljust(width)}  {what}" for key, what in KEYS]
    lines.append("Modes")
    lines += [f"  {name.ljust(width)}  {MODE_HELP[name]}" for name in MODES]
    lines.append("Commands")
    usage_width = max(len(spec.usage) for spec in COMMANDS)
    for spec in COMMANDS:
        aliases = f"  (also {', '.join(spec.aliases)})" if spec.aliases else ""
        lines.append(f"  {spec.usage.ljust(usage_width)}  {spec.summary}{aliases}")
    for name in extra_commands or []:
        lines.append(f"  {name.ljust(usage_width)}  (used here before)")
    lines.append("Plain language")
    lines.append("  create/do/make… runs · a question gets an answer · 'this project is going to be…' opens a discussion")
    lines.append("  stop / cancel mid-run stops at the next safe point · anything else typed mid-run reaches him there")
    return lines


class CommandCompleter:
    """Prefix completion over the command registry, ranked by use."""

    def __init__(self, store: Path | None = None) -> None:
        self.store = store
        self.counts: dict[str, int] = {}
        if store is not None:
            try:
                loaded = json.loads(store.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.counts = {str(k): int(v) for k, v in loaded.items() if str(k).startswith("/")}
            except (OSError, ValueError, TypeError):
                self.counts = {}

    # -- learning ---------------------------------------------------------------
    def record(self, line: str) -> None:
        """Count one used command. Unknown names are not learned: they were
        refused, and offering them again would teach a mistake."""

        name = line.strip().split(None, 1)[0].casefold() if line.strip() else ""
        if not name.startswith("/") or command_spec(name) is None:
            return
        canonical = command_spec(name).name  # type: ignore[union-attr]
        self.counts[canonical] = self.counts.get(canonical, 0) + 1
        if self.store is not None:
            try:
                self.store.parent.mkdir(parents=True, exist_ok=True)
                self.store.write_text(json.dumps(self.counts, indent=1, sort_keys=True), encoding="utf-8")
            except OSError:
                pass

    # -- suggesting ------------------------------------------------------------
    def suggest(self, text: str, limit: int = 6) -> list[Suggestion]:
        """Commands whose name or alias starts with what is typed so far."""

        typed = text.lstrip()
        if not typed.startswith("/") or " " in typed:
            return []
        prefix = typed.casefold()
        found: list[tuple[int, int, int, Suggestion]] = []
        for order, spec in enumerate(COMMANDS):
            names = (spec.name, *spec.aliases)
            match = next((name for name in names if name.startswith(prefix)), None)
            if match is None:
                continue
            # A canonical name that matches outranks an alias that matches
            # (`/c` is /cd before it is /help via /commands); use beats both.
            via_alias = 0 if spec.name.startswith(prefix) else 1
            text_out = spec.name if not via_alias else match
            found.append((-self.counts.get(spec.name, 0), via_alias, order, Suggestion(text_out, spec)))
        found.sort(key=lambda item: item[:3])
        return [item[3] for item in found[:limit]]

    def complete(self, text: str) -> str:
        """The value the composer should hold after tab."""

        suggestions = self.suggest(text)
        if not suggestions:
            return text
        top = suggestions[0]
        return top.text + (" " if top.spec.takes_argument else "")

    def hint(self, text: str) -> str:
        """One line for the bar under the composer, or "" for the default."""

        typed = text.lstrip()
        if not typed.startswith("/"):
            return ""
        if " " in typed:
            name = typed.split(None, 1)[0].casefold()
            spec = command_spec(name)
            return f"{spec.usage} — {spec.summary}" if spec else f"{name} is not a command · /help lists them"
        suggestions = self.suggest(text)
        if not suggestions:
            return f"no command starts with {typed} · /help lists them"
        top = suggestions[0]
        rest = "  ".join(item.text for item in suggestions[1:5])
        line = f"tab → {top.spec.usage} — {top.spec.summary}"
        return line + (f"   ·   {rest}" if rest else "")
