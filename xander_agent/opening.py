"""Five distinct opening compositions sharing one Mission vocabulary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpeningStyle:
    key: str
    name: str
    description: str
    css_class: str


OPENING_STYLES = (
    OpeningStyle("desk", "Mission Desk", "one clear mission and its next action", "style-desk"),
    OpeningStyle("compass", "Compass", "a calm navigation-first home", "style-compass"),
    OpeningStyle("chronicle", "Chronicle", "a readable history with Now in view", "style-chronicle"),
    OpeningStyle("workshop", "Workshop", "a single outcome prompt with useful context", "style-workshop"),
    OpeningStyle("resident", "Resident", "a quiet companion that stays out of the way", "style-resident"),
)

_STYLE_ALIASES = {
    "tactical": "desk",
    "bento": "workshop",
    "companion": "resident",
    "timeline": "chronicle",
    "glass": "resident",
}


def style_for(key: str) -> OpeningStyle:
    normalized = _STYLE_ALIASES.get(key.casefold().strip(), key.casefold().strip())
    for style in OPENING_STYLES:
        if style.key == normalized:
            return style
    return OPENING_STYLES[0]


def next_style(key: str) -> OpeningStyle:
    keys = [style.key for style in OPENING_STYLES]
    return style_for(keys[(keys.index(style_for(key).key) + 1) % len(keys)])


def wicked_logo(frame: int = 0) -> str:
    """Return a small deterministic logo frame; callers may animate it asynchronously."""

    eyes = ("◈", "◆", "◇", "◆")[frame % 4]
    return f"{eyes}  X A N D E R  {eyes}"


def render_opening(
    style: OpeningStyle,
    *,
    workspace: str,
    branch: str,
    dirty_count: int,
    power: str,
    mission: str,
    mission_state: str,
    history_count: int,
    desktop: str,
    frame: int = 0,
) -> str:
    """Render an opening menu in a distinct layout, without serialised JSON."""

    logo = wicked_logo(frame)
    folder = workspace.rstrip("/").split("/")[-1] or workspace
    repo = f"{branch} · dirty {dirty_count}" if branch != "not-git" else "not a git worktree"
    if style.key == "desk":
        return (
            f"[bold #ffcb6b]{logo}[/]  [bold]MISSION DESK[/]  {folder}  [dim]{repo}[/]\n"
            f"[bold]CURRENT MISSION[/]  {mission[:66]}\n"
            f"status: [bold]{mission_state}[/]   power: [bold #c3e88d]{power}[/]   history: {history_count}\n"
            f"[bold #c3e88d]NEXT[/]  Enter starts the Mission   [bold #ff5370]C[/] cancel active work\n"
            f"[dim]M Mission · H History · S Soul · D Desktop · V choose a different home[/]"
        )
    if style.key == "compass":
        return (
            f"[bold #82aaff]{logo}[/]  [bold]COMPASS[/]  {folder}\n"
            f"[bold #82aaff]▸ Mission[/]       start or return to the current outcome\n"
            f"  History          {history_count} recorded Missions with results\n"
            f"  Soul             stats, behavior, and model settings\n"
            f"  Desktop          {desktop} · explicit local observation\n"
            f"[dim]You are here: {mission_state} · {mission[:54]} · power {power} · {repo}[/]\n"
            f"[bold]Enter[/] open Mission   [bold]h[/] History   [bold]s[/] Soul   [bold]d[/] Desktop   [bold]v[/] change home"
        )
    if style.key == "chronicle":
        return (
            f"[bold #c3e88d]{logo}[/]  [bold]CHRONICLE[/]  {folder}\n"
            f"PAST  ──●────────●────────●────────◎  NOW\n"
            f"       {history_count} Missions recorded · power {power} · {desktop}\n"
            f"NOW  ◎  [bold]{mission_state}[/]  {mission[:63]}\n"
            f"      results are readable, deletable, and tied to this folder\n"
            f"[bold]Enter[/] begin   [bold]h[/] browse History   [bold]s[/] Soul   [bold]d[/] Desktop   [bold]c[/] cancel   [bold]v[/] change home"
        )
    if style.key == "workshop":
        return (
            f"[bold #f78c6c]{logo}[/]  [bold]WORKSHOP[/]  {folder}\n"
            f"[dim]{repo} · power {power} · {history_count} Missions · desktop {desktop}[/]\n"
            f"┌ WHAT OUTCOME SHOULD XANDER PROVE? ────────────────────────────────┐\n"
            f"│ {mission[:66]}\n"
            f"└────────────────────────────────────────────────────────────────────┘\n"
            f"[bold]Enter[/] start   [bold]m[/] edit Mission   [bold]h[/] History   [bold]s[/] Soul   [bold]v[/] choose home"
        )
    return (
        f"[bold #89ddff]{logo}[/]   [bold]XANDER / RESIDENT[/]  {folder}\n"
        f"Mission: [bold]{mission_state}[/] — {mission[:61]}\n"
        f"[dim]{branch} · {dirty_count} local changes · {history_count} Missions · power {power} · desktop {desktop}[/]\n"
        f"[bold]Enter[/] open Mission   [bold]h[/] History   [bold]s[/] Soul   [bold]d[/] Desktop   [bold]c[/] cancel   [bold]v[/] choose home"
    )
