"""Textual interface for Xander's task loop.

The visual language borrows Monica's compact, keyboard-first warmth without
importing or modifying Monica's implementation. Events arrive as structured
payloads and are narrated by :mod:`xander_agent.narrator` so following a run
feels like reading a good build log, not a JSON dump.
"""

from __future__ import annotations

import re
import subprocess
from threading import Event
from pathlib import Path
from typing import Any

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Button, Collapsible, ContentSwitcher, Input, Label, RichLog, Select, Static

from . import MANTRA_PHASES
from .cli import invoke_engine
from .conversation import talk
from .intents import Intent, MODE_HELP, MODE_REQUESTS, parse_intent, resolve_target
from .narrator import Narrator, STATUS_GLYPHS, _short, abilities_line
from .mission import MissionStore
from .power import PowerStatus, PowerZeroGuard, read_power_status
from .workboard import BoardTodo, WorkboardStore

_PHASE_INDEXES = {
    "analyze": 0,
    "research": 1,
    "set_up": 2,
    "set yourself up": 2,
    "work": 3,
    "test": 4,
    "judge_log": 5,
    "judge/log": 5,
    "learn": 6,
    "repeat": 7,
}

_CHANNELS = {
    "run": "#run-log",
    "research": "#research-log",
    "plan": "#plan-log",
    "diff": "#diff-log",
    "tests": "#tests-log",
}

_VIEW_BUTTONS = {
    "activity-view": "#nav-activity",
    "controls-view": "#nav-controls",
    "evidence-view": "#nav-evidence",
    "history-view": "#nav-history",
    "system-view": "#nav-system",
}


def _repository_status(workspace: Path) -> tuple[str, int]:
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if probe.returncode != 0:
            return "not-git", 0
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip()
        dirty_run = subprocess.run(
            ["git", "status", "--short"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if dirty_run.returncode != 0:
            return "not-git", 0
        return branch or "detached", len(dirty_run.stdout.splitlines())
    except (OSError, subprocess.SubprocessError):
        return "not-git", 0


def _xander_workspace() -> Path:
    return Path(__file__).resolve().parents[1]


class XanderApp(App[None]):
    """A task-focused shell shared by the standalone and sidecar workflows."""

    TITLE = "Xander"
    SUB_TITLE = "local coding agent"
    CSS = """
    Screen {
        background: #11151d;
        color: #d7dae0;
    }
    #status-bar {
        height: 1;
        padding: 0 2;
        color: #9aa4b5;
        background: #1b202b;
    }
    #mission-banner {
        height: 3;
        padding: 0 2;
        color: #d7dae0;
        background: #171c25;
        border-bottom: solid #303847;
    }
    #workspace-body {
        height: 1fr;
    }
    #loop-rail {
        width: 36;
        min-width: 30;
        height: 1fr;
        padding: 0 1;
        background: #141922;
        border-right: solid #303847;
        scrollbar-size: 0 1;
        scrollbar-color: #3c4454;
        scrollbar-background: #141922;
    }
    Screen.loop-collapsed #loop-rail {
        display: none;
    }
    #loop-heading {
        height: 3;
        padding: 1 1 0 1;
        color: #89ddff;
    }
    #loop-log {
        height: auto;
        min-height: 7;
        padding: 0 1;
        border: none;
        background: #141922;
    }
    #loop-help {
        height: 2;
        padding: 0 1;
        color: #697386;
    }
    #todo-panel {
        height: auto;
        max-height: 19;
        margin-top: 1;
        padding: 0;
        background: #171d27;
    }
    #todo-log {
        height: auto;
        min-height: 3;
        padding: 0 1;
        border: none;
        background: #10141b;
    }
    #todo-select {
        height: 3;
        margin: 0;
    }
    #todo-entry, #todo-actions {
        height: 3;
        margin: 0;
    }
    #todo-input {
        width: 1fr;
        border: tall #596173;
        background: #171c25;
    }
    #todo-entry Button, #todo-actions Button {
        min-width: 7;
        width: auto;
        margin-left: 1;
    }
    #main-stage {
        width: 1fr;
        height: 1fr;
        background: #11151d;
    }
    #view-nav {
        height: 3;
        padding: 0 1;
        background: #171c25;
        border-bottom: solid #303847;
    }
    #view-nav Button {
        width: auto;
        min-width: 11;
        height: 3;
        border: none;
        color: #8c96a8;
        background: #171c25;
    }
    #view-nav Button.active-nav {
        color: #ffcb6b;
        text-style: bold;
        background: #232a37;
    }
    #view-switcher, .surface {
        height: 1fr;
        background: #11151d;
    }
    #activity-view {
        padding: 1;
    }
    #focus-summary {
        height: 6;
        min-height: 5;
        padding: 0 1 1 1;
        color: #c9d1dc;
        background: #151a22;
        border: round #3c4454;
    }
    #controls-view, #evidence-view, #history-view, #system-view {
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
    }
    #controls-intro, #library-intro, #system-intro {
        height: auto;
        padding: 0 0 1 0;
        color: #c9d1dc;
    }
    #live-controls {
        height: 3;
        margin: 0 0 1 0;
    }
    #live-controls Label {
        width: auto;
        padding: 1 1 0 0;
        color: #82aaff;
    }
    #live-controls Select {
        width: 1fr;
        min-width: 16;
        margin-right: 1;
    }
    .mission-form-row {
        height: 3;
        margin: 0 0 1 0;
    }
    .mission-form-label {
        width: 16;
        padding: 1 1 0 0;
        color: #9aa4b5;
    }
    .mission-form-control {
        width: 1fr;
        border: tall #596173;
        background: #171c25;
    }
    .mission-form-control:focus {
        border: tall #ffb86c;
    }
    #control-actions, #library-actions, #learning-actions, #contest-actions {
        height: 3;
        margin: 0 0 1 0;
    }
    #control-actions Button, #library-actions Button, #learning-actions Button, #contest-actions Button {
        width: auto;
        margin-right: 1;
    }
    #learning-focus, #learning-sources, #contest-input {
        width: 1fr;
        border: tall #596173;
        background: #171c25;
    }
    #mission-guide-panel, #learning-panel, #contest-panel,
    #research-panel, #plan-panel, #changes-panel, #proof-panel,
    #soul-panel, #abilities-panel, #forms-panel, #configure-panel {
        height: auto;
        margin-bottom: 1;
        background: #151a22;
    }
    #mission-guide-log {
        height: 11;
        min-height: 7;
    }
    #learning-log {
        height: 7;
        min-height: 4;
    }
    #research-log, #plan-log, #diff-log, #tests-log {
        height: 12;
        min-height: 7;
    }
    #stats-log, #skills-log, #variants-log {
        height: 12;
        min-height: 7;
    }
    #settings-text {
        height: auto;
        min-height: 5;
        padding: 1;
    }
    #mission-library {
        width: 1fr;
        height: 3;
        margin-bottom: 1;
    }
    #mission-library-detail {
        height: 8;
        min-height: 5;
        padding: 1;
        border: round #3c4454;
        color: #c9d1dc;
    }
    RichLog {
        height: 1fr;
        border: round #3c4454;
        background: #10141b;
    }
    .empty-view {
        color: #798192;
        padding: 1;
    }
    #composer-row {
        dock: bottom;
        height: 3;
        padding: 0 1;
        background: #1b202b;
        border-top: solid #303847;
    }
    #composer-prompt {
        width: 3;
        padding: 1 0 0 1;
        color: #ffcb6b;
        text-style: bold;
    }
    #goal-input {
        width: 1fr;
        border: tall #596173;
        background: #151a22;
    }
    #goal-input:focus {
        border: tall #ffb86c;
    }
    #composer-mode {
        width: auto;
        min-width: 12;
        padding: 1 1 0 1;
        color: #798192;
    }
    """
    # Alt is the primary set: a focused Input swallows plain letters, but never
    # an Alt chord, so every view and control stays reachable mid-sentence
    # without leaving the composer. Ctrl equivalents are kept for muscle memory.
    BINDINGS = [
        # views — by number and by initial
        Binding("alt+1", "show_view('activity-view')", "Activity", show=False, priority=True),
        Binding("alt+2", "show_view('controls-view')", "Controls", show=False, priority=True),
        Binding("alt+3", "show_view('evidence-view')", "Evidence", show=False, priority=True),
        Binding("alt+4", "show_view('history-view')", "History", show=False, priority=True),
        Binding("alt+5", "show_view('system-view')", "Xander", show=False, priority=True),
        Binding("alt+a", "show_view('activity-view')", "Activity", show=False, priority=True),
        Binding("alt+c", "show_view('controls-view')", "Controls", show=False, priority=True),
        Binding("alt+e", "show_view('evidence-view')", "Evidence", show=False, priority=True),
        Binding("alt+h", "show_view('history-view')", "History", show=False, priority=True),
        Binding("alt+x", "show_view('system-view')", "Xander", show=False, priority=True),
        # panels and composer
        Binding("alt+b", "toggle_loop", "Hide/show the loop rail", show=False, priority=True),
        Binding("alt+o", "toggle_todos", "Pinned work", show=False, priority=True),
        Binding("alt+k", "clear_activity", "Clear activity", show=False, priority=True),
        Binding("alt+l", "focus_composer", "Back to the composer", show=False, priority=True),
        # run control
        Binding("alt+n", "new", "New", show=False, priority=True),
        Binding("alt+p", "pause", "Pause", show=False, priority=True),
        Binding("alt+r", "resume", "Resume", show=False, priority=True),
        Binding("alt+s", "contest", "Stop and correct", show=False, priority=True),
        Binding("alt+q", "quit", "Quit", show=False, priority=True),
        Binding("alt+slash", "help", "Help", show=False, priority=True),
        # kept so existing habits still work
        Binding("ctrl+1", "show_view('activity-view')", "Activity", show=False, priority=True),
        Binding("ctrl+2", "show_view('controls-view')", "Controls", show=False, priority=True),
        Binding("ctrl+3", "show_view('evidence-view')", "Evidence", show=False, priority=True),
        Binding("ctrl+4", "show_view('history-view')", "History", show=False, priority=True),
        Binding("ctrl+5", "show_view('system-view')", "Xander", show=False, priority=True),
        Binding("ctrl+b", "toggle_loop", "Toggle loop", show=False, priority=True),
        Binding("ctrl+o", "toggle_todos", "Toggle TODOs", show=False, priority=True),
        Binding("ctrl+l", "focus_composer", "Composer", show=False, priority=True),
        Binding("ctrl+k", "clear_activity", "Clear activity", show=False, priority=True),
        Binding("ctrl+n", "new", "New", show=False, priority=True),
        Binding("ctrl+p", "pause", "Pause", show=False, priority=True),
        Binding("ctrl+shift+p", "resume", "Resume", show=False, priority=True),
        Binding("ctrl+shift+x", "contest", "Stop and contest", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
        Binding("f1", "help", "Help", show=False, priority=True),
    ]

    phase_index = reactive(0)
    task_state = reactive("idle")

    def __init__(
        self,
        *,
        workspace: Path,
        variant: str = "default",
        caller: str = "human",
        autonomy: str | None = None,
        mode: str = "implement",
        model: str = "auto",
        task_id: str = "new",
        power_reader: Any = read_power_status,
        power_shutdown: Any = None,
    ) -> None:
        super().__init__()
        self.workspace = workspace.expanduser().resolve()
        self.variant = variant
        self.caller = caller
        self.autonomy = autonomy
        self.mode = mode
        self.model = model
        self.task_id = task_id
        self._current_view = "activity-view"
        self._current_guide: dict[str, Any] | None = None
        self._known_guide_step_ids: set[str] = set()
        self._live_value_note = "Ready for an outcome."
        self.goal = ""
        self._conversation: list[dict[str, str]] = []
        self._chat_busy = False
        self._chat_queue: list[str] = []
        self._active_run_mode = mode
        self._focus_work = "Waiting for an outcome"
        self._focus_thought = "Give Xander one outcome to understand and prove."
        self._focus_decision = "No decision yet."
        self._focus_change = "Nothing changed yet."
        self._focus_next = "Describe the result you want, then press Enter."
        self._activity_counts: dict[str, int] = {}
        self._mission_constraints: list[str] = []
        self._mission_allowed_paths: list[str] = []
        self._mission_acceptance_checks: list[str] = []
        self._mission_timeout: int | None = None
        self._mission_setup_policy = "ask"
        self._selected_mission_id: str | None = None
        self._delete_armed_id: str | None = None
        self._workboard_store = WorkboardStore()
        self._workboard = self._workboard_store.load(self.workspace)
        self._selected_todo_id: str | None = None
        self._todo_sequence_active = False
        self._active_todo_id: str | None = None
        self.narrator = Narrator(variant=variant, workspace=self.workspace)
        self._engine_busy = False
        self._order_queue: list[dict[str, str]] = []
        self._pending_choice: dict[str, Any] | None = None
        self._pending_approval: dict[str, Any] | None = None
        self._retry_task_id: str | None = None
        self._branch, self._dirty_count = _repository_status(self.workspace)
        self._power_status = PowerStatus(None)
        self._power_guard = PowerZeroGuard(reader=power_reader, shutdown=power_shutdown or self._shutdown_power)

    def _variant_options(self) -> list[tuple[str, str]]:
        try:
            from .variants import list_variants

            names = [profile.name for profile in list_variants()]
        except Exception:
            names = []
        if self.variant not in names:
            names.insert(0, self.variant)
        return [(name, name) for name in dict.fromkeys(names)]

    def compose(self) -> ComposeResult:
        yield Static(self._context_text(), id="status-bar")
        yield Static(self._mission_banner_text(), id="mission-banner")
        with Horizontal(id="workspace-body"):
            with VerticalScroll(id="loop-rail"):
                yield Static(self._loop_heading_text(), id="loop-heading")
                yield Static("\n".join(self._loop_lines()), id="loop-log")
                with Collapsible(
                    title=self._todo_panel_title(),
                    collapsed=True,
                    collapsed_symbol="▸",
                    expanded_symbol="▾",
                    id="todo-panel",
                ):
                    yield Static("[dim]No pinned work yet.[/]", id="todo-log")
                    yield Select([], prompt="Choose pinned work", id="todo-select")
                    with Horizontal(id="todo-entry"):
                        yield Input(placeholder="Pin another outcome", id="todo-input")
                        yield Button("Add", id="add-todo")
                    with Horizontal(id="todo-actions"):
                        yield Button("Run", variant="primary", id="run-todo-sequence")
                        yield Button("Done", id="complete-todo")
                yield Static("⌥B hides this rail · ⌥O pinned work · ⌥/ all keys", id="loop-help")
            with Vertical(id="main-stage"):
                with Horizontal(id="view-nav"):
                    yield Button("⌥A Activity", id="nav-activity", classes="active-nav")
                    yield Button("⌥C Controls", id="nav-controls")
                    yield Button("⌥E Evidence", id="nav-evidence")
                    yield Button("⌥H History", id="nav-history")
                    yield Button("⌥X Xander", id="nav-system")
                with ContentSwitcher(initial="activity-view", id="view-switcher"):
                    with Vertical(id="activity-view", classes="surface"):
                        yield Static(self._focus_summary_text(), id="focus-summary")
                        yield RichLog(id="run-log", wrap=True, highlight=False, markup=True)
                    with VerticalScroll(id="controls-view", classes="surface"):
                        yield Static(
                            "[bold #ffcb6b]LIVE CONTROLS[/]  Type work in the composer. Values changed here remain "
                            "editable while Xander runs and apply on the next engine dispatch.",
                            id="controls-intro",
                        )
                        with Horizontal(id="live-controls"):
                            yield Label("Mode")
                            yield Select(
                                [
                                    ("Build automatically", "implement"),
                                    ("Inspect only", "inspect"),
                                    ("Research only", "research"),
                                    ("Plan only", "plan"),
                                    ("Triage tests", "test-triage"),
                                    ("Talk / ask", "answer"),
                                ],
                                value=self.mode,
                                allow_blank=False,
                                id="mission-mode",
                            )
                            yield Label("Autonomy")
                            yield Select(
                                [(name, name) for name in ("full-auto", "supervised", "proposal-only")],
                                value=self.autonomy or "full-auto",
                                allow_blank=False,
                                id="mission-autonomy",
                            )
                            yield Label("Setup")
                            yield Select(
                                [(name, name) for name in ("ask", "allow", "never")],
                                value="ask",
                                allow_blank=False,
                                id="mission-setup",
                            )
                        with Horizontal(classes="mission-form-row"):
                            yield Label("Variant", classes="mission-form-label")
                            yield Select(
                                self._variant_options(),
                                value=self.variant,
                                allow_blank=False,
                                id="mission-variant",
                                classes="mission-form-control",
                            )
                            yield Label("Time (min)", classes="mission-form-label")
                            yield Input(placeholder="automatic", id="mission-time", classes="mission-form-control")
                        with Horizontal(classes="mission-form-row"):
                            yield Label("Proof", classes="mission-form-label")
                            yield Input(
                                placeholder="Optional command, e.g. pytest -q",
                                id="mission-proof",
                                classes="mission-form-control",
                            )
                        with Horizontal(classes="mission-form-row"):
                            yield Label("Allowed paths", classes="mission-form-label")
                            yield Input(
                                placeholder="Optional, comma-separated",
                                id="mission-allowed",
                                classes="mission-form-control",
                            )
                        with Horizontal(classes="mission-form-row"):
                            yield Label("Constraints", classes="mission-form-label")
                            yield Input(
                                placeholder="Optional, comma-separated",
                                id="mission-constraints",
                                classes="mission-form-control",
                            )
                        with Horizontal(id="control-actions"):
                            yield Button("Apply values", variant="primary", id="apply-controls")
                            yield Button("Reset values", id="reset-controls")
                            yield Button("Open History", id="open-library")
                        with Collapsible(title="Full work loop", collapsed=False, id="mission-guide-panel"):
                            yield RichLog(id="mission-guide-log", wrap=True, highlight=False, markup=True)
                        with Collapsible(title="Learning brief", collapsed=True, id="learning-panel"):
                            yield Input(placeholder="What should Xander learn or compare?", id="learning-focus")
                            yield Input(placeholder="Useful URLs, docs, repos, or commands", id="learning-sources")
                            with Horizontal(id="learning-actions"):
                                yield Button("Apply learning brief", id="save-learning")
                            yield RichLog(id="learning-log", wrap=True, highlight=False, markup=True)
                        with Collapsible(title="Contest or correct the approach", collapsed=True, id="contest-panel"):
                            yield Input(placeholder="What should change about the approach?", id="contest-input")
                            with Horizontal(id="contest-actions"):
                                yield Button("Stop active work", variant="error", id="contest-stop")
                                yield Button("Record direction", id="record-contest")
                    with VerticalScroll(id="evidence-view", classes="surface"):
                        with Collapsible(title="Research and local context", collapsed=True, id="research-panel"):
                            yield RichLog(id="research-log", wrap=True, highlight=False, markup=True)
                        with Collapsible(title="Plan and decisions", collapsed=False, id="plan-panel"):
                            yield RichLog(id="plan-log", wrap=True, highlight=False, markup=True)
                        with Collapsible(title="Changes", collapsed=False, id="changes-panel"):
                            yield RichLog(id="diff-log", wrap=False, highlight=True, markup=True)
                        with Collapsible(title="Proof", collapsed=False, id="proof-panel"):
                            yield RichLog(id="tests-log", wrap=True, highlight=False, markup=True)
                    with VerticalScroll(id="history-view", classes="surface"):
                        yield Static(
                            "[bold #82aaff]HISTORY[/]  Recoverable runs, results, and evolving work loops for this workspace.",
                            id="library-intro",
                        )
                        yield Select([], prompt="Choose a previous run", id="mission-library")
                        with Horizontal(id="library-actions"):
                            yield Button("Continue", variant="primary", id="continue-mission")
                            yield Button("Delete", variant="error", id="delete-mission")
                            yield Button("Refresh", id="refresh-library")
                        yield Static("Select a run to see its loop and result.", id="mission-library-detail")
                        yield RichLog(id="tasks-log", wrap=True, highlight=False, markup=True)
                    with VerticalScroll(id="system-view", classes="surface"):
                        yield Static(
                            "[bold #c3e88d]XANDER[/]  Behavior, available gear, forms, and the values used for future work.",
                            id="system-intro",
                        )
                        with Collapsible(title="Soul and outcomes", collapsed=False, id="soul-panel"):
                            yield RichLog(id="stats-log", wrap=True, highlight=False, markup=True)
                        with Collapsible(title="Available abilities", collapsed=True, id="abilities-panel"):
                            yield RichLog(id="skills-log", wrap=True, highlight=False, markup=True)
                        with Collapsible(title="Forms and variants", collapsed=True, id="forms-panel"):
                            yield RichLog(id="variants-log", wrap=True, highlight=False, markup=True)
                        with Collapsible(title="Current configuration", collapsed=True, id="configure-panel"):
                            yield Static(self._settings_text(), id="settings-text", classes="empty-view")
        with Horizontal(id="composer-row"):
            yield Static("❯", id="composer-prompt")
            yield Input(
                placeholder=f"Tell {self.variant} what you need · questions become conversation · Enter runs",
                id="goal-input",
            )
            yield Static(self._composer_mode_text(), id="composer-mode")

    def on_mount(self) -> None:
        self.query_one("#goal-input", Input).focus()
        self.query_one("#status-bar", Static).tooltip = f"Workspace: {self.workspace}"
        run_log = self.query_one("#run-log", RichLog)
        run_log.write("[bold #ffcb6b]Ready.[/] Tell me the result you want in the field below.")
        run_log.write("[dim]I will update the loop as the plan changes and surface only decisions, changes, proof, or blockers.[/]")
        self._render_guide(None)
        self._refresh_workboard(load_inputs=True)
        self._refresh_tasks()
        self._refresh_variants()
        self._refresh_stats()
        self._load_skills_panel()
        self._poll_power()
        self.set_interval(5, self._poll_power)

    # -- context strips -------------------------------------------------------
    def _context_text(self) -> str:
        folder = self.workspace.name or str(self.workspace)
        repository = "not a repo" if self._branch == "not-git" else self._branch
        if self._dirty_count:
            repository += f" +{self._dirty_count}"
        queue = f" · queued {len(self._order_queue)}" if self._order_queue else ""
        conversation = " · talking" if self._chat_busy else ""
        return (
            f"[bold #ffcb6b]XANDER[/]  {escape(self.variant)}  ·  {escape(folder)}  ·  {escape(repository)}  ·  "
            f"[bold]{escape(self.task_state)}[/]  ·  {escape(self._phase_name())}  ·  "
            f"power {escape(self._power_status.label)}{queue}{conversation}"
        )

    def _phase_name(self) -> str:
        return MANTRA_PHASES[max(0, min(self.phase_index, len(MANTRA_PHASES) - 1))]

    def _mission_banner_text(self) -> str:
        if not self.goal:
            return (
                "[bold #ffcb6b]Ready[/]\n"
                "[dim]Describe the result below. Use /controls when you want exact values.[/]"
            )
        guide = self._current_guide or {}
        current = str(guide.get("current") or self._focus_thought or self._live_value_note)
        progress = str(guide.get("progress") or "building the loop")
        return (
            f"[bold #ffcb6b]{escape(_short(self.goal, 120))}[/]\n"
            f"[dim]{escape(self._phase_name())} · {escape(progress)}[/]  {escape(_short(current, 130))}"
        )

    def _composer_mode_text(self) -> str:
        authority = self.autonomy or "profile"
        return f"{self.variant} · {self.mode} · {authority}"

    def _focus_summary_text(self) -> str:
        return (
            f"[bold #82aaff]NOW[/] {escape(_short(self._focus_work, 120))}\n"
            f"[bold #c792ea]DECISION[/] {escape(_short(self._focus_decision, 120))}\n"
            f"[bold #f78c6c]THINKING[/] {escape(_short(self._focus_thought, 120))}\n"
            f"[bold #ffcb6b]CHANGED[/] {escape(_short(self._focus_change, 105))}  "
            f"[bold #c3e88d]NEXT[/] {escape(_short(self._focus_next, 105))}"
        )

    def _loop_heading_text(self) -> str:
        progress = str((self._current_guide or {}).get("progress") or "waiting")
        return f"[bold #89ddff]ADAPTIVE LOOP[/]\n[dim]{escape(self._phase_name())} · {escape(progress)}[/]"

    def _todo_panel_title(self) -> str:
        pending = sum(todo.state != "done" for todo in self._workboard.todos)
        active = sum(todo.state == "active" for todo in self._workboard.todos)
        suffix = f" · {active} active" if active else ""
        return f"Pinned work · {pending} pending{suffix}"

    def _loop_lines(self) -> list[str]:
        guide = self._current_guide or {}
        steps = [step for step in guide.get("todo") or [] if isinstance(step, dict)]
        if not steps:
            return [
                "[dim]Xander will write the first loop before analysis and grow it from the actual plan.[/]",
            ]
        lines: list[str] = []
        for index, step in enumerate(steps, start=1):
            state = str(step.get("state") or "todo")
            glyph = {"done": "✓", "active": "◆", "blocked": "!"}.get(state, "○")
            style = {
                "done": "green",
                "active": "bold #ffcb6b",
                "blocked": "bold #ff5370",
            }.get(state, "#7e899b")
            text = escape(_short(str(step.get("text") or "Untitled step"), 90))
            lines.append(f"[{style}]{glyph} {index}[/]  {text}")
            evidence = " ".join(str(step.get("evidence") or "").split())
            if evidence and state in {"done", "blocked"}:
                lines.append(f"   [dim]{escape(_short(evidence, 82))}[/]")
        questions = [str(item) for item in guide.get("questions") or [] if str(item).strip()]
        if questions:
            lines.append(f"\n[bold yellow]?[/] {escape(_short(questions[0], 96))}")
        return lines

    def _refresh_loop(self) -> None:
        headings = self.query("#loop-heading")
        if len(headings):
            headings.first(Static).update(self._loop_heading_text())
        logs = self.query("#loop-log")
        if len(logs):
            logs.first(Static).update("\n".join(self._loop_lines()))

    def _refresh_focus(self) -> None:
        banners = self.query("#mission-banner")
        if len(banners):
            banners.first(Static).update(self._mission_banner_text())
        statuses = self.query("#status-bar")
        if len(statuses):
            statuses.first(Static).update(self._context_text())
        modes = self.query("#composer-mode")
        if len(modes):
            modes.first(Static).update(self._composer_mode_text())
        summaries = self.query("#focus-summary")
        if len(summaries):
            summaries.first(Static).update(self._focus_summary_text())
        composers = self.query("#goal-input")
        if len(composers):
            composers.first(Input).placeholder = (
                f"Tell {self.variant} what you need · questions become conversation · Enter runs"
            )
        self._refresh_loop()

    def _update_focus(self, payload: dict[str, Any]) -> None:
        event_type = str(payload.get("type") or payload.get("event") or "")
        message = " ".join(str(payload.get("message", "")).split())
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        phase = str(payload.get("phase") or data.get("phase") or "")
        result = data.get("result") if isinstance(data.get("result"), dict) else {}
        task = data.get("task") if isinstance(data.get("task"), dict) else {}
        if event_type == "phase":
            label = phase.replace("_", " ").title()
            self._focus_thought = f"{label}: {message}" if message else label
            self._focus_next = "Gather the next piece of evidence before moving on."
        elif event_type == "research":
            sources = len(data.get("sources") or [])
            tools = len(data.get("tools") or [])
            self._focus_thought = message or "Looking for local context and relevant tools."
            if sources or tools:
                self._focus_thought = f"{self._focus_thought} ({sources} sources, {tools} tools)"
            self._focus_next = "Turn the useful context into a small plan."
        elif event_type == "delegation":
            agent = str(data.get("agent") or "another worker")
            operation = str(data.get("operation") or message or "support work")
            status = str(data.get("status") or "working")
            self._focus_thought = f"{status.title()}: {agent} · {operation}"
            self._focus_next = "Follow the returned evidence before accepting the next step."
        elif event_type == "plan":
            actions = data.get("actions", 0)
            checks = data.get("checks", 0)
            decision = str(data.get("decision") or message or "Plan shaped from the available evidence.")
            self._focus_decision = decision
            self._focus_thought = message or f"Shaping {actions} bounded action(s)."
            self._focus_thought = f"{self._focus_thought} ({actions} actions, {checks} checks)"
            self._focus_next = "Review the plan, then make the smallest safe change."
        elif event_type == "logic_change":
            self._focus_decision = str(data.get("replacement_summary") or message or "Change course from new evidence.")
            self._focus_thought = "The approach changed after evidence; review or contest it before the next step."
            self._focus_next = "Use Ctrl+Shift+X to stop and contest, or let the changed approach continue."
        elif event_type in {"action", "patch"}:
            action = data.get("action") if isinstance(data.get("action"), dict) else {}
            expected = str(action.get("expected") or "").strip()
            if expected:
                self._focus_work = expected
            elif message and event_type == "patch":
                self._focus_work = message
            changed = result.get("changed_paths") or []
            if changed:
                self._focus_change = ", ".join(str(path) for path in changed[:4])
                if len(changed) > 4:
                    self._focus_change += f" (+{len(changed) - 4} more)"
            self._focus_thought = "Applying one bounded step and checking its result."
            if result.get("status") not in {None, "ok"}:
                self._focus_thought = f"The step reported {result.get('status')}; checking the evidence."
            self._focus_next = "Check the result before choosing another step."
        elif event_type == "test":
            self._focus_work = str(data.get("name") or message or "acceptance checks")
            status = str(result.get("status") or message or "in progress")
            self._focus_thought = f"Proof: {status}"
            self._focus_next = "Judge the evidence, not the intention."
        elif event_type == "voice":
            if message:
                self._focus_thought = message
        elif event_type == "error":
            self._focus_thought = f"Blocked: {message}"
            self._focus_next = "Stop repeating this path; choose a changed approach."
        elif event_type == "result":
            mission_task = task or (data.get("handoff") if isinstance(data.get("handoff"), dict) else {})
            status = str(payload.get("status") or data.get("status") or mission_task.get("status") or "recorded")
            self._focus_work = self.goal or self._focus_work
            self._focus_thought = f"Work {status}; the result is recorded."
            self._focus_next = "Open History for the concise record and result."
        self._refresh_focus()

    @staticmethod
    def _activity_signature_for(payload: dict[str, Any], line: str) -> str:
        signature = re.sub(r"\+\s*[\d.]+s", "", line)
        signature = re.sub(r"\ba\d+\b", "a#", signature)
        signature = re.sub(r"\battempt\s+\d+\b", "attempt #", signature, flags=re.IGNORECASE)
        return signature

    def _human_event_line(self, payload: dict[str, Any], narrated: str) -> str | None:
        event_type = str(payload.get("type") or payload.get("event") or "").casefold()
        message = " ".join(str(payload.get("message") or "").split())
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        result = data.get("result") if isinstance(data.get("result"), dict) else {}
        action = data.get("action") if isinstance(data.get("action"), dict) else {}
        phase = str(payload.get("phase") or data.get("phase") or "").replace("_", " ").title()

        if event_type == "phase":
            detail = f" — {escape(_short(message, 130))}" if message else ""
            return f"[bold #c792ea]{escape(phase or 'Next phase')}[/]{detail}"
        if event_type == "guide":
            guide = data.get("guide") if isinstance(data.get("guide"), dict) else {}
            progress = str(guide.get("progress") or "loop created")
            current = str(guide.get("current") or message)
            # The rail on the left already renders the live loop; repeating it
            # in the feed is the same fact twice. Only a question earns a line.
            questions = [str(item) for item in (guide.get("questions") or []) if str(item).strip()]
            if questions:
                return f"[bold yellow]He needs an answer[/] — {escape(_short(questions[0], 130))}"
            return None
        if event_type == "research":
            sources = len(data.get("sources") or [])
            tools = len(data.get("tools") or [])
            counts = []
            if sources:
                counts.append(f"{sources} source{'s' if sources != 1 else ''}")
            if tools:
                counts.append(f"{tools} tool{'s' if tools != 1 else ''}")
            suffix = f" · {', '.join(counts)}" if counts else ""
            return f"[bold #82aaff]Research[/] — {escape(_short(message or 'context collected', 130))}{suffix}"
        if event_type == "delegation":
            # "Master started — plan review" / "Master completed — plan review"
            # is pure ceremony: it costs two lines to say nothing happened.
            # Only a verdict or a failure is worth the operator's eye.
            status = str(data.get("status") or "").casefold()
            outcome = str(data.get("outcome") or "").casefold()
            error = str(data.get("error") or "")
            agent = str(data.get("agent") or "Support agent")
            if error or status in {"failed", "blocked"} or outcome in {"failed", "blocked"}:
                detail = error or str(data.get("operation") or message or "support work")
                return f"[bold #ff5370]{escape(agent)} could not help[/] — {escape(_short(detail, 120))}"
            if status == "selected" and data.get("role"):
                role = str(data["role"])
                reason = str(data.get("reason") or data.get("operation") or message)
                return f"[bold #89ddff]Model route[/] — {escape(role)} · {escape(_short(reason, 120))}"
            return None
        if event_type == "plan":
            actions = data.get("actions")
            checks = data.get("checks")
            if isinstance(actions, list):
                actions = len(actions)
            if isinstance(checks, list):
                checks = len(checks)
            size = []
            if isinstance(actions, int):
                size.append(f"{actions} step{'s' if actions != 1 else ''}")
            if isinstance(checks, int):
                size.append(f"{checks} check{'s' if checks != 1 else ''}")
            suffix = f" · {', '.join(size)}" if size else ""
            decision = str(data.get("decision") or message or "ready")
            why = str(data.get("why") or "")
            reason = f" · why: {escape(_short(why, 105))}" if why else ""
            return f"[bold #c3e88d]Decision[/] — {escape(_short(decision, 130))}{reason}{suffix}"
        if event_type == "logic_change":
            reason = str(data.get("reason") or message or "new evidence")
            replacement = str(data.get("replacement_summary") or data.get("replacement") or "")
            suffix = f" → {escape(_short(replacement, 90))}" if replacement else ""
            return f"[bold #f78c6c]Approach changed[/] — {escape(_short(reason, 120))}{suffix}"
        if event_type in {"action", "patch"}:
            expected = str(action.get("expected") or message or "bounded step")
            changed = [str(path) for path in result.get("changed_paths") or []]
            status = str(result.get("status") or "")
            wrote = data.get("wrote") if isinstance(data.get("wrote"), list) else []
            if wrote:
                # Say what the file now IS. "Changed — main.cpp" hid the fact
                # that main.cpp was a nine-line stub.
                described = []
                for item in wrote[:4]:
                    name = escape(str(item.get("path", "?")))
                    if "lines" not in item:
                        described.append(name)
                    elif item.get("placeholder"):
                        described.append(f"{name} [bold #ff5370]({item['lines']} lines — still a placeholder)[/]")
                    else:
                        described.append(f"{name} [dim]({item['lines']} lines)[/]")
                more = f" (+{len(wrote) - 4})" if len(wrote) > 4 else ""
                return f"[bold #ffcb6b]Wrote[/] — " + ", ".join(described) + more
            if changed:
                paths = ", ".join(changed[:4])
                if len(changed) > 4:
                    paths += f" (+{len(changed) - 4})"
                return f"[bold #ffcb6b]Changed[/] — {escape(paths)}"
            if status and status != "ok":
                reason = str(result.get("reason") or expected)
                return f"[bold #ff5370]Step {escape(status)}[/] — {escape(_short(reason, 130))}"
            if result and status == "ok":
                return f"[bold #c3e88d]Step done[/] — {escape(_short(expected, 135))}"
            verb = "Changed" if event_type == "patch" else "Doing"
            return f"[bold #ffcb6b]{verb}[/] — {escape(_short(expected, 135))}"
        if event_type == "test":
            # A check emits twice: once on start with no result, once with one.
            # The start line has nothing to say and was rendering as
            # "Proof <name> — <name>", so it is dropped entirely.
            if not result:
                return None
            name = str(data.get("name") or message or "acceptance check")
            status = str(result.get("status") or "")
            code = result.get("returncode")
            evidence = " ".join(str(result.get("reason") or "").split())
            if not evidence and status != "ok":
                tail = (result.get("stderr") or result.get("stdout") or "").strip().splitlines()
                evidence = tail[-1] if tail else ""
            if status == "ok":
                return f"[bold #c3e88d]Proof passed[/] — {escape(_short(name, 110))}"
            marker = f" (exit {code})" if isinstance(code, int) and code else ""
            detail = f" · {escape(_short(evidence, 90))}" if evidence else ""
            return f"[bold #ff5370]Proof failed[/]{marker} — {escape(_short(name, 100))}{detail}"
        if event_type == "approval":
            return f"[bold yellow]Your decision is needed[/] — {escape(_short(message, 135))}"
        if event_type == "error":
            return f"[bold #ff5370]Blocked[/] — {escape(_short(message or 'the current approach failed', 150))}"
        if event_type == "result":
            status = str(payload.get("status") or data.get("status") or "recorded")
            label = "Done" if status in {"complete", "completed"} else status.replace("_", " ").title()
            return f"[bold green]{escape(label)}[/] — {escape(_short(message or 'result recorded', 145))}"
        if event_type == "voice" and message:
            return f"[italic #b8c0cc]{escape(_short(message, 150))}[/]"
        if message:
            return f"[dim]{escape(_short(message, 150))}[/]"
        return narrated or None

    def _write_activity(self, payload: dict[str, Any], line: str) -> None:
        signature = self._activity_signature_for(payload, line)
        run_log = self.query_one("#run-log", RichLog)
        if signature in self._activity_counts:
            self._activity_counts[signature] += 1
            return
        run_log.write(line)
        self._activity_counts[signature] = 1

    def _flush_activity(self) -> None:
        repeated = sum(count - 1 for count in self._activity_counts.values() if count > 1)
        patterns = sum(1 for count in self._activity_counts.values() if count > 1)
        if repeated:
            self.query_one("#run-log", RichLog).write(
                f"[dim]folded {repeated} repeated entr{'y' if repeated == 1 else 'ies'} "
                f"across {patterns} unchanged pattern{'s' if patterns != 1 else ''}[/]"
            )

    def _refresh_workboard(self, *, load_inputs: bool = False) -> None:
        panels = self.query("#todo-panel")
        if len(panels):
            panels.first(Collapsible).title = self._todo_panel_title()
        todo_matches = self.query("#todo-log")
        if len(todo_matches):
            lines = []
            for todo in self._workboard.todos:
                glyph = {"done": "✓", "active": "▸", "blocked": "!"}.get(todo.state, "·")
                style = {"done": "green", "active": "bold #ffcb6b", "blocked": "bold #ff5370"}.get(todo.state, "dim")
                detail = f" · {todo.evidence}" if todo.evidence else ""
                lines.append(f"[{style}]{glyph}[/] {escape(todo.text)} [dim]({todo.state}){escape(detail)}[/]")
            todo_matches.first(Static).update("\n".join(lines or ["[dim]No pinned work yet.[/]"]))
        select_matches = self.query("#todo-select")
        if len(select_matches):
            select = select_matches.first(Select)
            previous = select.value
            options = [(f"{todo.state.upper()} · {_short(todo.text, 72)}", todo.id) for todo in self._workboard.todos]
            select.set_options(options)
            ids = {value for _, value in options}
            if isinstance(previous, str) and previous in ids:
                select.value = previous
            elif self._selected_todo_id in ids:
                select.value = self._selected_todo_id
            elif options:
                select.value = options[0][1]
            else:
                select.clear()
            self._selected_todo_id = select.value if isinstance(select.value, str) else None
        learning_matches = self.query("#learning-log")
        if len(learning_matches):
            lines = ["[dim]Verified lessons are added automatically after a task produces evidence.[/]"]
            for lesson in self._workboard.observed_lessons[-6:]:
                lines.append(f"[bold #c3e88d]✓[/] {escape(_short(lesson, 135))}")
            if self._workboard.learning_sources:
                lines.append("[bold #82aaff]sources:[/] " + escape(" · ".join(self._workboard.learning_sources)))
            if self._workboard.contest_comments:
                lines.append("[bold #ff5370]contests:[/] " + escape(_short(self._workboard.contest_comments[-1], 135)))
            learning_matches.first(RichLog).clear()
            for line in lines:
                learning_matches.first(RichLog).write(line)
        if load_inputs:
            for selector, value in (
                ("#learning-focus", self._workboard.learning_focus),
                ("#learning-sources", ", ".join(self._workboard.learning_sources)),
            ):
                input_matches = self.query(selector)
                if len(input_matches):
                    input_matches.first(Input).value = value

    def _learning_constraints(self) -> list[str]:
        constraints = [
            "Treat command exit codes, changed files, artifacts, and runtime behavior as ground truth; do not infer success from intent.",
            "When a check fails, test a materially different angle before repeating the same approach.",
        ]
        if self._workboard.learning_focus:
            constraints.append(f"Learning target: {self._workboard.learning_focus}")
        if self._workboard.learning_sources:
            constraints.append("Learning sources or destinations: " + ", ".join(self._workboard.learning_sources))
        constraints.extend(f"Operator contest/comment to respect: {item}" for item in self._workboard.contest_comments[-3:])
        return constraints

    def _settings_text(self) -> str:
        try:
            from .variants import load_variant

            profile = load_variant(self.variant)
            autonomy = self.autonomy or profile.autonomy
            routing = "  ".join(f"{role}:{model.rsplit('/', 1)[-1]}" for role, model in profile.model_routing.items())
        except Exception as exc:
            autonomy, routing = "full-auto", f"unavailable ({exc})"
        return escape(
            f"workspace={self.workspace}\n"
            f"caller={self.caller}  mode={self.mode}  model={self.model}  autonomy={autonomy}\n"
            f"setup={self._mission_setup_policy}  state={self.task_state}\n"
            f"abilities: {abilities_line()}\n"
            f"routing: {routing}"
        )

    def watch_phase_index(self, _: int) -> None:
        self._refresh_focus()

    def watch_task_state(self, _: str) -> None:
        self._refresh_focus()

    # -- side panels ----------------------------------------------------------
    def _fill_log(self, selector: str, lines: list[str]) -> None:
        log = self.query_one(selector, RichLog)
        log.clear()
        for line in lines:
            log.write(line)

    def _render_guide(self, guide: dict[str, Any] | None) -> None:
        previous_ids = set(self._known_guide_step_ids)
        self._current_guide = guide
        self._known_guide_step_ids = {
            str(step.get("id"))
            for step in (guide or {}).get("todo") or []
            if isinstance(step, dict) and step.get("id")
        }
        added = self._known_guide_step_ids - previous_ids
        if previous_ids and added:
            run_logs = self.query("#run-log")
            if len(run_logs):
                run_logs.first(RichLog).write(
                    f"[bold #89ddff]Loop updated[/] — Xander added {len(added)} new "
                    f"step{'s' if len(added) != 1 else ''} from the plan."
                )
        self._refresh_focus()
        matches = self.query("#mission-guide-log")
        if not len(matches):
            return
        log = matches.first(RichLog)
        log.clear()
        if not guide:
            log.write("[dim]The guide will be written before analysis and revised from real evidence.[/]")
            return
        statement = escape(str(guide.get("statement") or ""))
        progress = escape(str(guide.get("progress") or ""))
        current = escape(str(guide.get("current") or ""))
        log.write(f"[bold #ffcb6b]Outcome[/]  {statement}")
        log.write(f"[bold #82aaff]Progress[/] {progress}")
        log.write(f"[bold #c3e88d]Current[/]  {current}")
        for step in guide.get("todo") or []:
            if not isinstance(step, dict):
                continue
            state = str(step.get("state") or "todo")
            glyph = {"done": "✓", "active": "▸", "blocked": "!"}.get(state, "·")
            style = {"done": "green", "active": "bold #ffcb6b", "blocked": "bold red"}.get(state, "dim")
            log.write(f"  [{style}]{glyph}[/] {escape(str(step.get('text') or ''))}")
        for question in guide.get("questions") or []:
            log.write(f"[bold yellow]Question[/] {escape(str(question))}")
        if guide.get("last_change"):
            log.write(f"[bold #c792ea]Changed[/] {escape(str(guide['last_change']))}")
        if guide.get("result"):
            log.write(f"[bold green]Result[/] {escape(str(guide['result']))}")

    def _update_library_detail(self, mission_id: str | None = None) -> None:
        matches = self.query("#mission-library-detail")
        if not len(matches):
            return
        if not mission_id:
            matches.first(Static).update("Select a run to see its loop and result.")
            return
        try:
            mission = MissionStore().load(self.workspace, mission_id)
        except Exception as exc:
            matches.first(Static).update(f"[red]Run unavailable: {escape(str(exc))}[/]")
            return
        lines = [
            f"[bold #ffcb6b]{escape(mission.goal)}[/]",
            f"{escape(mission.status)} · {escape(mission.phase)} · {escape(mission.result)}",
        ]
        if mission.task.guide:
            lines.append(f"guide: {escape(mission.task.guide.progress)} · {escape(mission.task.guide.current)}")
            steps = [
                f"{'✓' if step.state == 'done' else '▸' if step.state == 'active' else '!' if step.state == 'blocked' else '·'} {step.text}"
                for step in mission.task.guide.todo[:3]
            ]
            if steps:
                lines.append("todo: " + " · ".join(escape(step) for step in steps))
            if mission.task.guide.questions:
                lines.append(f"question: {escape(mission.task.guide.questions[0])}")
        matches.first(Static).update("\n".join(lines))

    def _selected_mission(self) -> Any:
        matches = self.query("#mission-library")
        if not len(matches):
            return None
        selected = matches.first(Select).value
        if not isinstance(selected, str) or not selected:
            return None
        try:
            return MissionStore().load(self.workspace, selected)
        except Exception as exc:
            self.query_one("#run-log", RichLog).write(f"[red]Run unavailable:[/] {escape(str(exc))}")
            return None

    @staticmethod
    def _form_values(text: str) -> list[str]:
        return [item.strip() for item in re.split(r"[,;]", text) if item.strip()]

    def _selected_todo(self) -> BoardTodo | None:
        selected = self._selected_todo_id
        if not selected:
            matches = self.query("#todo-select")
            if len(matches) and isinstance(matches.first(Select).value, str):
                selected = matches.first(Select).value
        return next((todo for todo in self._workboard.todos if todo.id == selected), None)

    def _add_todo(self) -> None:
        field = self.query_one("#todo-input", Input)
        todo = self._workboard_store.add_todo(self._workboard, field.value)
        if todo is None:
            self.notify("Enter one useful TODO item first.", title="TODO")
            return
        field.value = ""
        self._selected_todo_id = todo.id
        self._refresh_workboard()
        self._refresh_focus()
        self.query_one("#run-log", RichLog).write(f"[bold #89ddff]TODO added[/] {escape(todo.text)} [dim](cosmetic until launched)[/]")

    def _save_learning(self) -> None:
        focus = self.query_one("#learning-focus", Input).value
        sources = self._form_values(self.query_one("#learning-sources", Input).value)
        self._workboard_store.set_learning(self._workboard, focus, sources)
        self._refresh_workboard(load_inputs=True)
        self._refresh_focus()
        detail = _short(focus or "no focus", 90)
        if sources:
            detail += " · sources: " + ", ".join(sources)
        self.query_one("#run-log", RichLog).write(f"[bold #c3e88d]learning brief saved[/] {escape(detail)}")
        self.narrator.record(f"learning brief: {detail}", task_id=self.task_id)

    def _record_contest_comment(self) -> None:
        field = self.query_one("#contest-input", Input)
        text = field.value.strip()
        if not text:
            self.notify("Write the contest or comment before recording it.", title="Contest/comment")
            field.focus()
            return
        self._workboard_store.add_comment(self._workboard, text)
        field.value = ""
        self._refresh_workboard()
        self.query_one("#run-log", RichLog).write(f"[bold #ff5370]operator contest/comment recorded[/] {escape(text)}")
        self.narrator.record(f"operator contest/comment: {text}", task_id=self.task_id)

    def _complete_selected_todo(self) -> None:
        todo = self._selected_todo()
        if todo is None:
            self.notify("Choose a TODO first.", title="TODO")
            return
        if todo.id == self._active_todo_id and (self.task_state == "running" or self._engine_busy):
            self.notify("Stop the active sequence before marking its TODO done.", title="TODO")
            return
        self._workboard_store.set_todo_state(self._workboard, todo.id, "done", "marked complete by operator")
        self._refresh_workboard()
        self._refresh_focus()
        self.query_one("#run-log", RichLog).write(f"[bold #c3e88d]TODO marked done[/] {escape(todo.text)}")

    def _run_todo_sequence(self) -> None:
        if self.task_state == "power-zero":
            return
        if self.task_state == "running" or self._engine_busy:
            self.notify("Stop the active work before starting pinned guidance.", title="Pinned sequence")
            return
        todo = self._selected_todo() or next((item for item in self._workboard.todos if item.state == "todo"), None)
        if todo is None:
            self.notify("Add or select a pending TODO first.", title="TODO sequence")
            return
        self._todo_sequence_active = True
        self._active_todo_id = todo.id
        self._workboard_store.set_todo_state(self._workboard, todo.id, "active")
        self._refresh_workboard()
        self.query_one("#run-log", RichLog).write(
            f"[bold #ffcb6b]TODO sequence started[/] one item at a time · {escape(todo.text)}"
        )
        self._dispatch(todo.text, self.mode)

    def _start_next_todo(self) -> bool:
        todo = next((item for item in self._workboard.todos if item.state == "todo"), None)
        if todo is None:
            return False
        self._active_todo_id = todo.id
        self._workboard_store.set_todo_state(self._workboard, todo.id, "active")
        self._refresh_workboard()
        self.query_one("#run-log", RichLog).write(
            f"[bold #ffcb6b]TODO sequence next[/] {escape(todo.text)}"
        )
        self._dispatch(todo.text, self.mode)
        return True

    def _release_todo(self, state: str, evidence: str = "") -> None:
        if self._active_todo_id:
            self._workboard_store.set_todo_state(self._workboard, self._active_todo_id, state, evidence)
            self._active_todo_id = None
            self._refresh_workboard()

    def _reset_controls(self) -> None:
        self._mission_constraints = []
        self._mission_allowed_paths = []
        self._mission_acceptance_checks = []
        self._mission_timeout = None
        self._mission_setup_policy = "ask"
        self.mode = "implement"
        self.autonomy = "full-auto"
        for selector in ("#mission-time", "#mission-proof", "#mission-allowed", "#mission-constraints"):
            matches = self.query(selector)
            if len(matches):
                matches.first(Input).value = ""
        matches = self.query("#mission-mode")
        if len(matches):
            matches.first(Select).value = "implement"
        matches = self.query("#mission-autonomy")
        if len(matches):
            matches.first(Select).value = "full-auto"
        matches = self.query("#mission-setup")
        if len(matches):
            matches.first(Select).value = "ask"
        matches = self.query("#mission-variant")
        if len(matches):
            matches.first(Select).value = self.variant
        self._live_value_note = "Controls reset."
        self._refresh_focus()

    def _capture_control_inputs(self, *, announce: bool = False) -> bool:
        mode = self.query_one("#mission-mode", Select).value
        authority = self.query_one("#mission-autonomy", Select).value
        setup_policy = self.query_one("#mission-setup", Select).value
        variant = self.query_one("#mission-variant", Select).value
        time_text = self.query_one("#mission-time", Input).value.strip()
        timeout = None
        if time_text:
            try:
                timeout = max(60, int(time_text) * 60)
            except ValueError:
                self.notify("Time must be a whole number of minutes.", title="Live controls")
                self.query_one("#mission-time", Input).focus()
                return False
        self.mode = str(mode) if isinstance(mode, str) else "implement"
        self.autonomy = str(authority) if isinstance(authority, str) else "full-auto"
        self._mission_setup_policy = str(setup_policy) if isinstance(setup_policy, str) else "ask"
        if isinstance(variant, str) and variant != self.variant:
            self.variant = variant
            self.narrator = Narrator(variant=variant, workspace=self.workspace)
        self._mission_timeout = timeout
        self._mission_acceptance_checks = self._form_values(self.query_one("#mission-proof", Input).value)
        self._mission_allowed_paths = self._form_values(self.query_one("#mission-allowed", Input).value)
        self._mission_constraints = self._form_values(self.query_one("#mission-constraints", Input).value)
        self._refresh_focus()
        settings = self.query("#settings-text")
        if len(settings):
            settings.first(Static).update(self._settings_text())
        if announce:
            timing = "next dispatch" if self.task_state == "running" or self._engine_busy else "ready"
            self.query_one("#run-log", RichLog).write(
                f"[bold #82aaff]Controls applied[/] — mode {escape(self.mode)}, "
                f"authority {escape(self.autonomy or 'profile')}, setup {escape(self._mission_setup_policy)} "
                f"[dim]({timing})[/]"
            )
        return True

    def _apply_controls(self) -> None:
        self._capture_control_inputs(announce=True)

    def _continue_selected_mission(self) -> None:
        if self.task_state == "power-zero":
            return
        if self.task_state == "running" or self._engine_busy:
            self.action_cancel()
            self.notify("Active work cancelled. Press Continue again when ready.", title="Work still active")
            return
        mission = self._selected_mission()
        if mission is None:
            self.notify("Choose a run from History first.", title="History")
            return
        self.goal = mission.goal
        self.task_id = mission.id
        self._retry_task_id = None
        self.task_state = "running"
        self.phase_index = 0
        self._focus_work = mission.goal
        self._focus_thought = "Reopening the work loop and looking for the next improvement."
        self._focus_next = "Update the guide, then prove the new result."
        guide = mission.task.guide.model_dump(mode="json") if mission.task.guide else None
        self._render_guide(guide)
        self._refresh_focus()
        self._show_view("activity-view")
        self.query_one("#run-log", RichLog).write(
            f"[bold #ffcb6b]▶ continuing run[/] {escape(mission.id)} · {escape(mission.goal)}"
        )
        self._run_resume(mission.id, [])

    def _delete_selected_mission(self) -> None:
        mission = self._selected_mission()
        if mission is None:
            self.notify("Choose a run from History first.", title="History")
            return
        if mission.id == self.task_id and (self.task_state == "running" or self._engine_busy):
            self.notify("Cancel the active work before deleting its run.", title="Work still active")
            return
        if self._delete_armed_id != mission.id:
            self._delete_armed_id = mission.id
            self.query_one("#run-log", RichLog).write(
                f"[yellow]Delete {escape(mission.id)}? Press Delete selected again to confirm.[/]"
            )
            return
        try:
            MissionStore().delete(self.workspace, mission.id)
        except Exception as exc:
            self.notify(f"Could not delete run: {exc}", title="History")
            return
        self._delete_armed_id = None
        self._selected_mission_id = None
        self._refresh_tasks()
        self.query_one("#run-log", RichLog).write(f"[dim]Run deleted:[/] {escape(mission.id)}")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "mission-library":
            self._selected_mission_id = event.value if isinstance(event.value, str) else None
            self._delete_armed_id = None
            self._update_library_detail(self._selected_mission_id)
        elif event.select.id == "todo-select":
            self._selected_todo_id = event.value if isinstance(event.value, str) else None
        elif isinstance(event.value, str) and event.select.id in {
            "mission-mode",
            "mission-autonomy",
            "mission-setup",
            "mission-variant",
        }:
            current = {
                "mission-mode": self.mode,
                "mission-autonomy": self.autonomy or "full-auto",
                "mission-setup": self._mission_setup_policy,
                "mission-variant": self.variant,
            }[event.select.id]
            if event.value == current:
                return
            label = {
                "mission-mode": "Mode",
                "mission-autonomy": "Authority",
                "mission-setup": "Setup policy",
                "mission-variant": "Variant",
            }[event.select.id]
            if event.select.id == "mission-mode":
                self.mode = event.value
            elif event.select.id == "mission-autonomy":
                self.autonomy = event.value
            elif event.select.id == "mission-setup":
                self._mission_setup_policy = event.value
            else:
                self.variant = event.value
                self.narrator = Narrator(variant=event.value, workspace=self.workspace)
            timing = "next dispatch" if self.task_state == "running" or self._engine_busy else "ready"
            self._live_value_note = f"{label} changed to {event.value}; {timing}."
            self.query_one("#run-log", RichLog).write(
                f"[bold #82aaff]{escape(label)}[/] → {escape(event.value)} [dim]({timing})[/]"
            )
            settings = self.query("#settings-text")
            if len(settings):
                settings.first(Static).update(self._settings_text())
            self._refresh_focus()

    def _show_view(self, view: str) -> None:
        if view not in _VIEW_BUTTONS:
            return
        switcher = self.query_one("#view-switcher", ContentSwitcher)
        switcher.current = view
        self._current_view = view
        for selector in _VIEW_BUTTONS.values():
            self.query_one(selector, Button).remove_class("active-nav")
        self.query_one(_VIEW_BUTTONS[view], Button).add_class("active-nav")
        if view == "history-view":
            self._refresh_tasks()
        elif view == "system-view":
            self._refresh_stats()
            self.query_one("#settings-text", Static).update(self._settings_text())

    def action_show_view(self, view: str) -> None:
        self._show_view(view)

    def action_toggle_loop(self) -> None:
        if self.screen.has_class("loop-collapsed"):
            self.screen.remove_class("loop-collapsed")
        else:
            self.screen.add_class("loop-collapsed")

    def action_toggle_todos(self) -> None:
        panel = self.query_one("#todo-panel", Collapsible)
        panel.collapsed = not panel.collapsed

    def action_focus_composer(self) -> None:
        self.query_one("#goal-input", Input).focus()

    def action_clear_activity(self) -> None:
        self.query_one("#run-log", RichLog).clear()
        self._activity_counts = {}
        self.query_one("#run-log", RichLog).write("[dim]Activity cleared. Work evidence and History are unchanged.[/]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "nav-activity": lambda: self._show_view("activity-view"),
            "nav-controls": lambda: self._show_view("controls-view"),
            "nav-evidence": lambda: self._show_view("evidence-view"),
            "nav-history": lambda: self._show_view("history-view"),
            "nav-system": lambda: self._show_view("system-view"),
            "apply-controls": self._apply_controls,
            "reset-controls": self._reset_controls,
            "open-library": self.action_history,
            "continue-mission": self._continue_selected_mission,
            "delete-mission": self._delete_selected_mission,
            "refresh-library": self.action_history,
            "add-todo": self._add_todo,
            "run-todo-sequence": self._run_todo_sequence,
            "complete-todo": self._complete_selected_todo,
            "save-learning": self._save_learning,
            "record-contest": self._record_contest_comment,
            "contest-stop": self.action_contest,
        }
        action = actions.get(event.button.id or "")
        if action:
            action()

    def _refresh_tasks(self) -> None:
        lines: list[str] = []
        records = []
        try:
            records = MissionStore().list(self.workspace, limit=20)
        except Exception as exc:
            lines = [f"[red]task history unavailable: {escape(str(exc))}[/]"]
        else:
            if not records:
                lines = ["[dim]no tasks yet — Xander is ready for orders[/]"]
            for mission in records:
                glyph, style = STATUS_GLYPHS.get(mission.status, ("·", "dim"))
                goal = " ".join(mission.goal.split())[:64]
                lines.append(
                    f"[{style}]{glyph}[/] {mission.id}  "
                    f"[dim]{mission.status} · {mission.phase}[/]  {escape(goal)}\n"
                    f"    [dim]result:[/] {escape(_short(mission.result, 100))}"
                )
                for moment in mission.timeline(limit=4):
                    kind = str(moment.get("kind", "thought")).upper()
                    message = escape(_short(str(moment.get("message", "")), 110))
                    lines.append(f"    [dim]{kind}:[/] {message}")
        try:
            library_matches = self.query("#mission-library")
        except Exception:
            library_matches = []
        if len(library_matches):
            select = library_matches.first(Select)
            previous = select.value
            options = [
                (f"{mission.status.upper()} · {_short(mission.goal, 72)}", mission.id)
                for mission in records
            ]
            select.set_options(options)
            ids = {value for _, value in options}
            if isinstance(previous, str) and previous in ids:
                select.value = previous
            elif options:
                select.value = options[0][1]
                self._selected_mission_id = options[0][1]
            else:
                select.clear()
                self._selected_mission_id = None
            self._update_library_detail(str(select.value) if isinstance(select.value, str) else None)
        self._fill_log("#tasks-log", lines)

    def _refresh_variants(self) -> None:
        lines: list[str] = []
        try:
            from .cli import army_payload

            report = army_payload()
        except Exception as exc:
            lines = [f"[red]army muster unavailable: {escape(str(exc))}[/]"]
        else:
            lines.append(
                f"[bold #ffcb6b]army of {report['size']}[/] · leader: [bold]{escape(report['leader'])}[/]"
            )
            for row in report["ranks"]:
                indent = "  " * int(row["depth"])
                marker = "[bold #ffcb6b]●[/]" if row["name"] == self.variant else "[dim]○[/]"
                lineage = f" ← {row['parent']}" if row.get("parent") else ""
                lines.append(
                    f"{indent}{marker} [bold]{escape(str(row['rank']))}[/] {escape(str(row['name']))} "
                    f"[dim]v{row['version']}{escape(lineage)} · {row['autonomy']} · "
                    f"lessons {row['lessons']} · wins {row.get('tasks_won', 0)}[/]"
                )
            lines.append("[dim]recruit with: xander variant clone <name> --from <source>[/]")
        self._fill_log("#variants-log", lines)

    def _refresh_stats(self) -> None:
        try:
            from .stats import render_lines, stats_payload

            lines = render_lines(stats_payload())
        except Exception as exc:
            lines = [f"[red]stats unavailable: {escape(str(exc))}[/]"]
        self._fill_log("#stats-log", lines)

    @work(thread=True, group="xander-meta")
    def _load_skills_panel(self) -> None:
        try:
            from .skills import SkillRegistry

            registry = SkillRegistry()
            report = registry.doctor()
            top = registry.list(limit=10, bucket="daily")
        except Exception as exc:
            lines = [f"[red]skill registry unavailable: {escape(str(exc))}[/]"]
        else:
            lines = [
                f"[bold #c3e88d]{report.get('indexed', 0)}[/] indexed skills · "
                f"[dim]daily {report.get('daily', 0)} · library {report.get('library', 0)} · "
                f"roots {len(report.get('roots', []))}[/]"
            ]
            for item in top:
                name = str(item.get("name", "?"))
                category = str(item.get("category") or item.get("bucket") or "")
                lines.append(f"  · {escape(name)} [dim]{escape(category)}[/]")
            lines.append("[dim](quartermaster) gear is grouped by hub and bounded by useful context, not a skill count[/]")
        self.call_from_thread(self._fill_log, "#skills-log", lines)

    def _start_chat(self, message: str) -> None:
        field = self.query_one("#goal-input", Input)
        field.value = ""
        if self._chat_busy:
            self._chat_queue.append(message)
            self.query_one("#run-log", RichLog).write(
                f"[dim]＋ queued conversation #{len(self._chat_queue)}:[/] {escape(_short(message, 100))}"
            )
            return
        history = [*self._conversation]
        self._conversation.append({"role": "user", "content": message})
        self._conversation = self._conversation[-24:]
        self._chat_busy = True
        self._focus_work = f"Talking with {self.variant}"
        self._focus_decision = "Conversation only · no work record created."
        self._focus_thought = "Reading your message and the current workspace context."
        self._focus_next = "Answer directly, or recognize when you are asking for autonomous work."
        self._refresh_focus()
        self._show_view("activity-view")
        self.query_one("#run-log", RichLog).write(
            f"[bold #82aaff]You → {escape(self.variant)}[/]  {escape(message)}"
        )
        self.narrator.record_chat("you", message)
        self._run_chat(message, history, self.variant)

    @work(thread=True, exclusive=True, group="xander-chat")
    def _run_chat(self, message: str, history: list[dict[str, str]], variant: str) -> None:
        try:
            result = talk(self.workspace, variant, message, history)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        self.call_from_thread(self._finish_chat, variant, result)

    def _finish_chat(self, variant: str, result: dict[str, Any]) -> None:
        self._chat_busy = False
        error = str(result.get("error") or "")
        if error:
            self._focus_thought = f"Conversation blocked: {error}"
            self._focus_next = "Check the selected clone's model route, then ask again."
            self.query_one("#run-log", RichLog).write(
                f"[bold #ff5370]{escape(variant)} could not answer[/] — {escape(_short(error, 300))}"
            )
            self.narrator.record_chat(variant, f"blocked: {error}")
        else:
            answer = str(result.get("text") or "").strip()
            stats = result.get("stats") if isinstance(result.get("stats"), dict) else {}
            model = str(stats.get("model") or result.get("role") or "language model")
            self._conversation.append({"role": "assistant", "content": answer})
            self._conversation = self._conversation[-24:]
            self._focus_thought = _short(answer, 150)
            self._focus_next = "Keep talking, or give a concrete work outcome and it will run automatically."
            self.query_one("#run-log", RichLog).write(
                f"[bold #f78c6c]{escape(variant)}[/] [dim]via {escape(model)}[/]\n{escape(answer)}"
            )
            self.narrator.record_chat(variant, answer)
        self._refresh_focus()
        self.query_one("#goal-input", Input).focus()
        if self._chat_queue:
            self._start_chat(self._chat_queue.pop(0))

    # -- orders ----------------------------------------------------------------
    def _handle_composer_command(self, value: str) -> bool:
        if not value.startswith("/"):
            return False
        command, _, argument = value.partition(" ")
        command = command.casefold()
        argument = argument.strip()
        field = self.query_one("#goal-input", Input)
        field.value = ""
        views = {
            "/activity": "activity-view",
            "/controls": "controls-view",
            "/evidence": "evidence-view",
            "/proof": "evidence-view",
            "/history": "history-view",
            "/xander": "system-view",
        }
        if command in views:
            self._show_view(views[command])
            return True
        actions = {
            "/loop": self.action_toggle_loop,
            "/todos": self.action_toggle_todos,
            "/pause": self.action_pause,
            "/resume": self.action_resume,
            "/stop": self.action_contest,
            "/cancel": self.action_cancel,
            "/new": self.action_new,
            "/clear": self.action_clear_activity,
            "/desktop": self.action_desktop,
        }
        if command in actions:
            actions[command]()
            return True
        if command in {"/ask", "/talk"}:
            if not argument:
                self.notify("Use /talk <message> to speak with the selected clone.", title="Conversation")
                return True
            self._start_chat(argument)
            return True
        if command == "/todo":
            if not argument:
                self.action_toggle_todos()
                return True
            self.query_one("#todo-input", Input).value = argument
            self._add_todo()
            return True
        if command == "/set":
            key, _, selected = argument.partition(" ")
            selectors = {
                "mode": "#mission-mode",
                "authority": "#mission-autonomy",
                "setup": "#mission-setup",
                "variant": "#mission-variant",
            }
            selector = selectors.get(key.casefold())
            if not selector or not selected.strip():
                self.notify("Use /set mode|authority|setup|variant <value>.", title="Live values")
                return True
            select = self.query_one(selector, Select)
            try:
                select.value = selected.strip()
            except Exception:
                self.notify(f"{selected.strip()} is not available for {key}.", title="Live values")
            return True
        if command == "/values":
            self.notify(
                f"mode {self.mode} · authority {self.autonomy or 'profile'} · "
                f"setup {self._mission_setup_policy} · variant {self.variant}",
                title="Live values",
            )
            return True
        if command in {"/help", "/how"}:
            self.action_help()
            return True
        self.notify(f"Unknown command: {command}. Type /help.", title="Composer")
        return True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.task_state == "power-zero":
            return
        if event.input.id == "todo-input":
            self._add_todo()
            return
        if event.input.id in {"learning-focus", "learning-sources"}:
            self._save_learning()
            return
        if event.input.id == "contest-input":
            self._record_contest_comment()
            return
        if event.input.id in {"mission-time", "mission-proof", "mission-allowed", "mission-constraints"}:
            self._apply_controls()
            return
        value = event.value.strip()
        if not value:
            return
        if event.input.id == "goal-input" and self._handle_composer_command(value):
            return
        if self._pending_approval is not None:
            self._answer_approval(value)
            return
        if self._pending_choice is not None:
            self._answer_question(value)
            return
        if self._retry_task_id and value.casefold() in {"retry", "try again"}:
            task_id = self._retry_task_id
            self._retry_task_id = None
            self.query_one("#goal-input", Input).value = ""
            self.task_state = "running"
            self.query_one("#run-log", RichLog).write(
                f"[bold #ffcb6b]▶ retrying[/] [dim]{escape(task_id)} in {escape(str(self.workspace))}[/]"
            )
            self._run_resume(task_id, [])
            return
        intent = parse_intent(value)
        goal_input = self.query_one("#goal-input", Input)
        if intent.kind == "help":
            goal_input.value = ""
            self.action_help()
            return
        if intent.kind == "mode":
            goal_input.value = ""
            self._set_mode(intent.argument)
            return
        if intent.kind == "chdir":
            goal_input.value = ""
            self._rebind_workspace(intent)
            return
        if intent.kind == "selfwork":
            goal_input.value = ""
            target = _xander_workspace()
            if self.workspace != target:
                if self.task_state == "running" or self._engine_busy:
                    self._order_queue.append(
                        {"goal": value, "mode": self.mode, "workspace": str(target)}
                    )
                    self.query_one("#run-log", RichLog).write(
                        f"[dim]＋ queued self-work in {escape(str(target))}:[/] "
                        f"{escape(_short(value, 100))}"
                    )
                    return
                self._rebind_workspace(Intent(kind="chdir", argument=str(target)))
                self.query_one("#run-log", RichLog).write(
                    "[bold #89ddff]self-target detected[/] working in Xander's checkout"
                )
            if self.task_state == "running" or self._engine_busy:
                self._order_queue.append({"goal": value, "mode": self.mode})
                return
            self._dispatch(value, self.mode)
            return
        if intent.kind == "feedback":
            # Feedback lands immediately — even mid-run — and never queues as work.
            goal_input.value = ""
            self._record_feedback(value)
            return
        if intent.kind == "advice" or self.mode == "answer":
            self._start_chat(value)
            return
        engine_mode = self.mode
        if self.task_state == "running" or self._engine_busy:
            self._order_queue.append({"goal": value, "mode": engine_mode})
            goal_input.value = ""
            self.query_one("#run-log", RichLog).write(
                f"[dim]＋ queued order #{len(self._order_queue)}:[/] {escape(_short(value, 100))}"
            )
            return
        goal_input.value = ""
        self._dispatch(value, engine_mode)

    def _approve_action(self, action: Any, reason: str) -> bool:
        decision = Event()
        pending = {
            "event": decision,
            "approved": False,
            "action": action,
            "reason": reason,
        }
        try:
            self.call_from_thread(self._show_action_approval, pending)
        except Exception:
            return False
        try:
            from textual.worker import get_current_worker

            worker = get_current_worker()
        except Exception:
            worker = None
        while not decision.wait(0.1):
            if worker is not None and worker.is_cancelled:
                return False
        return bool(pending.get("approved"))

    def _show_action_approval(self, pending: dict[str, Any]) -> None:
        self._pending_approval = pending
        self.task_state = "approval"
        action = pending.get("action")
        data = action.model_dump(mode="json") if hasattr(action, "model_dump") else {}
        kind = str(data.get("kind") or "action")
        target = str(data.get("path") or " ".join(data.get("argv") or []) or data.get("expected") or "step")
        reason = str(pending.get("reason") or "this action needs approval")
        log = self.query_one("#run-log", RichLog)
        log.write(
            f"[bold yellow]Approval needed[/] — {escape(kind)} {escape(_short(target, 130))}\n"
            f"[dim]{escape(_short(reason, 150))} · type yes or no[/]"
        )
        self._focus_thought = f"Waiting for approval: {kind}"
        self._focus_next = "Type yes to run this action once, or no to stop it."
        self._refresh_focus()
        self.query_one("#goal-input", Input).focus()

    def _resolve_pending_approval(self, approved: bool) -> bool:
        pending = self._pending_approval
        if pending is None:
            return False
        pending["approved"] = approved
        self._pending_approval = None
        self.task_state = "running"
        pending["event"].set()
        self.query_one("#run-log", RichLog).write(
            f"[bold #c3e88d]approval {'granted' if approved else 'denied'}[/]"
        )
        return True

    def _answer_approval(self, value: str) -> None:
        choice = value.strip().casefold()
        if choice in {"yes", "y", "approve", "allow", "1"}:
            self._resolve_pending_approval(True)
        elif choice in {"no", "n", "deny", "decline", "2"}:
            self._resolve_pending_approval(False)
        else:
            self.query_one("#run-log", RichLog).write(
                "[bold yellow]?[/] approval expects yes or no"
            )
        self.query_one("#goal-input", Input).value = ""

    def _answer_question(self, value: str) -> None:
        pending = self._pending_choice or {}
        options: list[str] = pending.get("options", [])
        tokens = [token for token in value.replace(",", " ").split() if token]
        selected: list[str] = []
        for token in tokens:
            if token.isdigit() and 1 <= int(token) <= len(options):
                selected.append(options[int(token) - 1])
            elif token in options:
                selected.append(token)
            else:
                self.query_one("#run-log", RichLog).write(
                    f"[bold yellow]?[/] unknown option {escape(token)} — answer with numbers or ids"
                )
                return
        self._pending_choice = None
        self.query_one("#goal-input", Input).value = ""
        self.task_state = "running"
        self.query_one("#run-log", RichLog).write(
            f"[bold #ffcb6b]▶ decision received[/] [dim]{escape(', '.join(selected))}[/]"
        )
        self._run_resume(str(pending.get("task_id", "")), selected)

    def action_new(self) -> None:
        if self.task_state == "power-zero":
            return
        self._resolve_pending_approval(False)
        if self.task_state == "running" or self._engine_busy:
            self.action_cancel()
        if self._pending_choice is not None:
            pending_task = self._pending_choice.get("task_id", "")
            self._pending_choice = None
            self.query_one("#run-log", RichLog).write(
                f"[dim]question dismissed — resume later with: xander resume {escape(str(pending_task))} --select <id>[/]"
            )
        self.goal = ""
        self._focus_work = "Waiting for an outcome"
        self._focus_thought = "Give Xander one outcome to understand and prove."
        self._focus_decision = "No decision yet."
        self._focus_change = "Nothing changed yet."
        self._focus_next = "Describe the result you want, then press Enter."
        self._activity_counts = {}
        self._retry_task_id = None
        self.task_id = "new"
        self.phase_index = 0
        self.task_state = "idle"
        self._current_guide = None
        self._known_guide_step_ids = set()
        self._live_value_note = "Ready for an outcome."
        goal_input = self.query_one("#goal-input", Input)
        goal_input.value = ""
        self._reset_controls()
        goal_input.focus()
        self._render_guide(None)
        self._show_view("activity-view")

    def action_cancel(self) -> None:
        self._resolve_pending_approval(False)
        workers = [worker for worker in self.workers if worker.group == "xander-task"]
        if not workers and not self._engine_busy and self.task_state != "running":
            return
        for worker in workers:
            worker.cancel()
        self._order_queue.clear()
        self._todo_sequence_active = False
        self._release_todo("todo", "operator cancelled before verification")
        self._retry_task_id = None
        self.task_state = "paused"
        self._flush_activity()
        self._activity_counts = {}
        self._focus_thought = "The active work was cancelled by the operator."
        self._focus_next = "Start new work, or return to History to resume an older run."
        self._refresh_focus()
        self.query_one("#run-log", RichLog).write(
            "[bold #ff5370]■ active work cancelled[/] [dim]— the current atomic step may wind down; no queued work will start[/]"
        )
        self.query_one("#goal-input", Input).focus()

    def _open_contest_editor(self) -> None:
        self._show_view("controls-view")
        self.query_one("#contest-panel", Collapsible).collapsed = False
        focus = self.query_one("#contest-input", Input).focus
        focus()
        self.call_after_refresh(focus)

    def action_contest(self) -> None:
        self._resolve_pending_approval(False)
        workers = [worker for worker in self.workers if worker.group == "xander-task"]
        if not workers and not self._engine_busy and self.task_state != "running":
            self._open_contest_editor()
            self.notify("No active AI work; enter a contest/comment below.", title="Contest/comment")
            return
        for worker in workers:
            worker.cancel()
        self._order_queue.clear()
        self._todo_sequence_active = False
        self._release_todo("todo", "operator stopped work for contest")
        self._retry_task_id = self.task_id if self.task_id != "new" else None
        self.task_state = "needs-attention"
        self._flush_activity()
        self._activity_counts = {}
        self._focus_thought = "Work stopped so the operator can contest the approach."
        self._focus_next = "Write a contest/comment, then resume only after the direction is clear."
        self._refresh_focus()
        self.query_one("#run-log", RichLog).write(
            "[bold #ff5370]■ AI work stopped for contest[/] [dim]the active TODO remains pending; comment before resuming[/]"
        )
        self._open_contest_editor()

    def action_run(self) -> None:
        if self.task_state == "power-zero":
            return
        goal_input = self.query_one("#goal-input", Input)
        goal = (goal_input.value or self.goal).strip()
        if not goal or self.task_state == "running":
            return
        if self._engine_busy:
            self.query_one("#run-log", RichLog).write(
                "[yellow]previous engine run is still winding down — order queued[/]"
            )
            self._order_queue.append({"goal": goal, "mode": self.mode})
            goal_input.value = ""
            return
        self._dispatch(goal, self.mode)

    def _dispatch(self, goal: str, engine_mode: str) -> None:
        if self.task_state == "power-zero":
            return
        if not self._capture_control_inputs():
            return
        if engine_mode != "answer":
            engine_mode = self.mode
        self._active_run_mode = engine_mode
        self.goal = goal
        self._current_guide = None
        self._known_guide_step_ids = set()
        self._focus_work = goal
        self._focus_thought = "Starting with local context and a bounded plan."
        self._focus_decision = "Choose a bounded path from local evidence."
        self._focus_change = "No changes in this outcome yet."
        self._focus_next = "Understand → Plan → Work → Proof."
        self._activity_counts = {}
        self._refresh_focus()
        self._retry_task_id = None
        self.task_state = "running"
        self.phase_index = 0
        authority = self.autonomy or "profile"
        self.query_one("#run-log", RichLog).write(
            f"[bold #ffcb6b]Autonomous work accepted[/] — {escape(goal)}\n"
            f"[dim]clone {escape(self.variant)} · mode {escape(engine_mode)} · autonomy {escape(authority)}[/]"
        )
        self._show_view("activity-view")
        self._refresh_focus()
        self._run_goal(goal, engine_mode)

    def _set_mode(self, wheel: str) -> None:
        run_log = self.query_one("#run-log", RichLog)
        if wheel not in MODE_REQUESTS:
            run_log.write(
                "[yellow]modes:[/] "
                + "   ".join(f"[bold]{name}[/] [dim]{MODE_HELP[name]}[/]" for name in MODE_REQUESTS)
            )
            return
        engine_mode, autonomy = MODE_REQUESTS[wheel]
        self.mode = engine_mode
        self.autonomy = autonomy
        self._refresh_focus()
        run_log.write(f"[bold #ffcb6b]mode → {wheel}[/] [dim]{MODE_HELP[wheel]}[/]")

    def _record_feedback(self, text: str) -> None:
        run_log = self.query_one("#run-log", RichLog)
        kind = "dislike"
        if re.search(r"\b(?:like|love|prefer|always|more)\b", text, re.IGNORECASE) and not re.search(
            r"\b(?:do\s*n[o']t\s+like|dislike|hate|never|stop|less)\b", text, re.IGNORECASE
        ):
            kind = "like"
        try:
            from .memory import MemoryStore

            namespace = self.variant
            try:
                from .variants import load_variant

                namespace = load_variant(self.variant).memory_namespace or self.variant
            except Exception:
                pass
            MemoryStore(namespace=namespace).add_preference(text, kind=kind, source="explicit")
            from .commentary import Commentator

            line = Commentator(voice="quiet").say("feedback_ack", text=text)
            run_log.write(f"[italic #f78c6c]❝ {escape(line)}[/]")
            self.narrator.record(f"preference recorded ({kind}): {text}", task_id=self.task_id)
        except Exception as exc:
            run_log.write(f"[red]could not record that preference: {escape(str(exc))}[/]")

    def _rebind_workspace(self, intent: Intent) -> None:
        run_log = self.query_one("#run-log", RichLog)
        if self.task_state == "running" or self._engine_busy:
            run_log.write("[yellow]workspace stays put mid-run — I'll move after this task lands[/]")
            return
        target = resolve_target(intent.argument, self.workspace)
        try:
            if not target.exists() and intent.create:
                target.mkdir(parents=True)
            target = target.resolve(strict=True)
        except OSError as exc:
            run_log.write(f"[red]can't rebind the workspace: {escape(str(exc))}[/]")
            return
        if not target.is_dir():
            run_log.write(f"[red]not a directory: {escape(str(target))}[/]")
            return
        self.workspace = target
        self._workboard = self._workboard_store.load(self.workspace)
        self._selected_todo_id = None
        self.narrator = Narrator(variant=self.variant, workspace=self.workspace)
        self._branch, self._dirty_count = _repository_status(self.workspace)
        self._refresh_focus()
        run_log.write(f"[bold #ffcb6b]workspace →[/] {escape(str(target))}")
        self._refresh_workboard(load_inputs=True)
        self._refresh_tasks()
        self._refresh_focus()

    @work(thread=True, exclusive=True, group="xander-task")
    def _run_goal(self, goal: str, engine_mode: str | None = None) -> None:
        self._invoke_and_finish(
            engine_mode or self.mode,
            goal=goal,
            autonomy=self.autonomy if self.caller == "human" else "proposal-only",
            constraints=[*self._mission_constraints, *self._learning_constraints()],
            acceptance_checks=self._mission_acceptance_checks,
            allowed_paths=self._mission_allowed_paths,
            timeout=self._mission_timeout,
            setup_policy=self._mission_setup_policy,
            log_events=False,
        )

    @work(thread=True, exclusive=True, group="xander-task")
    def _run_resume(self, task_id: str, selected: list[str]) -> None:
        self._invoke_and_finish(
            "resume",
            task_id=task_id,
            selected_options=selected,
            autonomy=self.autonomy if self.caller == "human" else "proposal-only",
        )

    def _invoke_and_finish(self, mode: str, **request: Any) -> None:
        from textual.worker import get_current_worker

        worker = get_current_worker()

        def sink(event: Any) -> None:
            if not worker.is_cancelled:
                self.call_from_thread(self._append_event, event)

        self._engine_busy = True
        try:
            result = invoke_engine(
                mode,
                workspace=self.workspace,
                variant=self.variant,
                caller=self.caller,
                event_sink=sink,
                approve=self._approve_action if self.caller == "human" else None,
                **request,
            )
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        finally:
            self._engine_busy = False
        if not worker.is_cancelled:
            self.call_from_thread(self._finish, result)

    def _append_event(self, event: Any) -> None:
        payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else event
        if not isinstance(payload, dict):
            payload = {"message": str(payload)}
        phase = str(payload.get("phase", "")).casefold()
        if phase in _PHASE_INDEXES:
            self.phase_index = _PHASE_INDEXES[phase]
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        guide = data.get("guide") if isinstance(data.get("guide"), dict) else None
        if guide:
            self._render_guide(guide)
        self._update_focus(payload)
        channel, narrated = self.narrator.narrate(payload)
        line = self._human_event_line(payload, narrated)
        if not line:
            return
        if channel != "run":
            self.query_one(_CHANNELS[channel], RichLog).write(line)
        self._write_activity(payload, line)

    def _finish(self, result: dict[str, Any]) -> None:
        task = result.get("task") if isinstance(result.get("task"), dict) else {}
        handoff = result.get("handoff") if isinstance(result.get("handoff"), dict) else {}
        self.task_id = str(result.get("task_id") or task.get("id") or handoff.get("task_id") or self.task_id)
        status = result.get("status") or task.get("status") or handoff.get("status")
        successful = bool(result.get("ok")) or str(status or "").casefold() in {
            "complete",
            "completed",
        }
        run_log = self.query_one("#run-log", RichLog)
        options = [
            option
            for option in ((task.get("plan") or {}).get("options") or [])
            if isinstance(option, dict) and option.get("id")
        ]
        if str(status or "").casefold() == "waiting_approval" and options:
            self._retry_task_id = None
            self._ask_question(options)
            self._refresh_tasks()
            return
        self.task_state = "complete" if successful else "needs-attention"
        self._retry_task_id = None if successful else self.task_id
        self.phase_index = 7
        self._refresh_focus()
        self._update_focus(
            {
                "type": "result",
                "status": status or ("completed" if successful else "needs-attention"),
                "message": "goal verified" if successful else "task needs attention",
                "data": result,
            }
        )
        guide = task.get("guide") if isinstance(task.get("guide"), dict) else None
        if guide:
            self._render_guide(guide)
        self._flush_activity()
        self._activity_counts = {}
        for line in self.narrator.summarize(result):
            run_log.write(line)
        lesson = str(task.get("lesson") or "")
        if lesson:
            self._workboard_store.add_observed_lesson(self._workboard, lesson)
            run_log.write(f"[bold #c3e88d]learning recorded automatically[/] {escape(_short(lesson, 150))}")
        if self._todo_sequence_active and self._active_todo_id:
            if successful:
                self._release_todo("done", lesson or "verified task result")
                run_log.write("[bold #c3e88d]TODO verified[/] advancing to the next item")
                self._refresh_workboard()
                if self._start_next_todo():
                    return
                self._todo_sequence_active = False
                run_log.write("[bold #c3e88d]TODO sequence complete[/] all pending items are verified")
            else:
                self._release_todo("blocked", str(task.get("failure") or "task did not verify"))
                self._todo_sequence_active = False
                run_log.write("[bold #ff5370]TODO sequence stopped[/] the failed item is blocked for review")
        self._refresh_workboard()
        self._refresh_tasks()
        self._refresh_stats()
        if self._order_queue:
            self._start_next_order()
        else:
            self._ask_where_next(successful)

    def _ask_question(self, options: list[dict[str, Any]]) -> None:
        self._pending_choice = {"task_id": self.task_id, "options": [str(option["id"]) for option in options]}
        self.task_state = "question"
        run_log = self.query_one("#run-log", RichLog)
        run_log.write("[bold yellow]? Here are the directions I see. Where do you want us to go?[/]")
        run_log.write("[dim]Answer with numbers or ids; I won’t change anything until you choose.[/]")
        for index, option in enumerate(options, start=1):
            title = _short(str(option.get("title") or option["id"]), 40)
            summary = _short(str(option.get("summary") or ""), 80)
            recommendation = " [bold green]← my pick[/]" if option.get("selected_by_default") else ""
            run_log.write(f"  [bold]\\[{index}][/] {escape(title)}{recommendation} [dim]{escape(summary)}[/]")
        self.query_one("#goal-input", Input).focus()

    def _ask_where_next(self, successful: bool) -> None:
        if successful:
            message = f"Where should we go next in {self.workspace}? Give me the next outcome."
        else:
            message = f"I’m still in {self.workspace}. Say retry, or tell me what direction to change."
        self.query_one("#run-log", RichLog).write(f"[bold yellow]?[/] {escape(message)}")
        self.narrator.record(message, task_id=self.task_id)
        self.query_one("#goal-input", Input).focus()

    def _start_next_order(self) -> None:
        if not self._order_queue or self.task_state == "running" or self._engine_busy:
            return
        entry = self._order_queue.pop(0)
        if isinstance(entry, str):  # legacy plain-text queue entries
            entry = {"goal": entry, "mode": self.mode}
        queued_workspace = str(entry.get("workspace") or "").strip()
        if queued_workspace:
            self._rebind_workspace(Intent(kind="chdir", argument=queued_workspace))
        self.query_one("#run-log", RichLog).write(f"[dim]▶ next from queue ({len(self._order_queue)} left)[/]")
        self._dispatch(str(entry["goal"]), str(entry.get("mode") or self.mode))

    def action_pause(self) -> None:
        workers = [worker for worker in self.workers if worker.group == "xander-task"]
        for worker in workers:
            worker.cancel()
        if workers:
            self.task_state = "paused"
            self._focus_thought = "The active work is paused by the operator."
            self._focus_next = "Resume when you are ready to continue this run."
            self._refresh_focus()
            self.query_one("#run-log", RichLog).write(
                "[yellow]paused by operator[/] [dim]— narration stops; the current engine step "
                "winds down in the background before resume can start[/]"
            )

    def action_resume(self) -> None:
        if self.goal and self.task_state == "paused":
            self.task_state = "idle"
            self.action_run()
        elif self._retry_task_id and self.task_state == "needs-attention":
            task_id = self._retry_task_id
            self._retry_task_id = None
            self.task_state = "running"
            self._run_resume(task_id, [])

    def action_tests(self) -> None:
        self._show_view("evidence-view")

    def action_history(self) -> None:
        self._refresh_tasks()
        self._show_view("history-view")
        focus = self.query_one("#mission-library", Select).focus
        focus()
        self.call_after_refresh(focus)

    def action_soul(self) -> None:
        self._refresh_stats()
        self._show_view("system-view")

    def action_desktop(self) -> None:
        self._show_view("activity-view")
        self.query_one("#run-log", RichLog).write("[bold #89ddff]desktop[/] explicit observation requested")
        self._capture_desktop()

    @work(thread=True, exclusive=True, group="xander-desktop")
    def _capture_desktop(self) -> None:
        try:
            from .desktop import capture_desktop

            path = capture_desktop()
            message = f"[bold #c3e88d]desktop snapshot saved[/] {escape(str(path))}"
        except Exception as exc:
            message = f"[yellow]desktop snapshot unavailable:[/] {escape(str(exc))}"
        self.call_from_thread(self._write_desktop_message, message)

    def _write_desktop_message(self, message: str) -> None:
        self.query_one("#run-log", RichLog).write(message)

    def _poll_power(self) -> None:
        if self._power_guard.tripped:
            return
        status = self._power_guard.reader()
        self._power_status = status
        self._refresh_focus()
        if status.capacity == 0:
            self.action_cancel()
            self.task_state = "power-zero"
            self._focus_thought = "Power reached 0%; Xander stopped to protect the machine."
            self._focus_next = "Reconnect power before starting more work."
            self._refresh_focus()
            self.query_one("#goal-input", Input).disabled = True
            self._power_guard.trip(status)
            self.query_one("#run-log", RichLog).write(
                "[bold #ff5370]POWER ZERO[/] Xander stopped; the PC shutdown request has been sent."
            )

    @work(thread=True, exclusive=True, group="xander-power")
    def _shutdown_power(self) -> None:
        from .power import request_poweroff

        request_poweroff()

    def action_help(self) -> None:
        self.notify(
            escape(
                "Alt works while you type.  "
                "⌥1-5 or ⌥A activity ⌥C controls ⌥E evidence ⌥H history ⌥X Xander  ·  "
                "⌥B rail  ⌥O pinned  ⌥K clear  ⌥L composer  ·  "
                "⌥N new  ⌥P pause  ⌥R resume  ⌥S stop  ⌥Q quit"
            ),
            title="Keys",
        )


def run_tui(
    *,
    workspace: Path,
    variant: str = "default",
    caller: str = "human",
    autonomy: str | None = None,
    mode: str = "implement",
    model: str = "auto",
) -> None:
    XanderApp(
        workspace=workspace,
        variant=variant,
        caller=caller,
        autonomy=autonomy,
        mode=mode,
        model=model,
    ).run()
