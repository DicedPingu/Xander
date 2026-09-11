"""Xander's terminal: one feed, one composer, nothing else.

There are no tabs. Everything Xander does — what you said, what he did,
what he asks, what he answers — lands in one scrollable feed in the order
it happened. The feed is ordinary text: select it with the mouse and
Ctrl+C copies it. The composer keeps a history: ↑ and ↓ walk it.

Every line you type is shown in the feed before anything happens to it,
including lines typed while a mission is running; those are handed to the
engine at its next safe point instead of being lost.
"""

from __future__ import annotations

import re
from pathlib import Path
from threading import Event, Lock
from typing import Any

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Input, Static

from .cli import invoke_engine
from .conversation import talk
from .intents import Intent, MODE_HELP, MODE_REQUESTS, MODES, command_help, parse_intent, resolve_target
from .mission import MissionStore
from .narrator import Narrator, _short
from .power import PowerStatus, PowerZeroGuard, read_power_status
from .workboard import WorkboardStore

_YOU = "#82aaff"
_XANDER = "#f78c6c"
_OK = "#c3e88d"
_WARN = "#ffcb6b"
_BAD = "#ff5370"
_DIM = "#6b7280"
_ASK = "#c792ea"

# Phase names → what the operator sees while it happens. Short, present tense.
_PHASE_TEXT = {
    "analyze": "looking at the workspace",
    "research": "reading up",
    "set_up": "getting ready",
    "work": "working",
    "test": "checking",
    "judge_log": "judging the result",
    "learn": "noting the lesson",
    "repeat": "trying a different approach",
}
_STOP_WORDS = re.compile(r"^\s*(?:stop|cancel|abort|halt|quit that|never mind|nevermind)\b", re.IGNORECASE)


def _repository_status(workspace: Path) -> tuple[str, int]:
    import subprocess

    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=workspace, capture_output=True, text=True, timeout=5,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return "", 0
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=workspace, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=workspace, capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()
        return branch or "detached", len(dirty)
    except (OSError, subprocess.SubprocessError):
        return "", 0


def _xander_workspace() -> Path:
    return Path(__file__).resolve().parent.parent


def _wheel_for(engine_mode: str, autonomy: str | None) -> str:
    if engine_mode == "answer":
        return "ask"
    if engine_mode == "plan":
        return "plan"
    return "yolo" if autonomy == "full-auto" else "build"


class Composer(Input):
    """The input line, with ↑/↓ history and Shift+Tab as the mode wheel."""

    BINDINGS = [
        Binding("up", "history_back", "Previous", show=False),
        Binding("down", "history_forward", "Next", show=False),
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.history: list[str] = []
        self._cursor = 0
        self._stash = ""

    def remember(self, text: str) -> None:
        if text and (not self.history or self.history[-1] != text):
            self.history.append(text)
            self.history = self.history[-200:]
        self._cursor = len(self.history)
        self._stash = ""

    def action_history_back(self) -> None:
        if not self.history:
            return
        if self._cursor == len(self.history):
            self._stash = self.value
        if self._cursor > 0:
            self._cursor -= 1
            self.value = self.history[self._cursor]
            self.cursor_position = len(self.value)

    def action_history_forward(self) -> None:
        if not self.history or self._cursor >= len(self.history):
            return
        self._cursor += 1
        self.value = self.history[self._cursor] if self._cursor < len(self.history) else self._stash
        self.cursor_position = len(self.value)


class XanderApp(App[None]):
    CSS = f"""
    Screen {{ layout: vertical; background: #0e1117; }}
    #status {{ height: 1; padding: 0 1; background: #161b22; color: #c9d1d9; }}
    #feed {{ height: 1fr; padding: 0 1; scrollbar-size: 1 1; }}
    #feed > Static {{ margin: 0 0 0 0; }}
    .you {{ color: {_YOU}; }}
    .xander {{ color: #e6edf3; }}
    .dim {{ color: {_DIM}; }}
    .ok {{ color: {_OK}; }}
    .warn {{ color: {_WARN}; }}
    .bad {{ color: {_BAD}; }}
    .ask {{ color: {_ASK}; }}
    .gap {{ height: 1; }}
    #composer-row {{ height: 1; background: #161b22; }}
    #prompt {{ width: 3; padding: 0 1; color: {_XANDER}; background: #161b22; text-style: bold; }}
    #composer {{ border: none; background: #161b22; color: #e6edf3; padding: 0; height: 1; width: 1fr; }}
    #composer:focus {{ border: none; }}
    #hint {{ height: 1; padding: 0 1; color: {_DIM}; background: #0e1117; }}
    """

    BINDINGS = [
        Binding("shift+tab", "cycle_mode", "Mode", show=False, priority=True),
        Binding("escape", "focus_composer", "Composer", show=False, priority=True),
        Binding("ctrl+k", "clear_feed", "Clear", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit", show=False, priority=True),
        Binding("f1", "help", "Help", show=False, priority=True),
    ]

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
        self.goal = ""
        self._draft = ""
        self._conversation: list[dict[str, str]] = []
        self._chat_busy = False
        self._chat_queue: list[str] = []
        self._engine_busy = False
        self._order_queue: list[dict[str, str]] = []
        self._pending_choice: dict[str, Any] | None = None
        self._pending_approval: dict[str, Any] | None = None
        self._retry_task_id: str | None = None
        self._steering_lock = Lock()
        self._steering_notes: list[str] = []
        self._abort_requested = False
        self._last_result: dict[str, Any] | None = None
        self._thinking: Static | None = None
        self._todo_sequence_active = False
        self._active_todo_id: str | None = None
        self._entries = 0
        self._workboard_store = WorkboardStore()
        self._workboard = self._workboard_store.load(self.workspace)
        self.narrator = Narrator(variant=variant, workspace=self.workspace)
        self._branch, self._dirty_count = _repository_status(self.workspace)
        self._power_status = PowerStatus(None)
        self._power_guard = PowerZeroGuard(reader=power_reader, shutdown=power_shutdown or self._shutdown_power)

    # -- layout ----------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static("", id="status")
        yield VerticalScroll(id="feed")
        with Horizontal(id="composer-row"):
            yield Static("›", id="prompt")
            yield Composer(placeholder="tell Xander what you want · /help", id="composer")
        yield Static("", id="hint")

    def on_mount(self) -> None:
        self._refresh_status()
        self.say(
            f"[bold]Xander[/] in [bold]{escape(str(self.workspace))}[/]  "
            f"[dim]· plain orders run · questions get answers · /help for commands[/]",
            "dim",
        )
        self._poll_power()
        self.set_interval(30, self._poll_power)
        self.query_one("#composer", Composer).focus()

    # -- feed ------------------------------------------------------------------
    def say(self, markup: str, style: str = "xander") -> Static:
        """Append one selectable entry to the feed and keep it scrolled to the end."""

        feed = self.query_one("#feed", VerticalScroll)
        entry = Static(markup, classes=style, markup=True)
        feed.mount(entry)
        self._entries += 1
        feed.scroll_end(animate=False)
        return entry

    def say_you(self, text: str) -> None:
        self.say(f"[bold {_YOU}]you ›[/] {escape(text)}", "you")

    def say_xander(self, text: str) -> None:
        self.say(f"[bold {_XANDER}]Xander ›[/] {escape(text)}", "xander")

    def _refresh_status(self) -> None:
        wheel = _wheel_for(self.mode, self.autonomy)
        repo = f" · {self._branch}" + (f" ~{self._dirty_count}" if self._dirty_count else "") if self._branch else ""
        power = f" · {self._power_status.label}" if self._power_status.capacity is not None else ""
        state = {
            "idle": "ready",
            "running": "working…",
            "approval": "needs your yes/no",
            "question": "needs your choice",
            "complete": "done",
            "needs-attention": "stuck",
            "paused": "paused",
            "power-zero": "power off",
        }.get(self.task_state, self.task_state)
        self.query_one("#status", Static).update(
            f"[bold {_XANDER}]XANDER[/] [dim]{escape(self.variant)}[/] · {escape(_short(str(self.workspace), 70))}"
            f"{escape(repo)} · [bold]{wheel}[/] · {escape(state)}{escape(power)}"
        )
        self.query_one("#hint", Static).update(
            f"{wheel} — {MODE_HELP[wheel]}   ·   shift+tab mode · ↑↓ history · ctrl+c copies selection · ctrl+q quit"
        )

    def watch_task_state(self, _: str) -> None:
        if self.is_mounted:
            self._refresh_status()

    # -- input -----------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.task_state == "power-zero":
            return
        composer = self.query_one("#composer", Composer)
        value = event.value.strip()
        composer.value = ""
        if not value:
            return
        composer.remember(value)
        self.say_you(value)
        self.handle_line(value)

    def handle_line(self, value: str) -> None:
        """Route one typed line. Everything typed is visible; nothing is dropped."""

        if value.startswith("/"):
            self._route_command(parse_intent(value))
            return
        if self._pending_approval is not None:
            self._answer_approval(value)
            return
        if self._pending_choice is not None:
            self._answer_question(value)
            return
        if self._retry_task_id and value.casefold() in {"retry", "try again", "again"}:
            task_id, self._retry_task_id = self._retry_task_id, None
            self.task_state = "running"
            self.say(f"[dim]retrying {escape(task_id)}[/]", "dim")
            self._run_resume(task_id, [])
            return
        intent = parse_intent(value)
        if self.task_state == "running" or self._engine_busy:
            self._steer(value, intent)
            return
        self._route_intent(value, intent)

    def _route_intent(self, value: str, intent: Intent) -> None:
        kind = intent.kind
        if kind == "chdir":
            self._rebind_workspace(intent)
        elif kind == "selfwork":
            target = _xander_workspace()
            if self.workspace != target:
                self._rebind_workspace(Intent(kind="chdir", argument=str(target)))
                self.say("[dim]working on myself, in my own checkout[/]", "dim")
            self._dispatch(value, self.mode if self.mode != "answer" else "implement")
        elif kind == "feedback":
            self._record_feedback(value)
        elif kind == "discuss":
            self._start_discussion(value)
        elif kind == "advice" or self.mode == "answer":
            self._start_chat(value)
        elif kind == "draft":
            self._draft = value
            if self.mode == "plan":
                self._dispatch(value, "plan")
            else:
                self.say(
                    "[dim]I'm not sure whether that's an order or a thought. "
                    "[bold]/work[/] runs it · [bold]/discuss[/] talks it through[/]",
                    "dim",
                )
        else:
            self._dispatch(value, self.mode if self.mode != "answer" else "implement")

    def _steer(self, value: str, intent: Intent) -> None:
        """A line typed mid-run: shown now, applied at the engine's next safe point."""

        if _STOP_WORDS.match(value):
            self._abort_requested = True
            self.say("[dim]heard — stopping at the next safe point[/]", "dim")
            return
        if intent.kind in {"order", "selfwork"} and not any(
            word in value.casefold() for word in ("instead", "also", "don't", "dont", "not ", "use ", "only")
        ):
            self._order_queue.append({"goal": value, "mode": self.mode})
            self.say(f"[dim]queued as the next order (#{len(self._order_queue)})[/]", "dim")
            return
        with self._steering_lock:
            self._steering_notes.append(value)
        self.say("[dim]heard — I'll apply that at the next safe point[/]", "dim")

    def _drain_steering(self) -> dict[str, Any]:
        with self._steering_lock:
            notes, self._steering_notes = self._steering_notes, []
            abort, self._abort_requested = self._abort_requested, False
        return {
            "lines": [f"you said: {note}" for note in notes],
            "constraints": notes,
            "abort": abort,
        }

    # -- commands --------------------------------------------------------------
    def _route_command(self, intent: Intent) -> None:
        kind = intent.kind
        argument = intent.argument
        if kind == "unknown_command":
            self.say(f"[{_WARN}]{escape(intent.reason)}[/]", "warn")
        elif kind == "help":
            self.action_help()
        elif kind == "mode":
            self._set_mode(argument)
        elif kind == "chdir":
            self._rebind_workspace(intent)
        elif kind == "discuss":
            topic = argument or self._draft
            if not topic:
                self.say("[dim]/discuss <topic> — or just describe the idea[/]", "dim")
                return
            self._start_discussion(topic)
        elif kind == "work":
            self._authorize_work(argument)
        elif kind == "research":
            if not argument:
                self.say("[dim]/research <URL or topic>[/]", "dim")
                return
            self._dispatch(argument, "research")
        elif kind == "goal":
            self._store_goal(argument)
        elif kind == "todo":
            self._add_todo(argument)
        elif kind == "talk":
            if not argument:
                self.say("[dim]/talk <message>[/]", "dim")
                return
            self._start_chat(argument)
        elif kind == "command":
            self._composer_command(intent)
        else:
            self.say(f"[{_WARN}]{escape(intent.command or '/')} has no handler[/]", "warn")

    def _composer_command(self, intent: Intent) -> None:
        name = intent.command
        argument = intent.argument
        if name in {"/history", "/missions"}:
            self._show_history()
        elif name in {"/evidence", "/proof", "/show"}:
            self._show_evidence(argument)
        elif name in {"/todo", "/todos"}:
            if argument.strip().casefold() in {"run", "go", "start"}:
                self._run_todo_sequence()
            elif argument:
                self._add_todo(argument)
            else:
                self._show_todos()
        elif name == "/pause":
            self.action_pause()
        elif name == "/resume":
            self.action_resume()
        elif name in {"/stop", "/cancel"}:
            self.action_cancel()
        elif name == "/new":
            self.action_new()
        elif name == "/clear":
            self.action_clear_feed()
        elif name == "/desktop":
            self._capture_desktop()
        elif name == "/values":
            self.say(
                f"[dim]clone {escape(self.variant)} · mode {escape(_wheel_for(self.mode, self.autonomy))} · "
                f"autonomy {escape(self.autonomy or 'profile')} · workspace {escape(str(self.workspace))}[/]",
                "dim",
            )
        elif name == "/set":
            parts = argument.split(None, 1)
            if len(parts) == 2 and parts[0] == "mode":
                self._set_mode(parts[1].strip())
            elif len(parts) == 2 and parts[0] in {"authority", "autonomy"}:
                self.autonomy = parts[1].strip()
                self._refresh_status()
                self.say(f"[dim]autonomy → {escape(self.autonomy)}[/]", "dim")
            elif len(parts) == 2 and parts[0] in {"variant", "clone"}:
                self.variant = parts[1].strip()
                self.narrator = Narrator(variant=self.variant, workspace=self.workspace)
                self._refresh_status()
                self.say(f"[dim]clone → {escape(self.variant)}[/]", "dim")
            else:
                self.say("[dim]/set mode|autonomy|variant <value>[/]", "dim")
        else:
            self.say(f"[{_WARN}]{escape(name)} is gone — there are no tabs any more; everything is here[/]", "warn")

    def action_help(self) -> None:
        lines = ["[bold]Commands[/]"]
        lines += [f"  {escape(line)}" for line in command_help("core")]
        lines += [
            "  /history — recent missions here · /show <id> — one mission's evidence",
            "  /todo [task] — list or pin a TODO · /pause /resume /stop /new /clear /desktop /values",
            "[bold]Keys[/]  shift+tab cycles ask/plan/build/yolo · ↑↓ composer history · "
            "mouse-select then ctrl+c copies · esc back to the composer · ctrl+q quits",
            "[bold]Plain language[/]  create/do/make… runs · a question gets an answer · "
            "'this project is going to be…' opens a discussion",
        ]
        self.say("\n".join(lines), "dim")

    def action_cycle_mode(self) -> None:
        current = _wheel_for(self.mode, self.autonomy)
        self._set_mode(MODES[(MODES.index(current) + 1) % len(MODES)])

    def action_focus_composer(self) -> None:
        self.query_one("#composer", Composer).focus()

    def action_clear_feed(self) -> None:
        feed = self.query_one("#feed", VerticalScroll)
        for child in list(feed.children):
            child.remove()
        self._entries = 0

    def _set_mode(self, wheel: str) -> None:
        if wheel not in MODE_REQUESTS:
            self.say(
                "modes: " + "   ".join(f"[bold]{name}[/] [dim]{MODE_HELP[name]}[/]" for name in MODE_REQUESTS),
                "dim",
            )
            return
        engine_mode, autonomy = MODE_REQUESTS[wheel]
        self.mode = engine_mode
        self.autonomy = autonomy
        self._refresh_status()
        self.say(f"[dim]mode → {wheel} · {MODE_HELP[wheel]}[/]", "dim")

    # -- conversation ----------------------------------------------------------
    def _start_discussion(self, text: str) -> None:
        self._draft = text
        self.say("[dim]talking it through — nothing runs until /work[/]", "dim")
        self._start_chat(text)

    def _start_chat(self, message: str) -> None:
        if self._chat_busy:
            self._chat_queue.append(message)
            return
        history = [*self._conversation]
        self._conversation.append({"role": "user", "content": message})
        self._conversation = self._conversation[-24:]
        self._chat_busy = True
        self.narrator.record_chat("you", message)
        self._thinking = self.say("[dim]…[/]", "dim")
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
        thinking = getattr(self, "_thinking", None)
        if thinking is not None:
            thinking.remove()
            self._thinking = None
        error = str(result.get("error") or "")
        if error:
            self.say(f"[{_BAD}]I can't answer right now: {escape(_short(error, 300))}[/]", "bad")
            self.narrator.record_chat(variant, f"blocked: {error}")
        else:
            answer = str(result.get("text") or "").strip()
            self._conversation.append({"role": "assistant", "content": answer})
            self._conversation = self._conversation[-24:]
            self.say_xander(answer)
            self.narrator.record_chat(variant, answer)
        self.query_one("#composer", Composer).focus()
        if self._chat_queue:
            self._start_chat(self._chat_queue.pop(0))

    # -- orders ----------------------------------------------------------------
    def _authorize_work(self, argument: str) -> None:
        goal, source = argument.strip(), "typed"
        if not goal and self._draft:
            goal, source = self._draft, "draft"
        if not goal:
            open_goals = self._workboard_store.open_goals(self._workboard)
            if open_goals:
                goal, source = open_goals[-1].text, "stored goal"
        if not goal:
            self.say("[dim]nothing to work on — /work <goal>, or discuss something first[/]", "dim")
            return
        self._draft = ""
        if source != "typed":
            self.say(f"[dim]working on the {source}: {escape(_short(goal, 120))}[/]", "dim")
        self._dispatch(goal, self.mode if self.mode != "answer" else "implement")

    def _store_goal(self, argument: str) -> None:
        text = argument.strip()
        if not text:
            open_goals = self._workboard_store.open_goals(self._workboard)
            if not open_goals:
                self.say("[dim]no stored goals here · /goal <goal> stores one[/]", "dim")
                return
            self.say("\n".join(f"  [dim]{escape(g.id)}[/] {escape(g.text)}" for g in open_goals), "dim")
            return
        goal = self._workboard_store.add_goal(self._workboard, text)
        if goal is not None:
            self.say("[dim]goal stored · /work runs it[/]", "dim")

    def _add_todo(self, text: str) -> None:
        text = text.strip()
        if not text:
            self.say("[dim]/todo <task>[/]", "dim")
            return
        todo = self._workboard_store.add_todo(self._workboard, text)
        if todo is not None:
            self.say(f"[dim]pinned: {escape(todo.text)}[/]", "dim")

    def _run_todo_sequence(self) -> None:
        """Work the pinned TODOs one at a time; each must verify before the next starts."""

        if self.task_state == "running" or self._engine_busy:
            self.say("[dim]finish or /stop the current work first[/]", "dim")
            return
        self._todo_sequence_active = True
        if not self._start_next_todo():
            self._todo_sequence_active = False
            self.say("[dim]no open TODOs to run · /todo <task> pins one[/]", "dim")

    def _start_next_todo(self) -> bool:
        for todo in self._workboard.todos:
            if todo.state == "todo":
                self._workboard_store.set_todo_state(self._workboard, todo.id, "active")
                self._active_todo_id = todo.id
                self.goal = todo.text
                self.task_state = "running"
                self.say(f"[dim]TODO → {escape(todo.text)}[/]", "dim")
                self._run_goal(todo.text, self.mode if self.mode != "answer" else "implement")
                return True
        return False

    def _release_todo(self, state: str, evidence: str = "") -> None:
        if self._active_todo_id:
            self._workboard_store.set_todo_state(self._workboard, self._active_todo_id, state, evidence)
        self._active_todo_id = None

    def _show_todos(self) -> None:
        todos = [t for t in self._workboard.todos if t.state != "done"]
        if not todos:
            self.say("[dim]no open TODOs here[/]", "dim")
            return
        self.say("\n".join(f"  [dim]{escape(t.state)}[/] {escape(t.text)}" for t in todos), "dim")

    def _dispatch(self, goal: str, engine_mode: str) -> None:
        if self.task_state == "power-zero":
            return
        if self.task_state == "running" or self._engine_busy:
            self._order_queue.append({"goal": goal, "mode": engine_mode})
            self.say(f"[dim]queued as the next order (#{len(self._order_queue)})[/]", "dim")
            return
        self.goal = goal
        self._retry_task_id = None
        self._last_result = None
        self.task_state = "running"
        self._run_goal(goal, engine_mode)

    def _rebind_workspace(self, intent: Intent) -> None:
        if self.task_state == "running" or self._engine_busy:
            self.say("[dim]I'll move after this task lands[/]", "dim")
            return
        target = resolve_target(intent.argument, self.workspace)
        try:
            if not target.exists() and intent.create:
                target.mkdir(parents=True)
            target = target.resolve(strict=True)
        except OSError as exc:
            self.say(f"[{_BAD}]can't open {escape(str(target))}: {escape(str(exc))}[/]", "bad")
            return
        if not target.is_dir():
            self.say(f"[{_BAD}]not a folder: {escape(str(target))}[/]", "bad")
            return
        self.workspace = target
        self._workboard = self._workboard_store.load(self.workspace)
        self.narrator = Narrator(variant=self.variant, workspace=self.workspace)
        self._branch, self._dirty_count = _repository_status(self.workspace)
        self._refresh_status()
        self.say(f"[dim]workspace → {escape(str(target))}[/]", "dim")

    def _record_feedback(self, text: str) -> None:
        kind = "dislike"
        if re.search(r"\b(?:like|love|prefer|always|more)\b", text, re.IGNORECASE) and not re.search(
            r"\b(?:do\s*n[o']t\s+like|dislike|hate|never|stop|less)\b", text, re.IGNORECASE
        ):
            kind = "like"
        try:
            from .memory import MemoryStore
            from .variants import load_variant

            namespace = load_variant(self.variant).memory_namespace or self.variant
            MemoryStore(namespace=namespace).add_preference(text, kind=kind, source="explicit")
            self.say("[dim]noted — I'll keep that in mind[/]", "dim")
            self.narrator.record(f"preference recorded ({kind}): {text}", task_id=self.task_id)
        except Exception as exc:
            self.say(f"[{_BAD}]couldn't note that: {escape(str(exc))}[/]", "bad")

    # -- engine ----------------------------------------------------------------
    @work(thread=True, exclusive=True, group="xander-task")
    def _run_goal(self, goal: str, engine_mode: str | None = None) -> None:
        self._invoke_and_finish(
            engine_mode or self.mode,
            goal=goal,
            autonomy=self.autonomy if self.caller == "human" else "proposal-only",
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
                steering=self._drain_steering,
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
        self.narrator.narrate(payload)  # plain-text log twin on disk
        line, style = self._render_event(payload)
        if line:
            self.say(line, style)

    def _render_event(self, payload: dict[str, Any]) -> tuple[str, str]:
        """One feed line per event that carries news; silence for the rest."""

        kind = str(payload.get("type") or "")
        message = str(payload.get("message") or "")
        data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if kind == "phase":
            phase = str(data.get("phase") or payload.get("phase") or "")
            return f"[dim]· {escape(_PHASE_TEXT.get(phase, message))}[/]", "dim"
        if kind == "action" or kind == "patch":
            result = data.get("result") if isinstance(data.get("result"), dict) else None
            action = data.get("action") if isinstance(data.get("action"), dict) else {}
            if result is None:
                what = action.get("path") or " ".join(action.get("argv") or []) or action.get("expected") or message
                return f"[dim]▸ {escape(_short(str(what), 140))}[/]", "dim"
            status = str(result.get("status") or "")
            if status == "ok":
                changed = ", ".join(result.get("changed_paths") or [])
                out = _short(str(result.get("stdout") or ""), 200)
                tail = changed or out
                return f"[{_OK}]✓[/] {escape(tail) if tail else 'ok'}", "ok"
            reason = result.get("reason") or _short(str(result.get("stderr") or ""), 200) or status
            return f"[{_BAD}]✗ {escape(str(reason))}[/]", "bad"
        if kind == "test":
            status = str(data.get("status") or "")
            name = str(data.get("name") or message)
            if status == "ok":
                return f"[{_OK}]✓ check: {escape(_short(name, 120))}[/]", "ok"
            if status:
                return f"[{_BAD}]✗ check: {escape(_short(name, 120))}[/]", "bad"
            return f"[dim]· checking {escape(_short(name, 120))}[/]", "dim"
        if kind == "approval":
            return "", "dim"  # the approval prompt itself is shown by _show_action_approval
        if kind == "steering":
            return f"[dim]↳ {escape(_short(message, 160))}[/]", "dim"
        if kind == "error":
            return f"[{_BAD}]✗ {escape(_short(message, 300))}[/]", "bad"
        if kind == "plan":
            steps = data.get("actions") or data.get("steps") or []
            count = len(steps) if isinstance(steps, list) else steps
            decision = str(data.get("decision") or message)
            return f"[dim]plan · {escape(_short(decision, 160))}" + (f" · {count} step(s)" if count else "") + "[/]", "dim"
        if kind == "result":
            if data.get("answer"):
                return "", "dim"  # answered in _finish so it always comes last
            return "", "dim"
        if kind == "voice":
            return f"[dim italic]{escape(_short(message, 160))}[/]", "dim"
        if kind == "delegation" and data.get("delegates"):
            helpers = [h for h in data.get("delegates") or [] if h != "local execution"]
            return (f"[dim]with {escape(', '.join(helpers))}[/]", "dim") if helpers else ("", "dim")
        if kind == "research":
            sources = data.get("sources") or []
            return (f"[dim]read {len(sources)} source(s)[/]", "dim") if sources else ("", "dim")
        return "", "dim"

    def _finish(self, result: dict[str, Any]) -> None:
        task = result.get("task") if isinstance(result.get("task"), dict) else {}
        handoff = result.get("handoff") if isinstance(result.get("handoff"), dict) else {}
        self.task_id = str(result.get("task_id") or task.get("id") or handoff.get("task_id") or self.task_id)
        status = str(result.get("status") or task.get("status") or handoff.get("status") or "").casefold()
        successful = bool(result.get("ok")) or status in {"complete", "completed"}
        self._last_result = result
        options = [
            option
            for option in ((task.get("plan") or {}).get("options") or [])
            if isinstance(option, dict) and option.get("id")
        ]
        if status == "waiting_approval" and options:
            self._ask_question(options)
            return
        self.task_state = "complete" if successful else "needs-attention"
        self._retry_task_id = None if successful else self.task_id
        self.narrator.summarize(result)

        answer = ""
        for item in reversed(task.get("evidence") or []):
            if isinstance(item, dict) and item.get("kind") in {"answer", "result"} and item.get("text"):
                answer = str(item["text"])
                break
        if not answer:
            plan = task.get("plan") if isinstance(task.get("plan"), dict) else {}
            if task.get("evidence") and any(
                isinstance(e, dict) and e.get("kind") == "quick" for e in task.get("evidence") or []
            ):
                answer = str(plan.get("summary") or "")
        if successful:
            self.say_xander(answer or self._describe_success(task))
        else:
            failure = str(result.get("error") or task.get("failure") or "it didn't verify")
            self.say_xander(f"I couldn't finish that: {_short(failure, 240)}  — say retry, or tell me what to change.")
        lesson = str(task.get("lesson") or "")
        if lesson and not any(isinstance(e, dict) and e.get("kind") == "quick" for e in task.get("evidence") or []):
            self._workboard_store.add_observed_lesson(self._workboard, lesson)
        self._branch, self._dirty_count = _repository_status(self.workspace)
        self._refresh_status()
        self.query_one("#composer", Composer).focus()
        if self._todo_sequence_active and self._active_todo_id:
            if successful:
                self._release_todo("done", lesson or "verified")
                if self._start_next_todo():
                    return
                self._todo_sequence_active = False
                self.say("[dim]every pinned TODO is done[/]", "dim")
            else:
                self._release_todo("blocked", str(task.get("failure") or "did not verify"))
                self._todo_sequence_active = False
        if self._order_queue:
            self._start_next_order()

    @staticmethod
    def _describe_success(task: dict[str, Any]) -> str:
        changed = sorted(
            {
                path
                for item in task.get("results") or []
                if isinstance(item, dict)
                for path in item.get("changed_paths") or []
            }
        )
        checks = [c for c in task.get("check_results") or [] if isinstance(c, dict)]
        parts = []
        if changed:
            parts.append("changed " + ", ".join(changed[:5]) + (f" (+{len(changed) - 5})" if len(changed) > 5 else ""))
        if checks:
            parts.append(f"{sum(1 for c in checks if c.get('status') == 'ok')}/{len(checks)} checks passed")
        return ("Done — " + "; ".join(parts) if parts else "Done.") + " What now?"

    def _start_next_order(self) -> None:
        if not self._order_queue or self.task_state == "running" or self._engine_busy:
            return
        entry = self._order_queue.pop(0)
        self.say(f"[dim]next from the queue ({len(self._order_queue)} left): {escape(_short(entry['goal'], 100))}[/]", "dim")
        self._dispatch(entry["goal"], entry.get("mode") or self.mode)

    # -- approvals and choices ----------------------------------------------------
    def _approve_action(self, action: Any, reason: str) -> bool:
        decision = Event()
        pending = {"event": decision, "approved": False, "action": action, "reason": reason}
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
        target = str(data.get("path") or " ".join(data.get("argv") or []) or data.get("expected") or "this step")
        reason = str(pending.get("reason") or "")
        self.say(
            f"[bold {_ASK}]? may I run[/] [bold]{escape(_short(target, 140))}[/]"
            f"[dim] — {escape(_short(reason, 120))} · yes / no[/]",
            "ask",
        )
        self.query_one("#composer", Composer).focus()

    def _resolve_pending_approval(self, approved: bool) -> bool:
        pending = self._pending_approval
        if pending is None:
            return False
        pending["approved"] = approved
        self._pending_approval = None
        self.task_state = "running"
        pending["event"].set()
        return True

    def _answer_approval(self, value: str) -> None:
        choice = value.strip().casefold()
        if choice in {"yes", "y", "approve", "allow", "ok", "go", "1"}:
            self._resolve_pending_approval(True)
        elif choice in {"no", "n", "deny", "decline", "stop", "2"}:
            self._resolve_pending_approval(False)
        else:
            self.say("[dim]yes or no?[/]", "dim")

    def _ask_question(self, options: list[dict[str, Any]]) -> None:
        self._pending_choice = {"task_id": self.task_id, "options": [str(option["id"]) for option in options]}
        self.task_state = "question"
        lines = [f"[bold {_ASK}]? which way?[/] [dim]answer with a number[/]"]
        for index, option in enumerate(options, start=1):
            title = _short(str(option.get("title") or option["id"]), 60)
            summary = _short(str(option.get("summary") or ""), 100)
            pick = f" [{_OK}]← my pick[/]" if option.get("selected_by_default") else ""
            lines.append(f"  [bold]{index}[/] {escape(title)}{pick} [dim]{escape(summary)}[/]")
        self.say("\n".join(lines), "ask")
        self.query_one("#composer", Composer).focus()

    def _answer_question(self, value: str) -> None:
        pending = self._pending_choice or {}
        options: list[str] = pending.get("options", [])
        selected: list[str] = []
        for token in value.replace(",", " ").split():
            if token.isdigit() and 1 <= int(token) <= len(options):
                selected.append(options[int(token) - 1])
            elif token in options:
                selected.append(token)
            else:
                self.say(f"[dim]'{escape(token)}' isn't one of the options — use the numbers[/]", "dim")
                return
        self._pending_choice = None
        self.task_state = "running"
        self._run_resume(str(pending.get("task_id", "")), selected)

    # -- lifecycle -------------------------------------------------------------
    def action_new(self) -> None:
        self._resolve_pending_approval(False)
        if self.task_state == "running" or self._engine_busy:
            self.action_cancel()
        self._pending_choice = None
        self._retry_task_id = None
        self._order_queue.clear()
        self._draft = ""
        self.task_state = "idle"
        self.say("[dim]fresh start[/]", "dim")

    def action_cancel(self) -> None:
        self._resolve_pending_approval(False)
        workers = [worker for worker in self.workers if worker.group == "xander-task"]
        for worker in workers:
            worker.cancel()
        if self._todo_sequence_active:
            self._release_todo("todo")
            self._todo_sequence_active = False
        if workers or self._engine_busy or self.task_state == "running":
            self._engine_busy = False
            self.task_state = "needs-attention"
            self.say("[dim]stopped — tell me what to change, or say retry[/]", "dim")

    def action_pause(self) -> None:
        workers = [worker for worker in self.workers if worker.group == "xander-task"]
        for worker in workers:
            worker.cancel()
        if workers:
            self.task_state = "paused"
            self.say("[dim]paused — /resume continues[/]", "dim")

    def action_resume(self) -> None:
        if self.goal and self.task_state == "paused":
            self.task_state = "idle"
            self._dispatch(self.goal, self.mode)
        elif self._retry_task_id and self.task_state == "needs-attention":
            task_id, self._retry_task_id = self._retry_task_id, None
            self.task_state = "running"
            self._run_resume(task_id, [])

    def _show_history(self) -> None:
        missions = MissionStore().list(self.workspace, limit=12)
        if not missions:
            self.say("[dim]no missions here yet[/]", "dim")
            return
        lines = ["[bold]recent missions[/] [dim]/show <id> for one[/]"]
        for mission in missions:
            lines.append(f"  [dim]{escape(mission.id)}[/] {escape(mission.status)} · {escape(_short(mission.goal, 80))}")
        self.say("\n".join(lines), "dim")

    def _show_evidence(self, mission_id: str) -> None:
        mission_id = mission_id.strip() or self.task_id
        if not mission_id or mission_id == "new":
            self.say("[dim]/show <mission id> · /history lists them[/]", "dim")
            return
        try:
            mission = MissionStore().load(self.workspace, mission_id)
        except Exception as exc:
            self.say(f"[{_BAD}]{escape(str(exc))}[/]", "bad")
            return
        self.say("\n".join(escape(line) for line in mission.summary_lines()), "dim")

    def _capture_desktop(self) -> None:
        try:
            from .desktop import capture_desktop

            path = capture_desktop()
            self.say(f"[dim]screenshot saved: {escape(str(path))}[/]", "dim")
        except Exception as exc:
            self.say(f"[{_BAD}]no screenshot: {escape(str(exc))}[/]", "bad")

    def _poll_power(self) -> None:
        try:
            status = self._power_guard.poll()
        except Exception:
            return
        self._power_status = status
        if self._power_guard.tripped and self.task_state != "power-zero":
            self.action_cancel()
            self.task_state = "power-zero"
            self.say(f"[{_BAD}]power is at zero — stopping everything[/]", "bad")
        if self.is_mounted:
            self._refresh_status()

    def _shutdown_power(self) -> None:
        from .power import request_poweroff

        request_poweroff()


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
