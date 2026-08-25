"""Textual interface for Xander's task loop.

The visual language borrows Monica's compact, keyboard-first warmth without
importing or modifying Monica's implementation. Events arrive as structured
payloads and are narrated by :mod:`xander_agent.narrator` so following a run
feels like reading a good build log, not a JSON dump.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from rich.markup import escape
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Label, RichLog, Static, TabbedContent, TabPane

from . import MANTRA_PHASES
from .cli import invoke_engine
from .narrator import Narrator, STATUS_GLYPHS, _short, abilities_line

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


class XanderApp(App[None]):
    """A task-focused shell shared by the standalone and sidecar workflows."""

    TITLE = "Xander"
    SUB_TITLE = "local coding agent"
    CSS = """
    Screen {
        background: #11151d;
        color: #d7dae0;
    }
    Header {
        background: #1b202b;
        color: #ffcb6b;
    }
    #context-bar {
        height: 4;
        padding: 1 2 0 2;
        color: #aab1c0;
        background: #171c25;
    }
    #phase-strip {
        height: 3;
        padding: 0 2 1 2;
        color: #798192;
        background: #171c25;
    }
    #goal-row {
        height: 3;
        padding: 0 1;
        background: #11151d;
    }
    #goal-label {
        width: 9;
        padding: 1 1;
        color: #ffcb6b;
    }
    #goal-input {
        width: 1fr;
        border: tall #596173;
        background: #171c25;
    }
    #goal-input:focus {
        border: tall #ffb86c;
    }
    TabbedContent {
        height: 1fr;
        margin: 0 1;
    }
    TabPane {
        padding: 1 2;
        background: #151a22;
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
    Footer {
        background: #1b202b;
    }
    """
    BINDINGS = [
        Binding("w", "work", "Work"),
        Binding("ctrl+n", "new", "New"),
        Binding("ctrl+r", "run", "Run"),
        Binding("ctrl+p", "pause", "Pause"),
        Binding("ctrl+shift+p", "resume", "Resume"),
        Binding("ctrl+t", "tests", "Tests"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("f1", "help", "Help"),
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
        self.narrator = Narrator(variant=variant, workspace=self.workspace)
        self._engine_busy = False
        self._order_queue: list[str] = []
        self._pending_choice: dict[str, Any] | None = None
        self._retry_task_id: str | None = None
        self._branch, self._dirty_count = _repository_status(self.workspace)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(self._context_text(), id="context-bar")
        yield Static(self._phase_text(), id="phase-strip")
        with Horizontal(id="goal-row"):
            yield Label("goal  ▸", id="goal-label")
            yield Input(placeholder="Give the order; Enter runs the full loop", id="goal-input")
        with TabbedContent(initial="goal-view"):
            with TabPane("Goal", id="goal-view"):
                yield Static("No goal yet. Xander will preserve unrelated work and prove completion.", id="goal-summary", classes="empty-view")
            with TabPane("Research", id="research-view"):
                yield RichLog(id="research-log", wrap=True, highlight=False, markup=True)
            with TabPane("Plan", id="plan-view"):
                yield RichLog(id="plan-log", wrap=True, highlight=False, markup=True)
            with TabPane("Run", id="run-view"):
                yield RichLog(id="run-log", wrap=True, highlight=False, markup=True)
            with TabPane("Diff", id="diff-view"):
                yield RichLog(id="diff-log", wrap=False, highlight=True, markup=True)
            with TabPane("Tests", id="tests-view"):
                yield RichLog(id="tests-log", wrap=True, highlight=False, markup=True)
            with TabPane("Tasks", id="tasks-view"):
                yield RichLog(id="tasks-log", wrap=True, highlight=False, markup=True)
            with TabPane("Skills", id="skills-view"):
                yield RichLog(id="skills-log", wrap=True, highlight=False, markup=True)
            with TabPane("Variants", id="variants-view"):
                yield RichLog(id="variants-log", wrap=True, highlight=False, markup=True)
            with TabPane("Settings", id="settings-view"):
                yield Static(self._settings_text(), id="settings-text", classes="empty-view")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#goal-input", Input).focus()
        self.query_one("#context-bar", Static).tooltip = f"Xander is bound to {self.workspace}"
        run_log = self.query_one("#run-log", RichLog)
        run_log.write(f"[bold #ffcb6b]Xander[/] reporting · [dim]{abilities_line()}[/]")
        run_log.write(f"[dim]I’m here: {escape(str(self.workspace))}[/]")
        run_log.write("[dim]Give me the outcome. I’ll show what I understood, do the work I can prove, and ask when direction matters.[/]")
        self._refresh_tasks()
        self._refresh_variants()
        self._load_skills_panel()

    # -- context strips -------------------------------------------------------
    def _context_text(self) -> str:
        repository = "not-git" if self._branch == "not-git" else f"{self._branch} dirty:{self._dirty_count}"
        return (
            f"here:{self.workspace}\n"
            f"repo:{repository}  caller:{self.caller}  mode:{self.mode}  variant:{self.variant}   "
            f"model:{self.model}  task:{self.task_id}  state:{self.task_state}"
        )

    def _phase_text(self) -> str:
        rendered = []
        for index, name in enumerate(MANTRA_PHASES):
            rendered.append(f"[bold #ffcb6b]◆ {name}[/]" if index == self.phase_index else f"[dim]{name}[/]")
        return "  ›  ".join(rendered)

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
            f"abilities: {abilities_line()}\n"
            f"routing: {routing}"
        )

    def watch_phase_index(self, _: int) -> None:
        matches = self.query("#phase-strip")
        if len(matches):
            matches.first(Static).update(self._phase_text())

    def watch_task_state(self, _: str) -> None:
        matches = self.query("#context-bar")
        if len(matches):
            matches.first(Static).update(self._context_text())

    # -- side panels ----------------------------------------------------------
    def _fill_log(self, selector: str, lines: list[str]) -> None:
        log = self.query_one(selector, RichLog)
        log.clear()
        for line in lines:
            log.write(line)

    def _refresh_tasks(self) -> None:
        lines: list[str] = []
        try:
            from .tasks import TaskStore

            records = [
                record
                for record in TaskStore().list(limit=200)
                if record.request.workspace.expanduser().resolve(strict=False) == self.workspace
            ][:20]
        except Exception as exc:
            lines = [f"[red]task history unavailable: {escape(str(exc))}[/]"]
        else:
            if not records:
                lines = ["[dim]no tasks yet — Xander is ready for orders[/]"]
            for record in records:
                glyph, style = STATUS_GLYPHS.get(record.status.value, ("·", "dim"))
                goal = " ".join(record.request.goal.split())[:64]
                lines.append(
                    f"[{style}]{glyph}[/] {record.id}  "
                    f"[dim]{record.status} · {record.phase} · a{record.attempt}[/]  {escape(goal)}"
                )
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
            lines.append("[dim](quartermaster) at most 3 skills load per task — only the essential[/]")
        self.call_from_thread(self._fill_log, "#skills-log", lines)

    # -- orders ----------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
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
        if self.task_state == "running" or self._engine_busy:
            self._order_queue.append(value)
            self.query_one("#goal-input", Input).value = ""
            self.query_one("#run-log", RichLog).write(
                f"[dim]＋ queued order #{len(self._order_queue)}:[/] {escape(_short(value, 100))}"
            )
            return
        self.goal = value
        self.query_one("#goal-summary", Static).update(value)
        self.action_run()

    def action_work(self) -> None:
        """The one command: focus the order line, ready for a task."""

        goal_input = self.query_one("#goal-input", Input)
        goal_input.focus()
        if self.task_state == "running":
            self.notify(escape("engine busy — [enter] queues the order"), title="Work")

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
        if self._pending_choice is not None:
            pending_task = self._pending_choice.get("task_id", "")
            self._pending_choice = None
            self.query_one("#run-log", RichLog).write(
                f"[dim]question dismissed — resume later with: xander resume {escape(str(pending_task))} --select <id>[/]"
            )
        self.goal = ""
        self._retry_task_id = None
        self.task_id = "new"
        self.phase_index = 0
        self.task_state = "idle"
        goal_input = self.query_one("#goal-input", Input)
        goal_input.value = ""
        goal_input.focus()
        self.query_one("#goal-summary", Static).update("No goal yet.")

    def action_run(self) -> None:
        goal_input = self.query_one("#goal-input", Input)
        goal = (goal_input.value or self.goal).strip()
        if not goal or self.task_state == "running":
            return
        if self._engine_busy:
            self.query_one("#run-log", RichLog).write(
                "[yellow]previous engine run is still winding down — order queued[/]"
            )
            self._order_queue.append(goal)
            goal_input.value = ""
            return
        self.goal = goal
        self._retry_task_id = None
        self.task_state = "running"
        self.phase_index = 0
        self.query_one("#run-log", RichLog).write(f"[bold #ffcb6b]▶ order accepted[/] {escape(goal)}")
        self._run_goal(goal)

    @work(thread=True, exclusive=True, group="xander-task")
    def _run_goal(self, goal: str) -> None:
        self._invoke_and_finish(
            self.mode,
            goal=goal,
            autonomy=self.autonomy if self.caller == "human" else "proposal-only",
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
        channel, line = self.narrator.narrate(payload)
        if channel != "run":
            self.query_one(_CHANNELS[channel], RichLog).write(line)
        self.query_one("#run-log", RichLog).write(line)

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
        self.query_one("#context-bar", Static).update(self._context_text())
        for line in self.narrator.summarize(result):
            run_log.write(line)
        self._refresh_tasks()
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
        goal = self._order_queue.pop(0)
        self._retry_task_id = None
        self.query_one("#run-log", RichLog).write(f"[dim]▶ next from queue ({len(self._order_queue)} left)[/]")
        self.goal = goal
        self.query_one("#goal-summary", Static).update(goal)
        self.task_state = "running"
        self.phase_index = 0
        self.query_one("#run-log", RichLog).write(f"[bold #ffcb6b]▶ order accepted[/] {escape(goal)}")
        self._run_goal(goal)

    def action_pause(self) -> None:
        workers = [worker for worker in self.workers if worker.group == "xander-task"]
        for worker in workers:
            worker.cancel()
        if workers:
            self.task_state = "paused"
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
        self.query_one(TabbedContent).active = "tests-view"

    def action_help(self) -> None:
        self.notify(
            escape("[w] work  [enter] run/queue  [^p] pause  [^P] resume  [^n] new  [^t] tests  [^q] quit"),
            title="Xander orders",
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
