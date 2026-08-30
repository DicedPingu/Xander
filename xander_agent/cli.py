"""Stable human and machine-readable command line interface for Xander."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import ABILITIES, MANTRA_PHASES, __version__
from .paths import (
    cache_dir,
    config_dir,
    ensure_runtime_dirs,
    logs_dir,
    migration_dir,
    projects_dir,
    shared_dir,
    state_dir,
    tasks_dir,
    variants_dir,
)
from .opening import OPENING_STYLES
from .variants import (
    VariantError,
    army,
    clone_variant,
    export_variant,
    import_variant,
    list_variants,
    load_variant,
    remove_variant,
)

CLI_SCHEMA = "xander.cli/v1"
_COMMANDS = {
    "learn",
    "say",
    "self",
    "run",
    "plan",
    "inspect",
    "research",
    "answer",
    "test-triage",
    "resume",
    "tasks",
    "doctor",
    "skills",
    "variant",
    "army",
    "stats",
    "hooks",
    "remote",
    "web",
    "serve",
    "mcp",
    "tui",
    "mission",
    "oversee",
}


class InterfaceError(RuntimeError):
    """A command cannot be handed to the core engine."""


def _serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


_THIN_EVENT_KEYS = {"event", "type", "message", "phase"}


class Emitter:
    def __init__(self, *, jsonl: bool = False, plain: bool = False) -> None:
        self.jsonl = jsonl
        self.plain = plain

    def __call__(self, event: Any) -> None:
        payload = _serializable(event)
        if not isinstance(payload, dict):
            payload = {"value": payload}
        if self.jsonl:
            envelope = {"schema": CLI_SCHEMA, **payload}
            print(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")), flush=True)
            return
        event_name = payload.get("event") or payload.get("type")
        # Only the one-line form when there is genuinely nothing else to say.
        # Keying off len(payload) silently swallowed every {"event": ..., "result": ...}
        # payload, which is most of what Xander has to report.
        if event_name and not (set(payload) - _THIN_EVENT_KEYS):
            detail = payload.get("message") or payload.get("phase") or ""
            print(f"{event_name}: {detail}".rstrip())
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))


def _interactive_approval(action: Any, reason: str) -> bool:
    if not sys.stdin.isatty():
        return False
    from .policy import action_text

    print(f"\nXander requests approval: {reason}", file=sys.stderr)
    print(f"Action: {action_text(action)}", file=sys.stderr)
    try:
        answer = input("Approve this action? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().casefold() in {"y", "yes"}


def _is_success(result: Mapping[str, Any]) -> bool:
    task = result.get("task") if isinstance(result.get("task"), Mapping) else {}
    handoff = result.get("handoff") if isinstance(result.get("handoff"), Mapping) else {}
    status = result.get("status") or task.get("status") or handoff.get("status")
    return bool(result.get("ok")) or str(status or "").casefold() in {
        "complete",
        "completed",
    }


def _load_engine() -> type[Any]:
    try:
        module = importlib.import_module("xander_agent.engine")
        return module.Engine
    except (ImportError, AttributeError) as exc:
        raise InterfaceError(
            "Xander's core engine is unavailable; reinstall the package or run `xander doctor`."
        ) from exc


def create_engine(
    workspace: Path,
    *,
    variant: str = "default",
    autonomy: str | None = None,
    event_sink: Callable[[Any], None] | None = None,
    approve: Callable[[Any, str], bool] | None = None,
    log_events: bool = True,
    steering: Callable[[], dict[str, Any]] | None = None,
) -> Any:
    engine_type = _load_engine()
    return engine_type(
        workspace=workspace.expanduser().resolve(),
        variant=variant,
        autonomy=autonomy,
        event_sink=event_sink,
        approve=approve,
        log_events=log_events,
        steering=steering,
    )


def _learn_command(args: Any, workspace: Path, emit: Callable[[dict[str, Any]], None]) -> int:
    """Queue targets, then work them while listening on the back channel."""

    import time

    from .learn import LearnStore
    from .learnrun import run_learning
    from .steering import SteeringInbox

    store = LearnStore()
    queue = store.load(workspace)

    if args.mission:
        store.set_mission(queue, args.mission)
    if args.depth is not None:
        queue.settings.depth = max(0, min(10, args.depth))
        store.save(queue)
    added = store.add_many(queue, list(args.targets), origin="operator") if args.targets else []

    if args.status or (not args.targets and not queue.pending()):
        emit(
            {
                "event": "learn",
                "mission": queue.mission,
                "progress": queue.progress(),
                "settings": queue.settings.model_dump(mode="json"),
                "restrictions": list(queue.restrictions),
                "steering": [
                    {"text": note.text, "outcome": note.outcome, "applied": bool(note.applied_at)}
                    for note in SteeringInbox(workspace).history(limit=10)
                ],
                "targets": [
                    {"subject": item.subject, "kind": item.kind, "state": item.state, "priority": item.priority}
                    for item in queue.targets[-40:]
                ],
            }
        )
        return 0

    if args.queue:
        emit(
            {
                "event": "learn",
                "message": f"queued {len(added)} target(s); {queue.progress()}",
                "queued": [item.subject for item in added],
                "progress": queue.progress(),
            }
        )
        return 0

    deadline = time.monotonic() + args.minutes * 60 if args.minutes else None
    summary = run_learning(
        workspace,
        variant=args.variant,
        caller="human",
        max_targets=args.max_targets,
        store=store,
        deadline=deadline,
    )
    emit(summary)
    return 0


def _self_command(args: Any, workspace: Path, emit: Callable[[dict[str, Any]], None]) -> int:
    from . import selfwork

    store = selfwork.SelfStore()
    command = args.self_command

    if command == "status":
        emit(selfwork.status(store))
        return 0
    if command == "set":
        record = store.load()
        value: Any = args.value
        if args.name == "config":
            value = args.value.casefold() in {"on", "true", "yes", "1"}
        elif args.name == "max_changed_files":
            try:
                value = int(args.value)
            except ValueError:
                emit({"event": "self", "ok": False, "error": "max_changed_files takes a whole number"})
                return 2
        # Round-trip through validation: assignment alone does not check a
        # Literal, and a silently-invalid policy is worse than a rejected one.
        candidate = record.policy.model_dump(mode="json")
        candidate[args.name] = value
        try:
            record.policy = selfwork.SelfPolicy.model_validate(candidate)
        except Exception as exc:
            allowed = {"code": "off|propose|verified", "models": "off|installed|any",
                       "system": "off|ask|allow"}.get(args.name, "")
            emit({"event": "self", "ok": False, "error": f"invalid {args.name}={args.value}"
                  + (f"; expected {allowed}" if allowed else ""), "detail": str(exc)[:200]})
            return 2
        store.save(record)
        emit({"event": "self", "ok": True, "message": f"{args.name} = {getattr(record.policy, args.name)}"})
        return 0
    if command == "models":
        changes = selfwork.tune_models(workspace, variant=args.variant, store=store)
        emit({"event": "self", "changes": [item.model_dump(mode="json") for item in changes]})
        return 0

    change = selfwork.improve_code(workspace, " ".join(args.goal), variant=args.variant, store=store)
    emit(
        {
            "event": "self",
            "message": f"{'kept' if change.kept else 'not kept'}: {change.detail}",
            "change": change.model_dump(mode="json"),
        }
    )
    return 0 if change.kept or not change.paths else 1


def invoke_engine(
    mode: str,
    *,
    workspace: Path,
    goal: str = "",
    variant: str = "default",
    autonomy: str | None = None,
    caller: str = "human",
    constraints: Sequence[str] = (),
    acceptance_checks: Sequence[str | dict[str, Any]] = (),
    allowed_paths: Sequence[str] = (),
    timeout: int | None = None,
    setup_policy: str = "ask",
    task_id: str | None = None,
    selected_options: Sequence[str] = (),
    event_sink: Callable[[Any], None] | None = None,
    approve: Callable[[Any, str], bool] | None = None,
    log_events: bool = True,
) -> dict[str, Any]:
    """Call the core through its public interface without importing it eagerly."""

    engine = create_engine(
        workspace,
        variant=variant,
        autonomy=autonomy,
        event_sink=event_sink,
        approve=approve,
        log_events=log_events,
    )
    if mode == "resume":
        if not task_id:
            raise InterfaceError("resume requires a task id")
        return _serializable(
            engine.resume(
                task_id,
                selected_options=tuple(selected_options),
                setup_policy=None if setup_policy == "ask" else setup_policy,
            )
        )
    checks = []
    for index, check in enumerate(acceptance_checks, start=1):
        if isinstance(check, dict):
            checks.append(check)
        else:
            argv = shlex.split(check)
            if not argv:
                raise InterfaceError("acceptance checks must be non-empty commands")
            checks.append({"name": f"acceptance-{index}", "argv": argv, "required": True})
    result = engine.execute(
        mode=mode,
        goal=goal,
        constraints=tuple(constraints),
        acceptance_checks=tuple(checks),
        allowed_paths=tuple(allowed_paths),
        timeout=timeout,
        caller=caller,
        setup_policy=setup_policy,
    )
    return _serializable(result)


def doctor_payload(workspace: Path, variant: str = "default") -> dict[str, Any]:
    ensure_runtime_dirs()
    dependencies: dict[str, str | None] = {}
    for package in ("textual", "mcp", "pydantic"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    engine_report: dict[str, Any]
    try:
        engine_report = _serializable(create_engine(workspace, variant=variant).doctor())
    except Exception as exc:
        engine_report = {"ok": False, "error": str(exc), "type": type(exc).__name__}
    repository = engine_report.get("repository")
    if isinstance(repository, dict) and isinstance(repository.get("dirty"), dict):
        dirty = repository.pop("dirty")
        repository["dirty_count"] = len(dirty)
        repository["dirty_sample"] = sorted(dirty)[:20]
    skills = engine_report.get("skills")
    if isinstance(skills, dict) and isinstance(skills.get("broken_symlinks"), list):
        broken = skills["broken_symlinks"]
        skills["broken_symlink_count"] = len(broken)
        skills["broken_symlinks"] = broken[:20]
    backend = engine_report.get("backend")
    engine_ok = engine_report.get("ok", True)
    if isinstance(backend, dict):
        engine_ok = engine_ok and bool(backend.get("available"))
    return {
        "event": "doctor",
        "ok": bool(engine_ok) and all(dependencies.values()),
        "version": __version__,
        "python": sys.version.split()[0],
        "mantra": list(MANTRA_PHASES),
        "abilities": list(ABILITIES),
        "workspace": str(workspace.expanduser().resolve()),
        "paths": {
            "config": str(config_dir()),
            "state": str(state_dir()),
            "cache": str(cache_dir()),
            "logs": str(logs_dir()),
            "projects": str(projects_dir()),
            "shared": str(shared_dir()),
            "tasks": str(tasks_dir()),
            "variants": str(variants_dir()),
            "migrations": str(migration_dir()),
        },
        "dependencies": dependencies,
        "engine": engine_report,
    }


def army_payload() -> dict[str, Any]:
    """Muster report: hierarchy, learning, and verified wins per clone."""

    rows = army()
    try:
        from .tasks import TaskStore

        records = TaskStore().list(limit=200)
        wins: dict[str, int] = {}
        for record in records:
            if record.status.value == "completed":
                wins[record.request.variant] = wins.get(record.request.variant, 0) + 1
        for row in rows:
            row["tasks_won"] = wins.get(row["name"], 0)
    except Exception:
        for row in rows:
            row.setdefault("tasks_won", 0)
    return {
        "event": "army",
        "schema": "xander.army/v1",
        "leader": next((row["name"] for row in rows if row["depth"] == 0), "default"),
        "size": len(rows),
        "ranks": rows,
    }


def _add_request_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--constraint", action="append", default=[], help="hard constraint; repeatable")
    parser.add_argument("--accept", action="append", default=[], help="acceptance command; repeatable")
    parser.add_argument("--allow", action="append", default=[], help="allowed relative path; repeatable")
    parser.add_argument("--timeout", type=int, help="deadline in seconds")
    parser.add_argument(
        "--setup-policy",
        choices=("ask", "allow", "never"),
        default="ask",
        help="missing toolchain handling: pause for approval, permit setup actions, or never install",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xander", description="Local coding agent for humans, Codex, and Claude")
    parser.add_argument("--version", action="version", version=f"xander {__version__}")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--variant", default="default")
    parser.add_argument(
        "--autonomy", choices=("full-auto", "supervised", "proposal-only"), default=None
    )
    parser.add_argument("--caller", choices=("human", "codex", "claude"), default="human")
    parser.add_argument("--json", action="store_true", help="emit stdout as JSONL only")
    parser.add_argument("--plain", action="store_true", help="disable the full-screen TUI")
    parser.add_argument(
        "--style",
        choices=tuple(style.key for style in OPENING_STYLES),
        default="desk",
        # Accepted but inert: the opening layouts it chose were removed in the
        # TUI rework. Kept so existing scripts and aliases do not break.
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="execute a goal through the complete mantra loop")
    run.add_argument("goal", nargs="+")
    _add_request_options(run)

    plan = subparsers.add_parser("plan", help="research and prepare a proposal without applying changes")
    plan.add_argument("goal", nargs="+")
    _add_request_options(plan)

    for name, help_text in {
        "inspect": "inspect a workspace and return evidence without modifying it",
        "research": "research a workspace and return an evidence bundle without modifying it",
        "answer": "answer a grounded question without modifying the workspace",
        "test-triage": "prepare a test-triage handoff without modifying the workspace",
    }.items():
        read_only = subparsers.add_parser(name, help=help_text)
        read_only.add_argument("goal", nargs="+")
        _add_request_options(read_only)

    resume = subparsers.add_parser("resume", help="resume a persisted task")
    resume.add_argument("task_id")
    resume.add_argument("--select", action="append", default=[], help="compatible plan option id; repeatable")
    resume.add_argument(
        "--setup-policy",
        choices=("ask", "allow", "never"),
        default="ask",
        help="for package/toolchain actions: ask, explicitly allow setup, or never install",
    )

    tasks = subparsers.add_parser("tasks", help="inspect persisted tasks")
    task_commands = tasks.add_subparsers(dest="tasks_command", required=True)
    task_commands.add_parser("list")
    task_show = task_commands.add_parser("show")
    task_show.add_argument("task_id")

    learn = subparsers.add_parser(
        "learn", help="queue URLs, packages, skills, categories or topics and work through them"
    )
    learn.add_argument("targets", nargs="*", help="anything to learn; prefix with pypi:/skill:/category:/topic: to be exact")
    learn.add_argument("--mission", default="", help="the standing objective for this learning run")
    learn.add_argument("--depth", type=int, help="follow-ups Xander may add per target (0 disables)")
    learn.add_argument("--max", type=int, default=0, dest="max_targets", help="stop after this many targets")
    learn.add_argument("--minutes", type=int, help="wall-clock budget for the whole run")
    learn.add_argument("--queue", action="store_true", help="add the targets and exit without working them")
    learn.add_argument("--status", action="store_true", help="show the queue, mission, restrictions and settings")

    say = subparsers.add_parser(
        "say", help="steer a running learn loop; returns immediately without interrupting it"
    )
    say.add_argument("message", nargs="+", help="plain text, or !focus / !drop / !add / !never / !set / !stop")

    selfp = subparsers.add_parser("self", help="Xander working on Xander: his code, config and model routing")
    self_commands = selfp.add_subparsers(dest="self_command", required=True)
    self_commands.add_parser("status", help="policy, what he kept, what was reverted")
    self_improve = self_commands.add_parser("improve", help="change his own code, keep it only if the suite passes")
    self_improve.add_argument("goal", nargs="+", help="what to improve about himself")
    self_commands.add_parser("models", help="re-route each role to the best build that actually responds")
    self_set = self_commands.add_parser("set", help="open or close a dial")
    self_set.add_argument("name", choices=["code", "config", "models", "system", "max_changed_files"])
    self_set.add_argument("value")

    subparsers.add_parser("doctor", help="check the interface, state paths, models, and tools")
    subparsers.add_parser("oversee", help="show the latest task, model choices, delegations, and blockers")

    skills = subparsers.add_parser("skills", help="query Xander's compact skill registry")
    skill_commands = skills.add_subparsers(dest="skills_command", required=True)
    skill_commands.add_parser("refresh")
    skill_commands.add_parser("list")
    skill_search = skill_commands.add_parser("search")
    skill_search.add_argument("query")
    skill_commands.add_parser("doctor")
    skill_commands.add_parser("embed", help="build the dense index so search understands meaning, not just words")
    skill_create = skill_commands.add_parser("create", help="write a new skill of Xander's own")
    skill_create.add_argument("name")
    skill_create.add_argument("--description", required=True)
    skill_create.add_argument("--body", help="skill body; omit to read from stdin")
    skill_create.add_argument("--replace", action="store_true")

    variants = subparsers.add_parser("variant", help="clone and move versioned Xander profiles")
    variant_commands = variants.add_subparsers(dest="variant_command", required=True)
    variant_clone = variant_commands.add_parser("clone")
    variant_clone.add_argument("name")
    variant_clone.add_argument("--from", dest="from_name", default="default")
    variant_clone.add_argument("--profile-version")
    variant_clone.add_argument("--engine")
    variant_clone.add_argument("--replace", action="store_true")
    variant_commands.add_parser("list")
    variant_show = variant_commands.add_parser("show")
    variant_show.add_argument("name")
    variant_export = variant_commands.add_parser("export")
    variant_export.add_argument("name")
    variant_export.add_argument("--output", type=Path, required=True)
    variant_export.add_argument("--include-engine", action="store_true")
    variant_import = variant_commands.add_parser("import")
    variant_import.add_argument("bundle", type=Path)
    variant_import.add_argument("--replace", action="store_true")
    variant_remove = variant_commands.add_parser("remove")
    variant_remove.add_argument("name")

    subparsers.add_parser("army", help="muster the clone army: leader, ranks, lessons, wins")

    subparsers.add_parser("stats", help="the scoreboard: success rate, streaks, tokens, squad activity")

    missions = subparsers.add_parser("mission", help="readable Mission history for this workspace")
    mission_commands = missions.add_subparsers(dest="mission_command", required=True)
    mission_commands.add_parser("list")
    mission_show = mission_commands.add_parser("show")
    mission_show.add_argument("mission_id")
    mission_delete = mission_commands.add_parser("delete")
    mission_delete.add_argument("mission_id")

    hooks = subparsers.add_parser("hooks", help="broad operator hooks, narrowed per mission")
    hook_commands = hooks.add_subparsers(dest="hooks_command", required=True)
    hook_commands.add_parser("list")
    hook_add = hook_commands.add_parser("add")
    hook_add.add_argument("--on", default="*", help="analyze|plan|test|victory|setback|*")
    hook_add.add_argument("--match", default="", help="regex narrowing this hook to matching goals")
    hook_add.add_argument("--constraint", default="", help="constraint to add to matching missions")
    hook_add.add_argument("--note", default="", help="line to speak at the moment")
    hook_narrow = hook_commands.add_parser("narrow", help="show which hooks a goal would arm")
    hook_narrow.add_argument("goal", nargs="+")

    remote = subparsers.add_parser("remote", help="MCP to MCP: use other MCP servers' tools")
    remote_commands = remote.add_subparsers(dest="remote_command", required=True)
    remote_commands.add_parser("list", help="configured remote MCP servers")
    remote_add = remote_commands.add_parser("add")
    remote_add.add_argument("name")
    remote_add.add_argument("command", nargs="+")
    remote_tools = remote_commands.add_parser("tools")
    remote_tools.add_argument("server")
    remote_call = remote_commands.add_parser("call")
    remote_call.add_argument("server")
    remote_call.add_argument("tool")
    remote_call.add_argument("--arguments", default="{}", help="JSON object of tool arguments")

    web_search = subparsers.add_parser("web", help="search online through Xander's providers")
    web_search.add_argument("query", nargs="+")
    web_search.add_argument(
        "--provider",
        action="append",
        default=[],
        help="duckduckgo|wikipedia|pypi|stackoverflow; repeatable",
    )
    web_search.add_argument("--fetch", help="fetch one URL as readable text instead of searching")

    serve = subparsers.add_parser("serve", help="loopback HTTP bridge for a browser addon or script")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--origin", default="", help="allowed CORS origin, e.g. moz-extension://<id>")

    subparsers.add_parser("mcp", help="serve Xander tools over MCP stdio")
    subparsers.add_parser("tui", help="open the full-screen terminal interface")
    return parser


def _normalize_legacy(argv: list[str]) -> list[str]:
    """Keep ``xander GOAL`` and the former plan/yolo flags working."""

    if not argv:
        return argv
    flags = {"--json", "--plain", "--version"}
    valued = {"--workspace", "--variant", "--autonomy", "--caller", "--style"}
    prefix: list[str] = []
    remainder: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        base = value.split("=", 1)[0]
        if value in flags or (base in valued and "=" in value):
            prefix.append(value)
            index += 1
        elif value in valued:
            if index + 1 >= len(argv):
                remainder.append(value)
                index += 1
            else:
                prefix.extend((value, argv[index + 1]))
                index += 2
        else:
            remainder.append(value)
            index += 1
    plan_only = "--plan-only" in remainder
    remainder = [item for item in remainder if item not in {"--plan-only", "--yolo"}]
    if remainder and not remainder[0].startswith("-") and remainder[0] not in _COMMANDS:
        remainder.insert(0, "plan" if plan_only else "run")
    return [*prefix, *remainder]


def _skill_command(args: argparse.Namespace) -> Any:
    try:
        module = importlib.import_module("xander_agent.skills")
    except ImportError as exc:
        raise InterfaceError("the skill registry is not installed") from exc
    command = args.skills_command
    if command == "create":
        registry_type = getattr(module, "SkillRegistry", None)
        if registry_type is None:
            raise InterfaceError("the skill registry is not installed")
        body = args.body
        if body is None:
            body = sys.stdin.read()
        return registry_type().author(
            args.name, args.description, body, replace=args.replace
        )
    direct = getattr(module, f"{command}_skills", None)
    if callable(direct):
        return direct(getattr(args, "query", None)) if command == "search" else direct()
    registry_type = getattr(module, "SkillRegistry", None)
    if registry_type is None:
        raise InterfaceError(f"the skill registry does not support `{command}`")
    registry = registry_type()
    method = getattr(registry, command)
    return method(getattr(args, "query", None)) if command == "search" else method()


def _run_command(args: argparse.Namespace, emit: Emitter) -> int:
    workspace = args.workspace.expanduser().resolve()
    if args.command in {None, "tui"}:
        from .tui import run_tui

        run_tui(
            workspace=workspace,
            variant=args.variant,
            caller=args.caller,
            autonomy=args.autonomy,
        )
        return 0
    if args.command == "mcp":
        from .mcp_server import run_stdio

        run_stdio()
        return 0
    if args.command == "doctor":
        report = doctor_payload(workspace, variant=args.variant)
        emit(report)
        return 0 if report["ok"] else 1
    if args.command == "oversee":
        from .oversight import oversight_payload

        emit(oversight_payload(workspace))
        return 0
    if args.command == "say":
        from .steering import SteeringInbox

        note = SteeringInbox(workspace).post(" ".join(args.message))
        if note is None:
            emit({"event": "say", "ok": False, "error": "nothing to say"})
            return 1
        # Deliberately does not wait for the engine: the point is that talking
        # to Xander mid-run never blocks the operator or the run.
        emit({"event": "say", "ok": True, "id": note.id, "text": note.text, "queued_at": note.received_at})
        return 0
    if args.command == "learn":
        return _learn_command(args, workspace, emit)
    if args.command == "self":
        return _self_command(args, workspace, emit)
    if args.command in {"run", "plan", "inspect", "research", "answer", "test-triage"}:
        mode = {
            "run": "implement",
            "plan": "plan",
            "inspect": "inspect",
            "research": "research",
            "answer": "answer",
            "test-triage": "test-triage",
        }[args.command]
        result = invoke_engine(
            mode,
            workspace=workspace,
            goal=" ".join(args.goal),
            variant=args.variant,
            autonomy="proposal-only" if args.command != "run" else args.autonomy,
            caller=args.caller,
            constraints=args.constraint,
            acceptance_checks=args.accept,
            allowed_paths=args.allow,
            timeout=args.timeout,
            setup_policy=args.setup_policy,
            event_sink=emit,
            approve=_interactive_approval if not args.json and args.caller == "human" else None,
        )
        emit({"event": "result", "result": result})
        return 0 if _is_success(result) else 1
    if args.command == "resume":
        result = invoke_engine(
            "resume",
            workspace=workspace,
            variant=args.variant,
            autonomy=args.autonomy,
            caller=args.caller,
            task_id=args.task_id,
            selected_options=args.select,
            setup_policy=args.setup_policy,
            event_sink=emit,
            approve=_interactive_approval if not args.json and args.caller == "human" else None,
        )
        emit({"event": "result", "result": result})
        return 0 if _is_success(result) else 1
    if args.command == "tasks":
        engine = create_engine(workspace, variant=args.variant, autonomy="proposal-only")
        result = engine.list_tasks() if args.tasks_command == "list" else engine.show_task(args.task_id)
        emit({"event": f"tasks.{args.tasks_command}", "result": _serializable(result)})
        return 0
    if args.command == "mission":
        from .mission import MissionStore

        store = MissionStore()
        if args.mission_command == "list":
            missions = store.list(workspace, limit=200)
            if args.json:
                emit({"event": "mission.list", "result": [mission.summary_lines() for mission in missions]})
            elif not missions:
                print(f"No Missions recorded in {workspace}.")
            else:
                print(f"Missions in {workspace}")
                for mission in missions:
                    print(f"  {mission.id}  {mission.status:<16} {mission.goal}")
            return 0
        mission = store.load(workspace, args.mission_id)
        if args.mission_command == "delete":
            store.delete(workspace, args.mission_id)
            if args.json:
                emit({"event": "mission.delete", "result": {"deleted": args.mission_id}})
            else:
                print(f"Deleted Mission {args.mission_id} from {workspace}.")
            return 0
        if args.json:
            emit({"event": "mission.show", "result": mission.summary_lines() + [f"  {item['kind']}: {item['message']}" for item in mission.timeline()]})
        else:
            print("\n".join(mission.summary_lines()))
            for item in mission.timeline():
                print(f"  {item['kind']:<9} {item['message']}")
        return 0
    if args.command == "skills":
        emit({"event": f"skills.{args.skills_command}", "result": _serializable(_skill_command(args))})
        return 0
    if args.command == "army":
        emit(army_payload())
        return 0
    if args.command == "stats":
        from .stats import render_lines, stats_payload

        payload = stats_payload()
        if args.json:
            emit(payload)
        else:
            from rich.console import Console

            console = Console()
            for line in render_lines(payload):
                console.print(line)
        return 0
    if args.command == "hooks":
        from .hooks import HookBook

        book = HookBook()
        if args.hooks_command == "add":
            hook = book.add(
                on=args.on, match=args.match, constraint=args.constraint, note=args.note
            )
            result: Any = {"on": hook.on, "match": hook.match, "constraint": hook.constraint, "note": hook.note}
        elif args.hooks_command == "narrow":
            goal = " ".join(args.goal)
            result = {
                "goal": goal,
                "constraints": book.constraints_for(goal),
                "notes": book.notes_for(goal, "analyze"),
            }
        else:
            result = [
                {"on": hook.on, "match": hook.match, "constraint": hook.constraint, "note": hook.note}
                for hook in book.hooks
            ]
        emit({"event": f"hooks.{args.hooks_command}", "result": result})
        return 0
    if args.command == "remote":
        from .mcp_client import add_server, call_remote_tool, list_remote_tools, load_servers

        if args.remote_command == "add":
            add_server(args.name, list(args.command))
            result = {"added": args.name, "command": list(args.command)}
        elif args.remote_command == "tools":
            result = list_remote_tools(args.server)
        elif args.remote_command == "call":
            try:
                arguments = json.loads(args.arguments)
            except json.JSONDecodeError as exc:
                raise InterfaceError(f"--arguments must be a JSON object: {exc}") from exc
            if not isinstance(arguments, dict):
                raise InterfaceError("--arguments must be a JSON object")
            result = {"output": call_remote_tool(args.server, args.tool, arguments)}
        else:
            result = load_servers()
        emit({"event": f"remote.{args.remote_command}", "result": _serializable(result)})
        return 0
    if args.command == "web":
        from . import web as web_module

        if args.fetch:
            emit({"event": "web.fetch", "result": {"url": args.fetch, "text": web_module.fetch_page(args.fetch)}})
            return 0
        providers = tuple(args.provider) if args.provider else ("duckduckgo", "wikipedia")
        rows = web_module.search(" ".join(args.query), providers=providers)
        emit({"event": "web.search", "result": rows})
        return 0
    if args.command == "serve":
        from .bridge import serve as serve_bridge

        serve_bridge(
            workspace,
            variant=args.variant,
            host=args.host,
            port=args.port,
            origin=args.origin,
        )
        return 0
    if args.command == "variant":
        command = args.variant_command
        if command == "clone":
            result = clone_variant(
                args.name,
                from_name=args.from_name,
                version=args.profile_version,
                engine_requirement=args.engine,
                replace=args.replace,
            ).model_dump(mode="json", by_alias=True)
        elif command == "list":
            result = [item.model_dump(mode="json", by_alias=True) for item in list_variants()]
        elif command == "show":
            result = load_variant(args.name).model_dump(mode="json", by_alias=True)
        elif command == "export":
            result = {"bundle": str(export_variant(args.name, args.output, include_engine=args.include_engine))}
        elif command == "import":
            result = import_variant(args.bundle, replace=args.replace).model_dump(mode="json", by_alias=True)
        else:
            result = {"quarantined": str(remove_variant(args.name))}
        emit({"event": f"variant.{command}", "result": result})
        return 0
    raise InterfaceError(f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _normalize_legacy(list(sys.argv[1:] if argv is None else argv))
    parser = build_parser()
    args = parser.parse_args(arguments)
    emit = Emitter(jsonl=args.json, plain=args.plain)
    try:
        return _run_command(args, emit)
    except (InterfaceError, VariantError, FileNotFoundError, ValueError) as exc:
        emit({"event": "error", "ok": False, "error": str(exc), "type": type(exc).__name__})
        return 2
    except KeyboardInterrupt:
        emit({"event": "interrupted", "ok": False})
        return 130


def entrypoint() -> None:
    raise SystemExit(main())
