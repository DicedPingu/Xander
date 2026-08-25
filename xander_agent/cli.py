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
    migration_dir,
    state_dir,
    tasks_dir,
    variants_dir,
)
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
_COMMANDS = {"run", "plan", "resume", "tasks", "doctor", "skills", "variant", "army", "stats", "mcp", "tui"}


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
        if event_name and len(payload) <= 3:
            detail = payload.get("message") or payload.get("phase") or ""
            print(f"{event_name}: {detail}".rstrip())
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))


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
) -> Any:
    engine_type = _load_engine()
    return engine_type(
        workspace=workspace.expanduser().resolve(),
        variant=variant,
        autonomy=autonomy,
        event_sink=event_sink,
    )


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
    task_id: str | None = None,
    selected_options: Sequence[str] = (),
    event_sink: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Call the core through its public interface without importing it eagerly."""

    engine = create_engine(
        workspace,
        variant=variant,
        autonomy=autonomy,
        event_sink=event_sink,
    )
    if mode == "resume":
        if not task_id:
            raise InterfaceError("resume requires a task id")
        return _serializable(engine.resume(task_id, selected_options=tuple(selected_options)))
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
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="execute a goal through the complete mantra loop")
    run.add_argument("goal", nargs="+")
    _add_request_options(run)

    plan = subparsers.add_parser("plan", help="research and prepare a proposal without applying changes")
    plan.add_argument("goal", nargs="+")
    _add_request_options(plan)

    resume = subparsers.add_parser("resume", help="resume a persisted task")
    resume.add_argument("task_id")
    resume.add_argument("--select", action="append", default=[], help="compatible plan option id; repeatable")

    tasks = subparsers.add_parser("tasks", help="inspect persisted tasks")
    task_commands = tasks.add_subparsers(dest="tasks_command", required=True)
    task_commands.add_parser("list")
    task_show = task_commands.add_parser("show")
    task_show.add_argument("task_id")

    subparsers.add_parser("doctor", help="check the interface, state paths, models, and tools")

    skills = subparsers.add_parser("skills", help="query Xander's compact skill registry")
    skill_commands = skills.add_subparsers(dest="skills_command", required=True)
    skill_commands.add_parser("refresh")
    skill_commands.add_parser("list")
    skill_search = skill_commands.add_parser("search")
    skill_search.add_argument("query")
    skill_commands.add_parser("doctor")

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

    subparsers.add_parser("mcp", help="serve Xander tools over MCP stdio")
    subparsers.add_parser("tui", help="open the full-screen terminal interface")
    return parser


def _normalize_legacy(argv: list[str]) -> list[str]:
    """Keep ``xander GOAL`` and the former plan/yolo flags working."""

    if not argv:
        return argv
    flags = {"--json", "--plain", "--version"}
    valued = {"--workspace", "--variant", "--autonomy", "--caller"}
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
    if args.command in {"run", "plan"}:
        mode = "implement" if args.command == "run" else "plan"
        result = invoke_engine(
            mode,
            workspace=workspace,
            goal=" ".join(args.goal),
            variant=args.variant,
            autonomy="proposal-only" if args.command == "plan" else args.autonomy,
            caller=args.caller,
            constraints=args.constraint,
            acceptance_checks=args.accept,
            allowed_paths=args.allow,
            timeout=args.timeout,
            event_sink=emit,
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
            event_sink=emit,
        )
        emit({"event": "result", "result": result})
        return 0 if _is_success(result) else 1
    if args.command == "tasks":
        engine = create_engine(workspace, variant=args.variant, autonomy="proposal-only")
        result = engine.list_tasks() if args.tasks_command == "list" else engine.show_task(args.task_id)
        emit({"event": f"tasks.{args.tasks_command}", "result": _serializable(result)})
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
