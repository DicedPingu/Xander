from __future__ import annotations

import difflib
import hashlib
import json
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import Field

from . import ABILITIES
from .backend import Backend, backend_for, get_backend
from .calibers import role_for_task, route_reason
from .commentary import Commentator
from .executor import ActionExecutor, Approval
from .memory import MemoryStore
from .models import (
    AcceptanceCheck,
    Action,
    ActionKind,
    ActionResult,
    ActionStatus,
    EditBlock,
    GuideStep,
    ModelPlan,
    MissionGuide,
    Phase,
    ResearchBundle,
    StrictModel,
    TaskRecord,
    TaskStatus,
    XanderEvent,
    XanderRequest,
    utc_now,
)
from .policy import (
    apply_edit_blocks,
    changed_paths,
    classify_risk,
    contains_secret_path,
    is_setup_action,
    neutral_intent_contract,
    resolve_inside,
    safe_environment,
    sha256_file,
    snapshot_workspace,
    validate_allowed_paths,
)
from .quick import QuickOrder, parse_quick_order
from .readiness import ToolReadiness, assess
from .research import Researcher
from .skills import SkillRegistry
from .squad import LURKER, MASTER, Squad, choose_form
from .tasks import TaskStore



def _substantive(text: str) -> list[str]:
    """Lines that are actually code. `#include` is not a comment."""

    lines = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("//", "*", "/*", "<!--")):
            continue
        if stripped.startswith("#") and not stripped.startswith(("#include", "#define", "#pragma", "#!")):
            continue
        lines.append(stripped)
    return lines


class Analysis(StrictModel):
    subject: str
    constraints: list[str] = Field(default_factory=list)
    task_type: str = "coding"
    complexity: int = Field(default=2, ge=1, le=5)
    approach: str = ""
    resource_needs: list[str] = Field(default_factory=list)


EventSink = Callable[[XanderEvent], None]

_AUTONOMY_ORDER = {
    "proposal": 0,
    "supervised": 1,
    "full-auto": 2,
}


class Engine:
    def __init__(
        self,
        workspace: Path,
        variant: str = "default",
        autonomy: str | None = None,
        event_sink: EventSink | None = None,
        *,
        backend: Backend | None = None,
        task_store: TaskStore | None = None,
        skill_registry: SkillRegistry | None = None,
        researcher: Researcher | None = None,
        memory: MemoryStore | None = None,
        approve: Approval | None = None,
        hooks: Any = None,
        log_events: bool = False,
        steering: Any = None,
    ) -> None:
        self.workspace = workspace.expanduser().resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("workspace must be a directory")
        self.variant = variant
        from .variants import load_variant

        self.profile = load_variant(variant)
        configured_autonomy = self.profile.autonomy if autonomy is None else autonomy
        self.autonomy = "proposal" if configured_autonomy == "proposal-only" else configured_autonomy
        if self.autonomy not in _AUTONOMY_ORDER:
            raise ValueError(f"unsupported autonomy: {self.autonomy}")
        self.event_sink = event_sink
        if backend is not None:
            self.backend = backend
        elif self.profile is not None:
            self.backend = backend_for(self.profile.model_routing)
        else:
            self.backend = get_backend()
        self.task_store = task_store or TaskStore()
        self.skills = skill_registry or SkillRegistry()
        self.researcher = researcher or Researcher(self.workspace, self.skills)
        namespace = self.profile.memory_namespace if self.profile and self.profile.memory_namespace else variant
        self.memory = memory or MemoryStore(namespace=namespace)
        self.approve = approve
        self.hooks = hooks
        self.log_events = log_events
        # Optional back channel. Called only at safe points; returns
        # {"lines": [...], "constraints": [...], "abort": bool}. None costs nothing.
        self.steering = steering
        self._sequence = 0
        self._last_guide: dict[str, Any] | None = None
        self._written: list[dict[str, Any]] = []
        self._voice = Commentator(voice="off")
        self._squad = Squad(helpers=[])
        self._gear: list[dict[str, str]] = []
        self._last_failures: list[ActionResult] = []  # failed checks of the previous attempt

    def execute(
        self,
        mode: str,
        goal: str,
        constraints: Sequence[str] = (),
        acceptance_checks: Sequence[dict[str, Any] | AcceptanceCheck] = (),
        allowed_paths: Sequence[str] = (),
        timeout: int | None = None,
        caller: str = "human",
        setup_policy: str = "ask",
    ) -> dict[str, Any]:
        checks = [
            value if isinstance(value, AcceptanceCheck) else AcceptanceCheck.model_validate(value)
            for value in acceptance_checks
        ]
        autonomy = "proposal" if caller in {"codex", "claude"} else self.autonomy
        time_budget = self._time_budget(goal) if timeout is None else timeout
        request = XanderRequest(
            caller=caller,
            mode=mode,
            workspace=self.workspace,
            goal=goal,
            constraints=list(constraints),
            acceptance_checks=checks,
            allowed_paths=list(allowed_paths),
            timeout=time_budget or 900,
            time_budget_seconds=time_budget,
            variant=self.variant,
            autonomy=autonomy,
            setup_policy=setup_policy,
        )
        task = self.execute_request(request)
        return self._result_payload(task)

    def execute_request(self, request: XanderRequest) -> TaskRecord:
        workspace = request.workspace.expanduser().resolve(strict=True)
        if workspace != self.workspace:
            raise ValueError("request workspace does not match engine workspace")
        task = self.task_store.create(request)
        self._initialize_guide(task)
        self.task_store.save(task)
        self._sequence = 0
        self._last_guide = None
        self._emit(task, "guide", "mission guide written", {"guide": task.guide.model_dump(mode="json")})
        self._emit(
            task,
            "task",
            "task created",
            {
                "status": task.status,
                "workspace": str(self.workspace),
                "mode": request.mode,
                "autonomy": request.autonomy,
            },
        )
        return self._run_safely(task)

    def resume(
        self,
        task_id: str,
        selected_options: Sequence[str] = (),
        setup_policy: str | None = None,
    ) -> dict[str, Any]:
        task = self.task_store.load(task_id)
        if task.request.workspace.expanduser().resolve(strict=True) != self.workspace:
            raise ValueError("task belongs to a different workspace")
        task.request.autonomy = min(
            (task.request.autonomy, self.autonomy),
            key=_AUTONOMY_ORDER.__getitem__,
        )
        if setup_policy is not None:
            task.request.setup_policy = setup_policy
        if task.guide is None:
            self._initialize_guide(task)
        self.task_store.save(task)
        if task.status == TaskStatus.COMPLETED:
            task.status = TaskStatus.PENDING
            task.failure = ""
            task.plan = None
            task.check_results = []
            task.request.selected_options = []
            if task.guide:
                for step in task.guide.todo:
                    step.state = "todo"
                    step.evidence = ""
                task.guide.current = "Mission reopened; looking for the next improvement."
                task.guide.progress = f"0/{len(task.guide.todo)} complete"
                task.guide.questions = []
                task.guide.result = ""
                task.guide.updated_at = utc_now()
        if selected_options:
            task.request.selected_options = list(dict.fromkeys(selected_options))
            self.task_store.save(task)
        self._sequence = 0
        self._last_guide = None
        self._emit(task, "task", "task resumed", {"status": task.status})
        return self._result_payload(self._run_safely(task, resume=True))

    def _run_safely(self, task: TaskRecord, resume: bool = False) -> TaskRecord:
        try:
            return self._run(task, resume=resume)
        except KeyboardInterrupt:
            task.status = TaskStatus.INTERRUPTED
            task.failure = f"interrupted during {task.phase.value}"
            try:
                task.lesson = self._record_learning(task)
            except Exception:
                pass
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.failure = f"{task.phase.value} failed: {type(exc).__name__}: {exc}"[:2_000]
            try:
                task.lesson = self._record_learning(task)
            except Exception:
                pass
        self.task_store.save(task)
        self._emit(task, "error", task.failure, self._result_payload(task))
        return task

    def _run(self, task: TaskRecord, resume: bool = False) -> TaskRecord:
        task.status = TaskStatus.RUNNING
        if task.request.mode == "implement" and task.snapshot is None and not resume:
            quick = parse_quick_order(task.request.goal, self.workspace)
            if quick is not None:
                return self._run_quick(task, quick)
        form = choose_form(task.request.mode, task.request.goal)
        self._voice = Commentator(
            backend=self.backend,
            memory=self.memory,
            voice=self.profile.voice if self.profile else "chatty",
            speaker=form,
        )
        self._squad = Squad.muster(
            task.request.goal,
            self._complexity(task.request.goal),
            bool(task.request.acceptance_checks),
            task.request.mode,
        )
        self._emit(
            task,
            "delegation",
            "delegation roster ready",
            {
                "agent": "Xander",
                "delegates": self._squad.helpers or ["local execution"],
                "status": "ready",
            },
        )
        if task.snapshot is None:
            self._say(task, "kickoff", goal=task.request.goal, form=form)
            if self._squad.helpers:
                self._emit(
                    task,
                    "voice",
                    "walking with me: " + ", ".join(self._squad.helpers),
                    {"moment": "muster", "speaker": form, "helpers": self._squad.helpers},
                )
            self._phase(task, Phase.ANALYZE, "resolving goal, constraints, and dirty state")
            task.snapshot = snapshot_workspace(self.workspace)
            analysis = self._analyze(task)
            task.subject = analysis.subject
            profile_directives = self.profile.directives if self.profile else []
            hook_constraints, hook_notes = self._hooks_for(task)
            task.effective_constraints = list(
                dict.fromkeys(
                    [
                        *task.request.constraints,
                        *profile_directives,
                        *analysis.constraints,
                        *hook_constraints,
                    ]
                )
            )
            for note in hook_notes:
                self._emit(task, "voice", note, {"moment": "hook", "speaker": form})
            task.evidence.append(
                {
                    "kind": "analysis",
                    "task_type": analysis.task_type,
                    "complexity": analysis.complexity,
                    "approach": analysis.approach,
                    "resource_needs": analysis.resource_needs,
                }
            )
            self._emit(
                task,
                "research",
                "solution and resource strategy mapped",
                {
                    "approach": analysis.approach,
                    "resource_needs": analysis.resource_needs,
                },
            )
            self.task_store.save(task)

        self._phase(task, Phase.RESEARCH, "local truth, current docs, existing solutions")
        if task.research is None or not resume:
            task.research = self.researcher.gather(task.request.goal, task.subject)
            self.task_store.save(task)
        self._emit(
            task,
            "research",
            "research selected",
            {
                "sources": task.research.sources if task.research else [],
                "skills": [item.get("name") for item in (task.research.skills if task.research else [])],
                "tools": sorted((task.research.tools if task.research else {}).keys()),
                "warnings": task.research.warnings if task.research else [],
            },
        )

        if self._squad.lurker and task.research is not None and not resume:
            monica_brief = getattr(self.researcher, "monica_brief", None)
            if callable(monica_brief):
                consultation = self._delegate(
                    task,
                    "Monica",
                    "reference",
                    "read-only reference brief",
                    lambda: monica_brief(task.request.goal),
                )
                if consultation:
                    task.research.documentation = (
                        task.research.documentation
                        + "\n\nMONICA READ-ONLY CONSULTATION (advice; verify against local evidence):\n"
                        + consultation
                    ).strip()
                    task.research.sources = [*task.research.sources, "agent:Monica"]
                    task.evidence.append({"kind": "delegated_research", "agent": "Monica", "chars": len(consultation)})
                    self.task_store.save(task)
            brief = self._delegate(
                task,
                LURKER,
                "critic",
                "constraint research brief",
                lambda: self._squad.lurker_brief(
                    self.backend, task.request.goal, task.research.documentation
                ),
            )
            if brief:
                task.research.documentation = (task.research.documentation + "\n\nLURKER BRIEF:\n" + brief).strip()
                self._emit(
                    task,
                    "voice",
                    f"dug into the tight constraint; {len(brief.splitlines())} technique line(s) on the table",
                    {"moment": "brief", "speaker": "Lurker"},
                )
                task.evidence.append({"kind": "lurker_brief", "chars": len(brief)})
                self.task_store.save(task)

        if task.request.mode == "answer":
            return self._answer(task)

        if task.request.mode in {"inspect", "research"}:
            task.status = TaskStatus.COMPLETED
            self._phase(task, Phase.JUDGE_LOG, "requested evidence bundle produced")
            task.evidence.append({"kind": "result", "verified": True, "scope": task.request.mode})
            self._phase(task, Phase.LEARN, "recording one evidence-linked lesson")
            task.lesson = self._record_learning(task)
            self.task_store.save(task)
            self._emit(task, "result", "read-only task complete", self._result_payload(task))
            return task

        self._phase(task, Phase.SET_UP, "gearing up: the skills, tools and model depth this order needs")
        selected_skills = self._select_gear(task.request.goal)
        self._gear = selected_skills  # the coder reads the same cards as the planner
        readiness = self._readiness(task, [check.argv for check in task.request.acceptance_checks])
        task.tool_readiness = readiness.as_dict()
        task.evidence.append({"kind": "tool-readiness", **readiness.as_dict()})
        self._emit(task, "research", "mission tool preflight completed", readiness.as_dict())
        if readiness.missing and not resume:
            readiness = self._gear_up(task, readiness)
        self._emit(
            task,
            "research",
            self._gear_line(readiness, selected_skills),
            {"gear": True, "tools": sorted(readiness.available), "missing": list(readiness.missing),
             "skills": [item["name"] for item in selected_skills]},
        )
        if readiness.missing and task.request.setup_policy != "allow":
            task.status = TaskStatus.WAITING_APPROVAL if task.request.setup_policy == "ask" else TaskStatus.UNVERIFIED
            task.failure = readiness.blocker()
            self._phase(task, Phase.JUDGE_LOG, task.failure)
            task.lesson = self._record_learning(task, selected_skills)
            self.task_store.save(task)
            self._emit(task, "result", "mission paused for required tooling", self._result_payload(task))
            return task
        resource_map = {
            "strategy": "local truth first; maintained resources next; install or build only when required",
            "skill_hubs": sorted({item.get("group", "uncategorized") for item in selected_skills}),
            "skills": [item["name"] for item in selected_skills],
            "local_tools": sorted((task.research.tools if task.research else {}).keys()),
            "external_sources": list((task.research.sources if task.research else [])[:8]),
        }
        task.evidence.append(
            {
                "kind": "gear",
                **resource_map,
                "variant": task.request.variant,
            }
        )
        self._emit(task, "research", "gear assembled", resource_map)
        self.task_store.save(task)

        previous_plan_hash = self._plan_hash(task.plan) if task.plan else ""
        completed_fingerprints = {
            result.action_hash
            for result in task.results
            if result.status == ActionStatus.OK and result.action_hash
        }
        if task.plan:
            successful_ids = {
                result.action_id for result in task.results if result.status == ActionStatus.OK
            }
            completed_fingerprints.update(
                self._action_hash(action)
                for action in task.plan.actions
                if action.id in successful_ids
            )
        budget = task.request.time_budget_seconds or max(task.request.timeout, 3600)
        deadline = time.monotonic() + budget
        max_attempts = max(3, min(12, budget // 240 + 1))
        attempt_limit = task.attempt + max_attempts
        reuse_existing_plan = resume and task.plan is not None
        planning_failure = ""
        repeated_failures: dict[str, int] = {}
        discovery_hashes: set[str] = set()
        # Every plan tried in this run. The screenshot loop alternated between
        # two failing plans (A, B, A, B); comparing only with the previous one
        # never noticed. A plan seen before is not a new approach.
        seen_plan_hashes: set[str] = set()
        discovery_rounds = 0
        inspected_paths = set()
        while task.attempt < attempt_limit:
            if time.monotonic() >= deadline:
                task.failure = f"work budget exhausted after approximately {budget} seconds"
                task.status = TaskStatus.UNVERIFIED
                break
            if self._steer(task):
                task.failure = "the operator stopped this target"
                task.status = TaskStatus.UNVERIFIED
                break
            task.attempt += 1
            self._phase(task, Phase.WORK, f"attempt {task.attempt}: structured plan and bounded actions")
            if not reuse_existing_plan:
                try:
                    task.plan = self._plan(task, selected_skills, failure=planning_failure)
                    planning_failure = ""
                    for path, verb in self._repair_plan(task):
                        detail = "instead of patching it whole" if verb == "edit" else "instead of patching it"
                        self._emit(task, "action", f"will {verb} {path} {detail}", {})
                except Exception as exc:
                    planning_failure = f"planner output was unusable: {type(exc).__name__}: {exc}"[:1000]
                    task.failure = planning_failure
                    task.evidence.append(
                        {"kind": "judgment", "verified": False, "reason": planning_failure}
                    )
                    self.task_store.save(task)
                    if task.attempt >= attempt_limit:
                        break
                    self._phase(task, Phase.REPEAT, f"repairing malformed plan: {planning_failure}")
                    continue
            reuse_existing_plan = False
            self.task_store.save(task)
            if task.plan:
                self._emit(
                    task,
                    "plan",
                    task.plan.summary,
                    {
                        "decision": task.plan.decision,
                        "why": task.plan.why,
                        "actions": len(task.plan.actions),
                        "checks": len(task.plan.acceptance_checks),
                        "options": [option.id for option in task.plan.options],
                        "steps": [
                            {"id": action.id, "text": action.expected or f"{action.kind}: {action.path}"}
                            for action in task.plan.actions
                        ],
                    },
                )
                self._say(
                    task,
                    "approach",
                    summary=task.plan.summary,
                    actions=len(task.plan.actions),
                    checks=len(task.plan.acceptance_checks),
                    default=next(
                        (option.title for option in task.plan.options if option.selected_by_default), ""
                    ),
                )
                complaint = self._delegate(
                    task,
                    MASTER,
                    "critic",
                    "plan review",
                    lambda: self._squad.master_review(
                        self.backend, task.request.goal, task.plan.summary
                    ),
                )
                if complaint:
                    self._emit(task, "voice", complaint, {"moment": "review", "speaker": MASTER})
                    task.evidence.append({"kind": "master_review", "verdict": complaint})
                    if complaint not in task.effective_constraints:
                        task.effective_constraints.append(f"Master's concern: {complaint}")
                    self.task_store.save(task)
            selection_error = self._option_selection_error(task)
            if selection_error:
                task.status = TaskStatus.WAITING_APPROVAL
                task.failure = selection_error
                self._phase(task, Phase.JUDGE_LOG, selection_error)
                self.task_store.save(task)
                self._emit(
                    task,
                    "approval",
                    selection_error,
                    {
                        "options": [option.model_dump(mode="json") for option in (task.plan.options if task.plan else [])],
                        "selected": task.request.selected_options,
                    },
                )
                return task
            if task.request.mode in {"plan", "test-triage"}:
                task.status = TaskStatus.COMPLETED
                self._phase(task, Phase.JUDGE_LOG, "structured handoff produced; caller retains execution authority")
                task.evidence.append({"kind": "handoff", "verified": True, "scope": task.request.mode})
                self._phase(task, Phase.LEARN, "recording one evidence-linked lesson")
                task.lesson = self._record_learning(task, selected_skills)
                self.task_store.save(task)
                self._emit(task, "result", "plan complete", self._result_payload(task))
                return task
            if task.request.autonomy == "proposal":
                self._repair_plan(task)
            discovery_only = task.request.mode == "implement" and self._is_discovery_plan(task.plan)
            if (
                task.request.autonomy == "proposal"
                and discovery_only
                and discovery_rounds >= self.MAX_DISCOVERY_ROUNDS
                and len(task.request.allowed_paths) == 1
            ):
                try:
                    target = resolve_inside(self.workspace, task.request.allowed_paths[0])
                except (OSError, ValueError):
                    target = None
                if target in inspected_paths and target.is_file() and not contains_secret_path(target):
                    relative = str(target.relative_to(self.workspace))
                    task.plan.actions = [Action(kind=ActionKind.PATCH, path=relative, expected=task.request.goal)]
                    task.plan.acceptance_checks = [AcceptanceCheck(
                        name=f"proposal target exists: {relative}", argv=["test", "-f", relative],
                    )]
                    task.evidence.append({"kind": "proposal-discovery-fallback", "path": relative})
                    discovery_only = False
            lint_failure = ""
            if discovery_only and discovery_rounds < self.MAX_DISCOVERY_ROUNDS:
                for repaired in self._fill_argv(task, task.plan):
                    self._emit(task, "plan", repaired, {})
                discovery_hash = self._plan_hash(task.plan)
                if discovery_hash in discovery_hashes:
                    lint_failure = "invalid plan: planner repeated the same discovery slice"
                else:
                    discovery_hashes.add(discovery_hash)
                    lint_failure = self._lint_plan(task.plan)
                if lint_failure:
                    results = []
                    blocking_failure = lint_failure
                else:
                    results = self._execute_actions(task, completed_fingerprints, only_inspect=True)
                    task.results.extend(results)
                    completed_fingerprints.update(
                        result.action_hash
                        for result in results
                        if result.status == ActionStatus.OK and result.action_hash
                    )
                    blocking_failure = self._blocking_failure(task, results)
                    if not blocking_failure:
                        successful_ids = {result.action_id for result in results if result.status == ActionStatus.OK}
                        for action in task.plan.actions:
                            if (
                                action.id in successful_ids and action.kind == ActionKind.INSPECT
                                and len(action.argv) > 1 and action.cwd in {"", "."}
                                and Path(action.argv[0]).name in {"cat", "sed", "head", "tail"}
                            ):
                                try:
                                    inspected_paths.add(resolve_inside(self.workspace, action.argv[-1]))
                                except (OSError, ValueError):
                                    continue
                        discovery_rounds += 1
                        task.evidence.append(
                            {
                                "kind": "discovery",
                                "round": discovery_rounds,
                                "actions": len(results),
                                "plan_hash": discovery_hash,
                            }
                        )
                        self._emit(
                            task,
                            "research",
                            "discovery slice complete; shaping the implementation from its evidence",
                            {"actions": len(results), "round": discovery_rounds},
                        )
                        self._phase(
                            task,
                            Phase.REPEAT,
                            "discovery complete; choose the change and proof from the inspected evidence",
                        )
                        planning_failure = (
                            "DISCOVERY COMPLETE. The inspection results are in PRIOR ACTION EVIDENCE. "
                            "Now return the implementation actions and required acceptance checks; "
                            "do not repeat discovery."
                        )
                        task.plan = None
                        task.check_results = []
                        reuse_existing_plan = False
                        self.task_store.save(task)
                        continue
            else:
                for repaired in self._fill_argv(task, task.plan):
                    self._emit(task, "plan", repaired, {})
                if task.request.mode == "implement" and not task.request.acceptance_checks:
                    synthesized = self._synthesize_checks(task.plan, task.request.goal)
                    if synthesized:
                        self._emit(
                            task,
                            "plan",
                            "proof derived from the plan itself: " + ", ".join(check.name for check in synthesized),
                            {"checks": [check.model_dump(mode="json") for check in synthesized]},
                        )
                        self.task_store.save(task)
                if task.request.autonomy == "proposal":
                    try:
                        lint_failure = self._prepare_proposal(task)
                    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                        lint_failure = f"proposal preparation failed: {type(exc).__name__}: {exc}"[:1000]
                lint_failure = lint_failure or self._lint_plan(
                    task.plan,
                    require_execution=task.request.mode == "implement",
                    require_checks=(
                        task.request.mode == "implement"
                        and not task.request.acceptance_checks
                    ),
                ) or self._phantom_failure(task.plan)
                if task.plan is not None:
                    seen_plan_hashes.add(self._plan_hash(task.plan))
                if discovery_only and discovery_rounds >= self.MAX_DISCOVERY_ROUNDS:
                    lint_failure = (
                        "invalid plan: discovery budget exhausted; the next plan must make the "
                        "selected change and include deterministic proof"
                    )
                if lint_failure:
                    results = []
                    blocking_failure = lint_failure
                else:
                    results = self._execute_actions(
                        task, completed_fingerprints,
                        only_inspect=task.request.autonomy == "proposal",
                    )
                    task.results.extend(results)
                    completed_fingerprints.update(
                        result.action_hash
                        for result in results
                        if result.status == ActionStatus.OK and result.action_hash
                    )
                    blocking_failure = self._blocking_failure(task, results)
            self.task_store.save(task)

            if task.request.autonomy == "proposal" and not blocking_failure:
                task.status = TaskStatus.UNVERIFIED
                task.failure = "proposal-only handoff; no mutations were applied; acceptance checks were not run"
                self._phase(task, Phase.JUDGE_LOG, task.failure)
                task.lesson = self._record_learning(task, selected_skills)
                self.task_store.save(task)
                self._emit(task, "result", "proposal ready", self._result_payload(task))
                return task

            self._phase(task, Phase.TEST, "running explicit acceptance checks")
            checks = task.request.acceptance_checks or (task.plan.acceptance_checks if task.plan else [])
            task.check_results = self._run_checks(task, checks, deadline=deadline) if not blocking_failure else []
            self.task_store.save(task)

            if self._squad.master:
                checks_passed = sum(1 for item in task.check_results if item.status == ActionStatus.OK)
                self._say(
                    task,
                    "progress",
                    master=MASTER,
                    attempt=task.attempt,
                    actions_ok=sum(1 for item in results if item.status == ActionStatus.OK),
                    actions_total=len(results),
                    checks_passed=checks_passed,
                    checks_total=len(checks),
                    state=blocking_failure or "",
                )

            self._phase(task, Phase.JUDGE_LOG, "independent deterministic judgment")
            success, failure = self._judge(task, checks, blocking_failure or self._stub_output_failure(task))
            if success and self._squad.master:
                verdict = self._delegate(
                    task,
                    MASTER,
                    "critic",
                    "result review",
                    lambda: self._squad.master_verdict(
                        self.backend, task.request.goal, self._failure_context(task)
                    ),
                )
                prior_verdicts = sum(1 for item in task.evidence if item.get("kind") == "master_verdict")
                if verdict:
                    self._emit(task, "voice", verdict, {"moment": "verdict", "speaker": MASTER})
                    task.evidence.append({"kind": "master_verdict", "happy": False, "verdict": verdict})
                    if prior_verdicts == 0 and task.attempt < attempt_limit:
                        success, failure = False, f"the Master is not happy: {verdict}"
                else:
                    task.evidence.append({"kind": "master_verdict", "happy": True})
                    self._emit(task, "voice", "the Master is happy with this", {"moment": "verdict", "speaker": MASTER})
            if success:
                task.status = TaskStatus.COMPLETED
                task.failure = ""
                task.evidence.append({"kind": "judgment", "verified": True, "checks": len(checks)})
                self._say(
                    task,
                    "victory",
                    checks=len(checks),
                    paths=len({path for item in task.results for path in item.changed_paths}),
                )
                self._phase(task, Phase.LEARN, "recording at most one evidence-linked lesson")
                task.lesson = self._record_learning(task, selected_skills)
                self.task_store.save(task)
                self._leave_handoff(task)
                self._leave_folder_map(task)
                self._emit(task, "result", "goal verified", self._result_payload(task))
                return task

            task.failure = failure
            self._say(task, "setback", reason=failure)
            task.status = TaskStatus.FAILED if checks else TaskStatus.UNVERIFIED
            task.evidence.append({"kind": "judgment", "verified": False, "reason": failure})

            signature = self._failure_signature(failure)
            repeated_failures[signature] = repeated_failures.get(signature, 0) + 1
            if repeated_failures[signature] >= self.REPEATED_FAILURE_LIMIT:
                task.failure = (
                    f"stopped after the same setback {repeated_failures[signature]} times: {failure} "
                    "Re-planning around it is not producing new evidence — the goal or the approach "
                    "has to change."
                )
                task.status = TaskStatus.UNVERIFIED
                task.evidence.append(
                    {"kind": "judgment", "verified": False, "reason": task.failure, "repeated": repeated_failures[signature]}
                )
                self.task_store.save(task)
                break

            self.task_store.save(task)
            if task.attempt >= attempt_limit:
                break
            self._phase(task, Phase.REPEAT, f"new evidence requires a changed approach: {failure}")
            try:
                new_plan = self._plan(task, selected_skills, failure=failure)
            except Exception as exc:
                planning_failure = f"planner output was unusable: {type(exc).__name__}: {exc}"[:1000]
                task.failure = planning_failure
                task.plan = None
                self._last_failures = [r for r in task.check_results if r.status != ActionStatus.OK]
                task.check_results = []
                reuse_existing_plan = False
                self.task_store.save(task)
                continue
            new_hash = self._plan_hash(new_plan)
            current_hash = self._plan_hash(task.plan)
            if new_hash in ({current_hash, previous_plan_hash} | seen_plan_hashes) - {""}:
                task.failure = "replanning repeated the same approach as an earlier attempt; stopped"
                break
            if current_hash and new_hash != current_hash:
                change = {
                    "from": current_hash[:12],
                    "to": new_hash[:12],
                    "reason": failure,
                    "summary": new_plan.summary,
                    "attempt": task.attempt + 1,
                }
                task.evidence.append({"kind": "logic_change", **change})
                self._emit(task, "logic_change", "approach changed after new evidence", change)
            previous_plan_hash = current_hash
            task.plan = new_plan
            # The same repair the first plan gets: a body-less patch on a
            # small existing file becomes a rewrite the coder fills. Replans
            # skipped this and hit "patch without a unified patch body" twice.
            for path, verb in self._repair_plan(task):
                detail = "instead of patching it whole" if verb == "edit" else "instead of patching it"
                self._emit(task, "action", f"will {verb} {path} {detail}", {})
            self._last_failures = [r for r in task.check_results if r.status != ActionStatus.OK]
            task.check_results = []
            reuse_existing_plan = True
            resume = True
            self.task_store.save(task)

        if task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.FAILED
        self._phase(task, Phase.LEARN, "recording one evidence-linked failure lesson")
        task.lesson = self._record_learning(task, selected_skills)
        self.task_store.save(task)
        self._leave_handoff(task)
        self._emit(task, "error", task.failure or "task did not verify", self._result_payload(task))
        return task

    def _hooks_for(self, task: TaskRecord) -> tuple[list[str], list[str]]:
        """Narrow the operator's broad hook book down to this mission."""

        try:
            from .hooks import HookBook

            book = self.hooks if self.hooks is not None else HookBook()
            goal = task.request.goal
            return book.constraints_for(goal), book.notes_for(goal, "analyze")
        except Exception:
            return [], []

    def _say(self, task: TaskRecord, moment: str, **context: Any) -> None:
        """Voice one moment through the commentator; silence is always safe."""

        try:
            line = self._voice.say(moment, **context)
        except Exception:
            return
        if line:
            self._emit(task, "voice", line, {"moment": moment, "speaker": self._voice.speaker})

    def _delegate(
        self,
        task: TaskRecord,
        agent: str,
        role: str,
        operation: str,
        call: Callable[[], Any],
    ) -> Any:
        self._emit(
            task,
            "delegation",
            f"delegating {operation} to {agent}",
            {"agent": agent, "role": role, "operation": operation, "status": "started"},
        )
        before = dict(getattr(self.backend, "last_stats", {}) or {})
        error = ""
        try:
            result = call()
        except Exception as exc:
            result = ""
            error = f"{type(exc).__name__}: {exc}"[:500]
        after = dict(getattr(self.backend, "last_stats", {}) or {})
        returned = bool(str(result or "").strip())
        data: dict[str, Any] = {
            "agent": agent,
            "role": role,
            "operation": operation,
            "status": "failed" if error else "completed",
            "outcome": "returned" if returned else "no result",
        }
        if error:
            data["error"] = error
        if after and after != before and after.get("model"):
            data["model"] = after["model"]
        task.evidence.append({"kind": "delegation", **data})
        self._emit(
            task,
            "delegation",
            f"{agent} {operation}: {data['outcome']}",
            data,
        )
        return result

    def _record_model_stats(self, task: TaskRecord, role: str) -> None:
        stats = getattr(self.backend, "last_stats", None)
        if isinstance(stats, dict) and stats:
            task.evidence.append({"kind": "model", "role": role, **stats})
            self._emit(
                task,
                "delegation",
                f"model completed {role} work",
                {
                    "agent": "Xander",
                    "role": role,
                    "operation": "model generation",
                    "status": "completed",
                    "model": stats.get("model", ""),
                },
            )

    def _record_learning(
        self,
        task: TaskRecord,
        selected_skills: Sequence[dict[str, str]] = (),
    ) -> str:
        lesson = self.memory.learn_from(task)
        if not lesson:
            return ""
        group = next(
            (
                str(item.get("group"))
                for item in selected_skills
                if item.get("group") not in {None, "", "core-workflow"}
            ),
            "research-analysis"
            if task.request.mode in {"inspect", "research", "answer"}
            else "code-quality",
        )
        has_verified_check = any(
            result.status == ActionStatus.OK for result in task.check_results
        )
        record = (
            getattr(self.skills, "record_experience", None)
            if task.status == TaskStatus.COMPLETED and has_verified_check
            else None
        )
        if callable(record):
            try:
                hub = record(
                    group,
                    lesson,
                    namespace=self.profile.memory_namespace or self.variant,
                )
                if hub:
                    task.evidence.append(
                        {"kind": "self_extension", "group": group, "skill": hub.get("name", "")}
                    )
            except Exception as exc:
                task.evidence.append(
                    {
                        "kind": "warning",
                        "phase": "learn",
                        "message": f"skill hub update failed: {exc}"[:500],
                    }
                )
        return lesson

    def _run_quick(self, task: TaskRecord, quick: QuickOrder) -> TaskRecord:
        """Do one small order directly: no research, roster, planner, or judge.

        Policy still applies — the executor asks for approval exactly when it
        would for any other action — but a one-line order gets a one-line
        answer: "I created done.txt. What now?"
        """

        task.snapshot = snapshot_workspace(self.workspace)
        task.subject = quick.kind.replace("_", " ")
        task.guide = MissionGuide(
            statement=task.request.goal,
            todo=[GuideStep(id=f"action-{action.id}", text=action.expected) for action in quick.actions],
            current="Small order; doing it directly.",
            progress=f"0/{len(quick.actions)} complete",
        )
        task.plan = ModelPlan(summary=quick.reply, decision="small order, done directly", actions=list(quick.actions))
        task.evidence.append({"kind": "quick", "order": quick.kind, "paths": quick.paths})
        task.phase = Phase.WORK
        self.task_store.save(task)

        for action in quick.actions:
            if action.kind == ActionKind.CREATE and (self.workspace / action.path).exists():
                reply = f"{action.path} already exists here, so I left it alone. What now?"
                task.status = TaskStatus.COMPLETED
                task.evidence.append({"kind": "result", "verified": True, "scope": "quick", "text": reply})
                self.task_store.save(task)
                self._emit(task, "result", reply, {"answer": reply, "quick": True, **self._result_payload(task)})
                return task

        executor = ActionExecutor(
            self.workspace,
            task.snapshot,
            autonomy=task.request.autonomy,
            allowed_paths=task.request.allowed_paths,
            approve=self._approve(task),
            default_timeout=task.request.timeout,
        )
        failure = ""
        for action in quick.actions:
            self._emit(task, "action", f"{action.kind}: {action.expected}", {"action": action.model_dump(mode="json")})
            result = executor.run(action)
            task.results.append(result)
            event_type = "patch" if action.kind == ActionKind.CREATE else "action"
            self._emit(
                task,
                event_type,
                result.status,
                {"result": result.model_dump(mode="json"), "wrote": self._remember_written(result)},
            )
            if result.status != ActionStatus.OK:
                failure = result.reason or result.stderr.strip() or f"{action.expected}: {result.status}"
                break

        if failure:
            task.status = TaskStatus.UNVERIFIED
            task.failure = failure[:2_000]
            reply = f"I couldn't do that: {failure}"
            self.task_store.save(task)
            self._emit(task, "result", reply, {"answer": reply, "quick": True, **self._result_payload(task)})
            return task

        task.status = TaskStatus.COMPLETED
        task.evidence.append({"kind": "result", "verified": True, "scope": "quick", "text": quick.reply})
        self.task_store.save(task)
        self._emit(task, "result", quick.reply, {"answer": quick.reply, "quick": True, **self._result_payload(task)})
        return task

    def _answer(self, task: TaskRecord) -> TaskRecord:
        """Answer mode: research happened; reply in prose, change nothing."""

        self._phase(task, Phase.JUDGE_LOG, "composing a direct answer; no files change")
        role = role_for_task(task.request.mode, task.request.goal, self._complexity(task.request.goal))
        reason = route_reason(task.request.mode, task.request.goal, self._complexity(task.request.goal))
        task.evidence.append({"kind": "model_route", "phase": "answer", "role": role, "reason": reason})
        self._emit(
            task,
            "delegation",
            f"routing answer generation to {role}",
            {
                "agent": "Xander",
                "role": role,
                "operation": "answer generation",
                "status": "selected",
                "reason": reason,
            },
        )
        if not self.backend.available():
            task.status = TaskStatus.UNVERIFIED
            task.failure = "no local model is reachable to compose an answer"
            self.task_store.save(task)
            self._emit(task, "error", task.failure, self._result_payload(task))
            return task
        research = task.research or ResearchBundle()
        reply = self.backend.generate(
            f"QUESTION: {task.request.goal}\n"
            f"WORKSPACE: {self.workspace}\n"
            f"OPERATOR PREFERENCES: {json.dumps(self.memory.preference_lines())}\n"
            f"LOCAL EVIDENCE:\n{research.local_context[:16_000]}\n"
            f"DOCUMENTATION:\n{research.documentation[:8_000]}\n\n"
            "Answer as the operator's capable partner: direct, concrete, grounded in "
            "the evidence above. Match the size of the message: a one-line remark or "
            "complaint gets one or two sentences back, never a paragraph about process; "
            "a real question gets what it needs, under 250 words. Recommend a next move "
            "when one exists. Do not propose file edits — this is advice, not work.",
            role=role,
            think=False,
            timeout=min(450, task.request.timeout),
        ).strip()
        self._record_model_stats(task, role)
        task.status = TaskStatus.COMPLETED
        task.evidence.append({"kind": "answer", "verified": True, "text": reply[:4000]})
        self._phase(task, Phase.LEARN, "recording one evidence-linked lesson")
        task.lesson = self._record_learning(task)
        self.task_store.save(task)
        self._emit(task, "result", reply, {"answer": reply, **self._result_payload(task)})
        return task

    def _analyze(self, task: TaskRecord) -> Analysis:
        goal = " ".join(task.request.goal.split())
        lowered = goal.casefold()
        resource_needs = ["existing workspace files and dirty state", "installed task-relevant tools"]
        if any(token in lowered for token in ("wasm", "webassembly", "web assembly", "wasi")):
            resource_needs.append("installed WebAssembly compiler/runtime and maintained examples")
        if task.request.mode == "implement":
            resource_needs.append("deterministic build, behavior, and artifact checks")
        return Analysis(
            subject=" ".join(goal.split()[:12]),
            constraints=list(task.request.constraints),
            task_type="coding" if task.request.mode == "implement" else task.request.mode,
            complexity=self._complexity(goal),
            approach=(
                "Inspect local truth, map installed and maintained resources, choose the smallest "
                "complete implementation, execute it inside the workspace, then prove behavior "
                "and let the Master judge the evidence."
            ),
            resource_needs=resource_needs,
        )

    def _plan(self, task: TaskRecord, selected_skills: list[dict[str, str]], failure: str = "") -> ModelPlan:
        if not self.backend.available():
            raise RuntimeError("a local model is required to create a coding plan")
        research = task.research or ResearchBundle()
        skills = "\n\n".join(
            f"HUB {item.get('group', 'uncategorized')} · SKILL {item['name']}:\n{item['content']}"
            for item in selected_skills
        )
        lessons = self.memory.relevant_lessons(task.request.goal)
        prior = self._failure_context(task) if task.results or failure else ""
        guide = task.guide.model_dump(mode="json") if task.guide else {}
        prompt = f"""
Create the smallest effective structured coding plan for this exact workspace.

GOAL: {task.request.goal}
MODE: {task.request.mode}
WORKSPACE: {self.workspace}
LIVING MISSION GUIDE: {json.dumps(guide)}
ALLOWED PATHS: {json.dumps(task.request.allowed_paths)}
CONSTRAINTS: {json.dumps(task.effective_constraints)}
STANDING DIRECTIVES: {json.dumps(self.memory.directives())}
RELEVANT VERIFIED LESSONS: {json.dumps(lessons)}
OPERATOR PREFERENCES: {json.dumps(self.memory.preference_lines())}
FAILURE TO CORRECT: {failure}

LOCAL EVIDENCE:
{research.local_context[:4_000]}

CURRENT DOCUMENTATION:
{research.documentation[:3_000]}

TOOL PREFLIGHT:
{json.dumps(task.tool_readiness)}

STORAGE BOUNDARY:
Keep Xander task records, logs, screenshots, and learned skills inside ASKAR/Xander;
keep shared reviewed material inside ASKAR/shared. If this is a new project and no external
workspace was explicitly opened, use ASKAR/Xander/projects/<name>. An external workspace is
for the requested source, AI-configuration, system-configuration, or operating-system changes;
never put Xander management files there. The one exception is a hidden `.ai-context.md`
describing the project for whoever works here next; that is documentation, not a log. For a non-Git external workspace, do not emit a PATCH
action (it applies through git): use CREATE for a missing file, EDIT for one region of an existing
file, or an explicit command whose expected result names the target and proof. Package/toolchain
setup requires the explicit setup policy.

SETUP POLICY: {task.request.setup_policy}
If required tooling is missing and SETUP POLICY is `allow`, make the first executable action an
exact package/toolchain setup action and then verify the installed command. If it is `ask` or
`never`, do not attempt the mission: leave the task paused with the missing tool and install command.

CONTAINED TESTING:
If a check executes untrusted, risky, or third-party code and a container runtime is available, use
`podman run --rm --network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges --tmpfs /tmp:rw ...`.
Never use `--privileged`, host networking or PID namespace, host device access, a container socket,
or a writable host bind mount for containment. If the runtime is missing, report that prerequisite
instead of pretending an unsafe host test is equivalent.

ASSEMBLED GEAR HUBS (general workflow plus task-specific guidance):
{skills}

PRIOR ACTION EVIDENCE:
{prior[:3_000]}

Rules:
- Return only the schema.
- Every action is something the executor can perform, not a prose step or future intention.
- Return at most 6 materially executable actions. Prefer a few complete file actions over many setup or placeholder steps.
- If local evidence is insufficient to choose a safe edit, return one bounded discovery slice containing only INSPECT/NOTE actions. The engine will return their output once. Otherwise, for implement mode include the real PATCH, EDIT, CREATE, COMMAND, or PIPELINE change before validation. Never return an empty plan or a second discovery slice.
- Acceptance checks supplied by the operator are run by the engine after mutations; do not duplicate them as work actions unless a changed-state diagnostic is genuinely needed.
- Use inspect actions before uncertain edits, but omit them when no inspection is needed.
- Every inspect or command action MUST contain a non-empty argv array whose first item is the executable. Never put its command in path, content, expected, or a shell string.
- Valid inspect example: {{"kind":"inspect","cwd":".","argv":["rg","--files"],"expected":"List existing project files"}}.
- Valid command example: {{"kind":"command","cwd":".","argv":["npm","test"],"expected":"Run the project tests"}}.
- Valid create example — this is how you write a NEW file. `path` and `content` are both REQUIRED,
  `argv` stays empty, and `content` holds the entire finished file, not a description of it:
  {{"kind":"create","cwd":".","path":"src/game.py","content":"def main():\\n    print('hi')\\n\\nmain()\\n","expected":"src/game.py exists and runs"}}
- Valid patch example — this is how you change an EXISTING file. `path` and `patch` are both REQUIRED,
  and `patch` must be a real unified diff with a/ and b/ headers and @@ hunks:
  {{"kind":"patch","cwd":".","path":"src/game.py","patch":"--- a/src/game.py\\n+++ b/src/game.py\\n@@ -1,2 +1,2 @@\\n-print('hi')\\n+print('hello')\\n","expected":"the greeting changed"}}
- Valid edit example — this is how you change one region of an EXISTING file, especially a large one,
  without writing a diff or the whole file. `path` and `edits` are both REQUIRED; each `search` must be
  copied verbatim from the current file and must match exactly once, `replace` is what it becomes:
  {{"kind":"edit","cwd":".","path":"src/game.py","edits":[{{"search":"print('hi')","replace":"print('hello')"}}],"expected":"the greeting changed"}}
- A create action without `content`, a patch action without a unified `patch` body, or an edit action
  without `path` and at least one `edits` block, is invalid and will be rejected before anything runs.
  If you intend to write a whole new file, write the whole file.
- All cwd and path values are relative to WORKSPACE (normally cwd "."); never repeat, guess, or misspell the absolute workspace path.
- Never emit shell strings, cd commands, heredocs, interactive editors, or a command action with argv [].
- CREATE automatically creates missing parent directories. For workspace files use CREATE, PATCH, or EDIT; never use mkdir, touch, cp, mv, tee, or an interpreter as a substitute for a file action.
- FAILURE TO CORRECT is evidence, not commentary. Never repeat an action or command that just produced that failure; choose a materially different approach unless the plan first resolves its cause.
- Existing files must change through a unified patch, or an edit action, with paths that already exist. CREATE is only for a path that does not exist.
- Prefer EDIT over PATCH for a large existing file (roughly 200+ lines) or when you are only changing
  one small, clearly identifiable region: a short, exact `search` snippet is far more reliable than a
  hand-written diff. Use PATCH only when several scattered hunks must apply together atomically.
- Every mutating action states its expected observable result and is blocking unless genuinely optional.
- Actions execute in listed order, so normally leave depends_on empty. If used, depends_on may contain only exact action ids declared in this same plan, never labels such as "create:path".
- Assign a parallel_group only to independent read-only inspections that are safe to run concurrently.
- If the operator asks for ideas or a choice, return exactly the requested number of typed options, declare prerequisites/conflicts, set selection_required, and mark one honest recommendation with selected_by_default. If they asked for suggestions only, do not attach mutating actions yet.
- Write option titles and summaries in direct, natural language. Act like the operator's capable partner: make the tradeoffs concrete, recommend a next move, and avoid detached consultant filler.
- Include at least one narrow deterministic acceptance check for implementation work.
- OPERATOR PREFERENCES shape style, defaults, and phrasing; explicit CONSTRAINTS always outrank them.
- Do not touch paths unrelated to the goal. Do not add dependencies or abstractions unless required.
- Use what is installed: python3 with its standard library (urllib, html.parser, json, os, time), curl, and
  plain coreutils. Never plan around selenium, requests, or any package that is not already on this machine;
  "your own browser" means a headless fetch with urllib or curl, never the operator's browser.
- A command that needs logic (loops, timing, parsing) is a small script: CREATE it, then a command action
  runs it as ["python3","name.py"]. Never squeeze a program into a single command line.
- Creating a script is not the goal; its result is. If the goal names an output (seen.txt, title.txt, a
  running program), the plan ends with the step that produces it, and the proof checks that output.
- Scripts run unattended: never input(), never prompt, never wait for a key. Paths are known; hardcode them.
- Name only files that LOCAL EVIDENCE lists or that an earlier action of this plan creates. A command
  on a file that does not exist is rejected before anything runs. Never invent helper scripts to
  remove, check or wrap something one command already does.
- Removing or installing software: LOCAL EVIDENCE says how it is installed (dpkg/apt, snap, pip,
  flatpak, a bare binary). Use that manager's own command, e.g. ["apt-get","purge","-y","name"] or
  ["pip","uninstall","-y","name"]. Root and -y are added for you and the operator is asked once;
  never write sudo yourself. Prove removal with ["test","!","-e","/usr/bin/name"] (the path
  LOCAL EVIDENCE showed), never with a query that fails when the package is gone.
- Compiled languages: a Rust program is a cargo project (["cargo","init","--name","x"] or CREATE
  Cargo.toml + src/main.rs), built with ["cargo","build"] and proven with ["cargo","test"] or by
  running the binary from target/debug/. WebAssembly from text: CREATE a .wat file, then
  ["wat2wasm","name.wat","-o","name.wasm"], then run it from a small node script with WebAssembly.instantiate.
- When a check fails inside a file that exists, the next plan fixes THAT file (CREATE with its full
  corrected content). Never add a second file that does the same job, and never add a second test
  file next to a failing one: one program, one test file, corrected in place.
- `actions` is the plan. It is never empty in implement mode: at least the change and the proof.
  Prose fields stay short (`summary` ≤ 20 words, `decision` ≤ 25, `why` ≤ 40, each `expected` ≤ 15)
  so the room goes to `content`, `patch` and `argv`.
- `decision` names the single next move you chose, in one plain sentence the operator can read.
- `why` says what evidence made that the best move over the alternative you rejected. Cite the local
  evidence, prior failure, or check that decided it; never restate the goal back as the reason.
""".strip()
        complexity = self._complexity(task.request.goal)
        role = role_for_task(
            task.request.mode,
            task.request.goal,
            complexity,
            retry=bool(failure) or task.attempt > 1,
        )
        reason = route_reason(
            task.request.mode,
            task.request.goal,
            complexity,
            retry=bool(failure) or task.attempt > 1,
        )
        task.evidence.append({"kind": "model_route", "phase": "plan", "role": role, "reason": reason})
        self._emit(
            task,
            "delegation",
            f"routing plan generation to {role}",
            {
                "agent": "Xander",
                "role": role,
                "operation": "plan generation",
                "status": "selected",
                "reason": reason,
            },
        )
        raw = self.backend.generate(prompt, role=role, schema=ModelPlan, think=False, timeout=task.request.timeout)
        self._record_model_stats(task, role)
        plan = ModelPlan.model_validate_json(raw)
        self._resolve_dependencies(plan)
        if task.request.mode == "implement" and not plan.acceptance_checks and not task.request.acceptance_checks:
            plan.acceptance_checks = self._infer_checks(plan)
        return plan

    @staticmethod
    def _resolve_dependencies(plan: ModelPlan) -> None:
        """`depends_on: ["add.wat"]` means the step that creates add.wat.

        The planner names dependencies by path or label as often as by id.
        A label that matches an earlier action's path, id or expected text
        becomes that id; anything else is dropped, because actions run in
        listed order anyway and an unresolvable label only blocks the step.
        """

        seen: list[Action] = []
        for action in plan.actions:
            resolved: list[str] = []
            for label in action.depends_on:
                key = label.strip().casefold()
                if not key:
                    continue
                match = next(
                    (
                        earlier
                        for earlier in seen
                        if key in {earlier.id.casefold(), earlier.path.casefold(), earlier.expected.casefold()}
                        or (earlier.path and key.endswith(earlier.path.casefold()))
                    ),
                    None,
                )
                if match is not None and match.id not in resolved:
                    resolved.append(match.id)
            action.depends_on = resolved
            seen.append(action)

    def _readiness(self, task: TaskRecord, commands: Sequence[Sequence[str]] = ()) -> ToolReadiness:
        return assess(task.request.goal, self.workspace, commands, brain=getattr(self.backend, "doctor", None))

    def _gear_up(self, task: TaskRecord, readiness: ToolReadiness) -> ToolReadiness:
        """Upgrade himself for this order before the first action.

        Analysis said what the order needs; if a tool is not on the machine
        he gets it now, deterministically, instead of hoping the planner
        remembers to. Each install goes through the executor, so root and
        package policy still ask the operator (or the ``--yes`` / setup
        policy already given). Returns the readiness after the attempt.
        """

        if not readiness.missing or not task.snapshot:
            return readiness
        policy = task.request.setup_policy
        if policy == "never" or (policy == "ask" and self.approve is None):
            return readiness
        executor = ActionExecutor(
            self.workspace,
            task.snapshot,
            autonomy=task.request.autonomy,
            allowed_paths=task.request.allowed_paths,
            approve=self._approve(task),
            default_timeout=min(task.request.timeout, 1_200),
        )
        import shlex

        installed: list[str] = []
        for need in readiness.needs:
            if need.name not in readiness.missing:
                continue
            for suggestion in need.install:
                try:
                    argv = shlex.split(suggestion)
                except ValueError:
                    continue
                if not argv or argv[0] in {"install", "ollama"} or "`" in suggestion:
                    continue  # prose ("install or enable `x`"), not a command
                action = Action(kind=ActionKind.COMMAND, argv=argv, expected=f"install {need.name}", blocking=False)
                self._emit(task, "action", f"upgrading myself: {need.name} via {' '.join(argv)}", {"action": action.model_dump(mode="json")})
                result = executor.run(action)
                task.results.append(result)
                self._emit(task, "action", result.status, {"result": result.model_dump(mode="json")})
                if result.status == ActionStatus.OK:
                    installed.append(need.name)
                    break
        if installed:
            task.evidence.append({"kind": "gear_up", "installed": installed})
            self._emit(
                task,
                "voice",
                "upgraded myself for this order: installed " + ", ".join(installed),
                {"moment": "gear", "speaker": self._voice.speaker},
            )
        after = self._readiness(task, [check.argv for check in task.request.acceptance_checks])
        task.tool_readiness = after.as_dict()
        self.task_store.save(task)
        return after

    @staticmethod
    def _gear_line(readiness: ToolReadiness, skills: Sequence[dict[str, str]]) -> str:
        tools = sorted(readiness.available)
        parts = []
        if tools:
            parts.append("tools " + ", ".join(tools[:8]))
        names = [str(item.get("name")) for item in skills if item.get("name")]
        if names:
            parts.append("skills " + ", ".join(names[:5]))
        if readiness.missing:
            parts.append("still missing " + ", ".join(readiness.missing))
        return "geared up: " + (" · ".join(parts) if parts else "nothing extra needed")

    @staticmethod
    def _infer_checks(plan: ModelPlan) -> list[AcceptanceCheck]:
        checks = []
        for action in plan.actions:
            if action.kind == ActionKind.INSPECT and action.argv and action.acceptance_check:
                checks.append(AcceptanceCheck(name=action.acceptance_check, argv=action.argv, cwd=action.cwd))
        return checks

    # Three identical setbacks is enough. The fourth is not new evidence.
    REPEATED_FAILURE_LIMIT = 3
    MAX_DISCOVERY_ROUNDS = 1

    @staticmethod
    def _failure_signature(failure: str) -> str:
        """Collapse a setback to what makes it the *same* setback.

        Digits are flattened because the observed loop was Xander dodging
        "create refuses to overwrite an existing path" by creating hello2.py,
        hello3.py ... hello8.py. Those are one wall hit eight times, not eight
        problems, and any counter keyed on the raw string would never notice.
        """

        text = str(failure or "").strip().casefold()
        # Action ids are fresh every plan; "validation action before first
        # mutation: d598ba203f4b" is one wall, not twelve.
        text = re.sub(r"\b[0-9a-f]{12}\b", "#", text)
        return re.sub(r"\d+", "#", text)[:300]

    def _execute_actions(
        self,
        task: TaskRecord,
        completed_fingerprints: set[str],
        only_inspect: bool = False,
    ) -> list[ActionResult]:
        if not task.plan or not task.snapshot:
            return []
        executor = ActionExecutor(
            self.workspace,
            task.snapshot,
            autonomy=task.request.autonomy,
            allowed_paths=task.request.allowed_paths,
            approve=self._approve(task),
            default_timeout=task.request.timeout,
        )
        results: list[ActionResult] = []
        selected = set(task.request.selected_options)
        actions = [
            action
            for action in task.plan.actions
            if (not action.option_id or action.option_id in selected)
            and (not only_inspect or action.kind in {ActionKind.INSPECT, ActionKind.NOTE})
        ]
        completed_ids = {
            action.id
            for action in actions
            if self._action_hash(action) in completed_fingerprints
        }
        # Every action ever run in this mission, by id: results outlive the
        # plan that made them, and the coder needs the command behind a result.
        if not hasattr(self, "_action_index"):
            self._action_index: dict[str, Action] = {}
        self._action_index.update({action.id: action for action in actions})
        index = 0
        while index < len(actions):
            # Safe point: between steps, with nothing in flight to corrupt.
            if self._steer(task):
                break
            action = actions[index]
            if action.id in completed_ids:
                index += 1
                continue
            missing = [dependency for dependency in action.depends_on if dependency not in completed_ids]
            if missing:
                result = ActionResult(
                    action_id=action.id,
                    action_hash=self._action_hash(action),
                    status=ActionStatus.BLOCKED,
                    reason="unmet dependencies: " + ", ".join(missing),
                )
                results.append(result)
                if action.blocking:
                    break
                index += 1
                continue
            group = [action]
            if action.kind == ActionKind.INSPECT and action.parallel_group:
                cursor = index + 1
                while cursor < len(actions):
                    candidate = actions[cursor]
                    if (
                        candidate.kind == ActionKind.INSPECT
                        and candidate.parallel_group == action.parallel_group
                        and candidate.id not in completed_ids
                        and all(dependency in completed_ids for dependency in candidate.depends_on)
                    ):
                        group.append(candidate)
                        cursor += 1
                    else:
                        break
            if len(group) > 1:
                for candidate in group:
                    self._emit(task, "action", f"parallel inspect: {candidate.expected}", {"action": candidate.model_dump(mode="json")})
                with ThreadPoolExecutor(max_workers=min(4, len(group)), thread_name_prefix="xander-inspect") as pool:
                    group_results = list(pool.map(executor.run, group))
                results.extend(group_results)
                completed_ids.update(result.action_id for result in group_results if result.status == ActionStatus.OK)
                for result in group_results:
                    self._emit(task, "action", result.status, {"result": result.model_dump(mode="json")})
                if any(candidate.blocking and result.status != ActionStatus.OK for candidate, result in zip(group, group_results)):
                    break
                index += len(group)
                continue
            self._emit(task, "action", f"{action.kind}: {action.expected}", {"action": action.model_dump(mode="json")})
            filled = self._fill_content(task, action) or self._fill_edit(task, action)
            if filled:
                self._emit(task, "action", filled, {"action": action.model_dump(mode="json")})
            if filled.startswith("blocked:"):
                result = ActionResult(
                    action_id=action.id,
                    action_hash=self._action_hash(action),
                    status=ActionStatus.BLOCKED,
                    reason=filled.removeprefix("blocked:").strip(),
                )
            else:
                result = executor.run(action)
            results.append(result)
            if result.status == ActionStatus.OK:
                completed_ids.add(result.action_id)
                completed_fingerprints.add(self._action_hash(action))
            event_type = "patch" if action.kind in {ActionKind.PATCH, ActionKind.CREATE, ActionKind.EDIT} else "action"
            self._emit(
                task,
                event_type,
                result.status,
                {"result": result.model_dump(mode="json"), "wrote": self._remember_written(result)},
            )
            if action.blocking and result.status != ActionStatus.OK:
                break
            index += 1
        return results

    def _select_gear(self, goal: str) -> list[dict[str, str]]:
        """Quartermaster: layer general and task gear within a context budget."""
        complexity = self._complexity(goal)
        context_budget = 3_000 if complexity <= 2 else 5_000
        assemble = getattr(self.skills, "assemble", None)
        if callable(assemble):
            return assemble(
                goal,
                preferred_groups=self.profile.skill_groups if self.profile else (),
                context_budget=context_budget,
                namespace=self.profile.memory_namespace or self.variant,
            )
        return self.skills.load_selected(goal, limit=max(4, complexity * 2), max_chars=context_budget)

    _NAMED_OUTPUT = re.compile(
        r"\b(?:to|into|in|as|called|named|write|create|save|produce|generate)\s+(?:a\s+|the\s+|an?\s+new\s+)?(?:file\s+)?"
        r"[\"'`]?(?P<file>[\w][\w.-]*\.(?:txt|md|json|csv|py|html|log|yaml|yml|toml|rs|js|wat|wasm))\b",
        re.IGNORECASE,
    )

    @classmethod
    def _goal_outputs(cls, goal: str) -> list[str]:
        """Files the sentence itself promises: "write … to seen.txt" must leave seen.txt."""

        return list(dict.fromkeys(m.group("file") for m in cls._NAMED_OUTPUT.finditer(goal)))

    _NAMED_COMMAND = re.compile(
        r"\b(?:run|execute)\s+(?P<cmd>(?:cargo|python3?|pytest|node|npm|pnpm|go|make|bash|sh|dart|flutter|wat2wasm|rustc)\b"
        r"(?:[^,.;\n]|\.(?=\w))*?)(?=\s*(?:[,;]|\.(?!\w)|\bthen\b|\band\b|$))",
        re.IGNORECASE,
    )

    @classmethod
    def _goal_commands(cls, goal: str) -> list[list[str]]:
        """Commands the sentence itself names as the proof: "run cargo test".

        Observed 2026-09-13: after `cargo test` failed, the next plan quietly
        dropped that check and "completed" on the remaining ones. A command
        the operator wrote into the order is not the planner's to drop.
        """

        commands: list[list[str]] = []
        for match in cls._NAMED_COMMAND.finditer(goal):
            text = match.group("cmd").strip()
            if re.search(r"\b(?:it|them|this|that)\b", text.split()[0] if text else ""):
                continue
            try:
                argv = shlex.split(text)
            except ValueError:
                continue
            if argv and argv not in commands and cls._sane_argv(argv):
                commands.append(argv)
        return commands

    @staticmethod
    def _synthesize_checks(plan: ModelPlan | None, goal: str = "") -> list[AcceptanceCheck]:
        """Derive proof from a plan that forgot to state any.

        A small model reliably produces the change and unreliably produces
        the ``acceptance_checks`` array; rejecting the whole plan three times
        for that omission was the "same wall" loop. Files the plan creates
        or patches must exist afterwards, Python files must compile, and a
        command-only plan is proven by its own exit codes. The synthesized
        checks are appended to the plan so the judge and the log see them.
        """

        if plan is None or not plan.actions:
            return []
        checks: list[AcceptanceCheck] = []
        seen: set[str] = {check.name for check in plan.acceptance_checks}
        for name in Engine._goal_outputs(goal):
            label = f"exists: {name}"
            if label not in seen:
                seen.add(label)
                checks.append(AcceptanceCheck(name=label, argv=["test", "-e", name]))
        existing_argv = [check.argv for check in plan.acceptance_checks]
        for argv in Engine._goal_commands(goal):
            label = f"the order says run: {' '.join(argv)}"
            if label not in seen and argv not in existing_argv:
                seen.add(label)
                checks.append(AcceptanceCheck(name=label, argv=argv, timeout=900))
        if any(check.required for check in plan.acceptance_checks):
            plan.acceptance_checks = [*plan.acceptance_checks, *checks]
            return checks
        for action in plan.actions:
            paths: list[str] = []
            if action.kind in {ActionKind.CREATE, ActionKind.EDIT} and action.path:
                paths = [action.path]
            elif action.kind == ActionKind.PATCH:
                for line in action.patch.splitlines():
                    if line.startswith("+++ ") and not line.startswith("+++ /dev/null"):
                        token = line[4:].split("\t", 1)[0]
                        paths.append(token[2:] if token.startswith("b/") else token)
            for path in paths:
                if path in seen:
                    continue
                seen.add(path)
                checks.append(AcceptanceCheck(name=f"exists: {path}", argv=["test", "-e", path]))
                if path.endswith(".py"):
                    checks.append(AcceptanceCheck(name=f"compiles: {path}", argv=["python3", "-m", "py_compile", path]))
        if not checks and any(
            action.kind in {ActionKind.COMMAND, ActionKind.PIPELINE} for action in plan.actions
        ):
            checks.append(AcceptanceCheck(name="actions exited cleanly", argv=["true"]))
        plan.acceptance_checks = [*plan.acceptance_checks, *checks]
        return checks

    @staticmethod
    def _is_discovery_plan(plan: ModelPlan | None) -> bool:
        return bool(plan and plan.actions) and not plan.acceptance_checks and all(
            action.kind in {ActionKind.INSPECT, ActionKind.NOTE}
            for action in plan.actions
        )

    @staticmethod
    def _lint_plan(
        plan: ModelPlan | None,
        *,
        require_execution: bool = False,
        require_checks: bool = False,
    ) -> str:
        """Reject structurally doomed plans before execution so the replan
        prompt receives a precise correction instead of executor noise."""

        if plan is None:
            return "no plan was produced"
        problems: list[str] = []
        if not plan.actions:
            problems.append("no executable actions")
        has_change = any(
            action.kind in {
                ActionKind.COMMAND,
                ActionKind.PIPELINE,
                ActionKind.PATCH,
                ActionKind.CREATE,
                ActionKind.EDIT,
            }
            for action in plan.actions
        )
        has_required_check = any(check.required for check in plan.acceptance_checks)
        if require_execution and not has_change and not has_required_check:
            problems.append("implementation plan has no executable change")
        if require_checks and not any(check.required for check in plan.acceptance_checks):
            problems.append("implementation plan has no required acceptance checks")
        action_ids = [action.id for action in plan.actions]
        duplicate_action_ids = sorted(
            {action_id for action_id in action_ids if action_ids.count(action_id) > 1}
        )
        if duplicate_action_ids:
            problems.append(
                "duplicate action ids: " + ", ".join(duplicate_action_ids[:6])
            )
        check_names = [check.name for check in plan.acceptance_checks]
        duplicate_check_names = sorted(
            {name for name in check_names if check_names.count(name) > 1}
        )
        if duplicate_check_names:
            problems.append(
                "duplicate acceptance-check names: "
                + ", ".join(duplicate_check_names[:6])
            )
        missing_argv = [
            action.id
            for action in plan.actions
            if action.kind in {ActionKind.INSPECT, ActionKind.COMMAND} and not action.argv
        ]
        if missing_argv:
            problems.append(
                "actions "
                + ", ".join(missing_argv[:6])
                + " are inspect/command but carry no argv; every inspect or command action "
                + "must state its full command as an argv array"
            )
        missing_patch = [action.id for action in plan.actions if action.kind == ActionKind.PATCH and not action.patch]
        if missing_patch:
            problems.append("patch actions without a unified patch body: " + ", ".join(missing_patch[:6]))
        missing_path = [
            action.id
            for action in plan.actions
            if action.kind == ActionKind.CREATE and not action.path
        ]
        if missing_path:
            problems.append("create actions without a target path: " + ", ".join(missing_path[:6]))
        missing_edit = [
            action.id
            for action in plan.actions
            if action.kind == ActionKind.EDIT and not action.path
        ]
        if missing_edit:
            problems.append("edit actions without a target path: " + ", ".join(missing_edit[:6]))
        missing_pipeline = [
            action.id
            for action in plan.actions
            if action.kind == ActionKind.PIPELINE
            and (not action.pipeline or any(not stage for stage in action.pipeline))
        ]
        if missing_pipeline:
            problems.append("pipeline actions without commands: " + ", ".join(missing_pipeline[:6]))
        mutation_seen = False
        for action in plan.actions:
            if action.kind == ActionKind.PATCH and action.patch.strip():
                mutation_seen = True
            elif action.kind == ActionKind.EDIT and action.path:
                mutation_seen = True
            elif action.kind == ActionKind.CREATE and action.path:
                # Content may still be empty here: the coder fills it at
                # execution time. The order is what this check is about.
                mutation_seen = True
            elif not mutation_seen:
                commands = action.pipeline if action.kind == ActionKind.PIPELINE else [action.argv]
                if action.kind == ActionKind.COMMAND and any(Engine._is_validation_command(command) for command in commands):
                    problems.append(f"validation action before first mutation: {action.id}")
                    break
        return "invalid plan: " + "; ".join(problems) if problems else ""

    #: Commands that only make sense on a file that is already there. `rm
    #: remove-stegosuite.sh` for a script nobody wrote is the observed case.
    _CONSUMES_EVERY_OPERAND = {"rm", "cat", "head", "tail", "wc", "unlink", "rmdir"}
    _CONSUMES_FIRST_OPERAND = {"python3", "python", "bash", "sh", "node", "ruby", "perl"}
    _FILE_LIKE = re.compile(r"^[\w][\w./ -]*\.[A-Za-z0-9]{1,5}$")

    def _phantom_failure(self, plan: ModelPlan | None) -> str:
        """Reject a plan that acts on files that do not exist and that no
        earlier step of the same plan creates. The executor would fail the
        same way, but its "No such file" told the planner nothing it acted
        on; this names the phantom and says what the folder really holds."""

        if plan is None:
            return ""
        created: set[str] = set()
        phantoms: list[str] = []
        for action in plan.actions:
            if action.kind == ActionKind.CREATE and action.path:
                created.add(str(Path(action.cwd or ".") / action.path))
                continue
            if action.kind == ActionKind.PATCH:
                continue
            if action.kind == ActionKind.EDIT:
                if not action.path:
                    continue
                relative = str(Path(action.cwd or ".") / action.path)
                if relative in created or Path(action.path).name in {Path(c).name for c in created}:
                    continue
                target = self.workspace / action.cwd / action.path
                if not target.exists():
                    phantoms.append(f"{action.path} (step {action.id})")
                continue
            commands = action.pipeline if action.kind == ActionKind.PIPELINE else [action.argv]
            for argv in commands:
                if not argv:
                    continue
                executable = Path(argv[0]).name
                operands = [token for token in argv[1:] if not token.startswith("-")]
                if executable in self._CONSUMES_EVERY_OPERAND:
                    candidates = operands
                elif executable in self._CONSUMES_FIRST_OPERAND:
                    candidates = [token for token in operands[:1] if self._FILE_LIKE.match(token)]
                else:
                    continue
                for token in candidates:
                    if "://" in token or token in {".", "..", "/"}:
                        continue
                    relative = str(Path(action.cwd or ".") / token)
                    if relative in created or Path(token).name in {Path(c).name for c in created}:
                        continue
                    target = Path(token) if Path(token).is_absolute() else self.workspace / action.cwd / token
                    if not target.exists():
                        phantoms.append(f"{token} (step {action.id})")
        if not phantoms:
            return ""
        try:
            here = sorted(p.name for p in self.workspace.iterdir() if not p.name.startswith("."))[:30]
        except OSError:
            here = []
        return (
            "invalid plan: it acts on files that do not exist and no earlier step creates: "
            + ", ".join(phantoms[:6])
            + ". The folder holds: "
            + (", ".join(here) if here else "nothing")
            + ". Name only files that exist or that this plan creates first."
        )

    @staticmethod
    def _is_validation_command(argv: Sequence[str]) -> bool:
        if not argv:
            return False
        executable = Path(argv[0]).name
        tokens = set(argv[1:])
        if executable in {"pytest", "wasm-validate"}:
            return True
        if executable in {"cargo", "npm", "pnpm", "yarn", "flutter", "dart"}:
            return bool(tokens & {"test", "check", "validate", "analyze"})
        if executable in {"python", "python3"}:
            return bool(tokens & {"pytest", "unittest", "compileall"})
        if executable == "wasm-tools":
            return "validate" in tokens
        return False

    @staticmethod
    def _option_selection_error(task: TaskRecord) -> str:
        if not task.plan or not task.plan.options:
            return ""
        options = {option.id: option for option in task.plan.options}
        selected = set(task.request.selected_options)
        if not selected and not task.plan.selection_required:
            selected = {option.id for option in task.plan.options if option.selected_by_default}
            task.request.selected_options = sorted(selected)
        if task.plan.selection_required and not selected:
            return "option selection required; resume with one or more compatible option ids"
        unknown = selected - set(options)
        if unknown:
            return "unknown option ids: " + ", ".join(sorted(unknown))
        for option_id in selected:
            option = options[option_id]
            missing = set(option.requires) - selected
            conflicts = set(option.conflicts) & selected
            if missing:
                return f"option {option_id} requires: {', '.join(sorted(missing))}"
            if conflicts:
                return f"option {option_id} conflicts with: {', '.join(sorted(conflicts))}"
        return ""

    def _approve(self, task: TaskRecord) -> Approval:
        def approve(action: Action, reason: str) -> bool:
            self._emit(task, "approval", reason, {"action": action.model_dump(mode="json")})
            if self.approve:
                return bool(self.approve(action, reason))
            if (
                task.request.setup_policy == "allow"
                and is_setup_action(action)
                and classify_risk(action).value == "high"
            ):
                self._emit(
                    task,
                    "approval",
                    "setup policy approved the package/toolchain action",
                    {
                        "action": action.model_dump(mode="json"),
                        "status": "approved",
                        "reason": "explicit setup policy: allow",
                    },
                )
                return True
            return False

        return approve

    def _run_checks(
        self,
        task: TaskRecord,
        checks: Sequence[AcceptanceCheck],
        *,
        deadline: float | None = None,
    ) -> list[ActionResult]:
        if not task.snapshot:
            return []
        results = []
        for check in checks:
            action_id = f"check-{hashlib.sha256(check.name.encode()).hexdigest()[:8]}"
            remaining = check.timeout
            if deadline is not None:
                remaining = min(remaining, int(deadline - time.monotonic()))
            if remaining < 1:
                result = ActionResult(
                    action_id=action_id,
                    status=ActionStatus.FAILED,
                    reason="task deadline exhausted before acceptance check",
                )
                results.append(result)
                self._emit(task, "test", result.status, {"name": check.name, "result": result.model_dump(mode="json")})
                if check.required:
                    break
                continue
            executor = ActionExecutor(
                self.workspace,
                task.snapshot,
                autonomy="full-auto",
                allowed_paths=task.request.allowed_paths,
                default_timeout=remaining,
            )
            action = Action(
                id=action_id,
                kind=ActionKind.COMMAND,
                cwd=check.cwd,
                argv=self._ground_check_argv(check.argv, check.cwd),
                expected=check.name,
                blocking=check.required,
            )
            self._emit(task, "test", check.name, {"argv": check.argv, "cwd": check.cwd})
            result = executor.run(action)
            results.append(result)
            self._emit(task, "test", result.status, {"name": check.name, "result": result.model_dump(mode="json")})
            if check.required and result.status != ActionStatus.OK:
                break
        return results

    def _stub_output_failure(self, task: TaskRecord) -> str:
        """A file the order names must hold real content, not an intention.

        repos.md saying "I'll search for 8 repositories" passed "exists:
        repos.md" (2026-09-13). Existence is necessary; a placeholder body
        is a failure the judge names, so the next plan writes the thing.
        """

        if task.request.mode != "implement":
            return ""
        for name in self._goal_outputs(task.request.goal):
            path = self.workspace / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            lines = [line for line in text.splitlines() if line.strip()]
            intention = re.match(r"\s*(?:i(?:'ll| will| am going to| need to)|let me|first,? i)\b", text, re.IGNORECASE)
            if self._PLACEHOLDER.search(text) or intention or (len(lines) <= 1 and len(text) < 200):
                return (
                    f"{name} is a placeholder ({len(lines)} line(s), starts {text.strip()[:60]!r}); "
                    "the order wants its real content, written from the evidence gathered"
                )
            unproven = self._unproven_urls(task, text)
            if unproven:
                return (
                    f"{name} names {len(unproven)} URL(s) that appear in no evidence gathered by this mission: "
                    + ", ".join(unproven[:4])
                    + ". Write only what the searches actually returned; search again if they returned too little."
                )
        return ""

    _URL = re.compile(r"https?://[^\s)>\]\"']+")

    def _unproven_urls(self, task: TaskRecord, text: str) -> list[str]:
        """URLs in an output that no command of this mission ever printed.

        A 9B build asked for eight repositories writes eight plausible URLs
        from memory (two of seven were 404 on 2026-09-13) even with the real
        search results in front of it. Provenance is checked, not requested.
        """

        urls = list(dict.fromkeys(url.rstrip(".,;:") for url in self._URL.findall(text)))
        if not urls:
            return []
        evidence = "\n".join(result.stdout for result in task.results if result.stdout).casefold()
        if not evidence.strip():
            return []  # nothing was gathered; nothing can be checked
        unproven = []
        for url in urls:
            key = url.casefold()
            repo = re.match(r"https?://github\.com/([^/]+/[^/#?]+)", key)
            token = repo.group(1).removesuffix(".git") if repo else key
            if token not in evidence:
                unproven.append(url)
        return unproven

    def _ground_check_argv(self, argv: list[str], cwd: str = ".") -> list[str]:
        """`./wordfreq sample.txt` when the binary is at target/debug/wordfreq:
        the check meant the built program, so run the built program."""

        if not argv or not argv[0].startswith("./"):
            return argv
        base = self.workspace / cwd
        if (base / argv[0]).exists():
            return argv
        name = Path(argv[0]).name
        for candidate in (f"target/debug/{name}", f"target/release/{name}", f"build/{name}", f"bin/{name}"):
            if (base / candidate).is_file():
                return [candidate, *argv[1:]]
        return argv

    @staticmethod
    def _blocking_failure(task: TaskRecord, results: list[ActionResult]) -> str:
        if not task.plan:
            return "no plan"
        actions = {action.id: action for action in task.plan.actions}
        for result in results:
            action = actions.get(result.action_id)
            if action and action.blocking and result.status != ActionStatus.OK:
                return result.reason or result.stderr or f"blocking action {action.id} failed"
        return ""

    @staticmethod
    def _judge(task: TaskRecord, checks: Sequence[AcceptanceCheck], blocking_failure: str) -> tuple[bool, str]:
        if blocking_failure:
            return False, blocking_failure
        if not checks:
            return False, "no explicit acceptance checks were produced"
        required = [check for check in checks if check.required]
        if not required:
            return False, "no required acceptance checks were produced"
        by_id = {result.action_id: result for result in task.check_results}
        expected_ids = [f"check-{hashlib.sha256(check.name.encode()).hexdigest()[:8]}" for check in required]
        # The failed check comes first, with its error line: "did not run" is
        # only the consequence of an earlier failure, and a setback message
        # that never changes hides the progress each attempt actually made.
        failed = [
            f"{check.name} — {Engine._check_detail(by_id[action_id])}"
            for check, action_id in zip(required, expected_ids)
            if action_id in by_id and by_id[action_id].status != ActionStatus.OK
        ]
        if failed:
            return False, "required checks failed: " + "; ".join(failed)
        missing = [check.name for check, action_id in zip(required, expected_ids) if action_id not in by_id]
        if missing:
            return False, "required checks did not run: " + ", ".join(missing)
        return True, ""

    @staticmethod
    def _check_detail(result: ActionResult) -> str:
        """The one line of a failed check's output that says why."""

        text = "\n".join(part for part in (result.reason, result.stdout, result.stderr) if part)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for pattern in ("panicked at", "AssertionError", "assert", "Error", "error", "FAILED", "failed"):
            hit = next((line for line in lines if pattern in line), "")
            if hit:
                return hit[:160]
        return (lines[-1] if lines else f"exit {result.returncode}")[:160]

    def _failure_context(self, task: TaskRecord) -> str:
        rows = []
        for result in [*task.results[-8:], *task.check_results[-5:]]:
            rows.append(
                json.dumps(
                    {
                        "action_id": result.action_id,
                        "status": result.status,
                        "returncode": result.returncode,
                        "stdout": result.stdout[-2000:],
                        "stderr": result.stderr[-2000:],
                        "reason": result.reason,
                    },
                    default=str,
                )
            )
        return "\n".join(rows)

    @staticmethod
    def _complexity(goal: str) -> int:
        score = 1
        lowered = goal.lower()
        score += min(2, len(goal.split()) // 20)
        if any(word in lowered for word in ("architecture", "migrate", "agent", "wasm", "webassembly", "web assembly", "security", "performance")):
            score += 1
        if any(word in lowered for word in ("complete", "full", "hard", "autonomous", "multi-agent")):
            score += 1
        return min(score, 5)

    @staticmethod
    def _time_budget(goal: str) -> int | None:
        match = re.search(
            r"\b(?:work|spend|keep working)(?:\s+(?:for|about|roughly))?\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)\b",
            goal,
            re.IGNORECASE,
        )
        if not match:
            return None
        value = float(match.group(1))
        unit = match.group(2).lower()
        seconds = int(value * (3600 if unit.startswith(("hour", "hr")) else 60))
        return max(60, min(seconds, 86_400))

    @staticmethod
    def _plan_hash(plan: ModelPlan | None) -> str:
        if not plan:
            return ""
        action_ids = {action.id: index for index, action in enumerate(plan.actions)}
        payload = {
            "actions": [
                {
                    "kind": action.kind,
                    "cwd": action.cwd,
                    "argv": action.argv,
                    "pipeline": action.pipeline,
                    "patch": action.patch,
                    "path": action.path,
                    "content": action.content,
                    "expected": " ".join(action.expected.split()).casefold(),
                    "acceptance_check": action.acceptance_check,
                    "blocking": action.blocking,
                    "risk": action.risk,
                    "option_id": action.option_id,
                    "depends_on": [action_ids.get(item, item) for item in action.depends_on],
                    "parallel_group": action.parallel_group,
                }
                for action in plan.actions
            ],
            "checks": [
                {
                    "argv": check.argv,
                    "cwd": check.cwd,
                    "timeout": check.timeout,
                    "required": check.required,
                }
                for check in plan.acceptance_checks
            ],
            "selected_options": sorted(
                option.id for option in plan.options if option.selected_by_default
            ),
            "selection_required": plan.selection_required,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _action_hash(action: Action) -> str:
        payload = action.model_dump(mode="json")
        payload.pop("id", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _initialize_guide(self, task: TaskRecord) -> None:
        goal = task.request.goal
        task.guide = MissionGuide(
            statement=f"Deliver a verified result for: {goal}",
            todo=[
                GuideStep(id="understand", text="Understand the workspace, constraints, and current state"),
                GuideStep(id="shape", text="Shape a small plan and agree on what counts as proof"),
                GuideStep(id="work", text="Make the smallest useful change"),
                GuideStep(id="prove", text="Run checks and judge the result"),
                GuideStep(id="learn", text="Record the result, lesson, and next direction"),
            ],
            current="Guide written; resolving the workspace and definition of done.",
            progress="0/5 complete",
            questions=[],
        )

    def _update_guide(self, task: TaskRecord, event_type: str, message: str, data: dict[str, Any]) -> None:
        guide = task.guide
        if guide is None or event_type == "guide":
            return
        stage_ids = ("understand", "shape", "work", "prove", "learn")
        phase_stage = {
            "analyze": "understand",
            "research": "understand",
            "set_up": "shape",
            "work": "work",
            "test": "prove",
            "judge_log": "prove",
            "learn": "learn",
            "repeat": "work",
        }
        phase = str(task.phase)
        if event_type == "phase":
            stage = phase_stage.get(phase)
            if stage in stage_ids:
                active_index = stage_ids.index(stage)
                for step in guide.todo:
                    if step.id in stage_ids:
                        index = stage_ids.index(step.id)
                        step.state = "done" if index < active_index else "active" if index == active_index else "todo"
            guide.current = message
        elif event_type == "plan":
            guide.current = f"Plan: {message}"
            guide.todo = [step for step in guide.todo if not step.id.startswith("action-")]
            for item in data.get("steps") or []:
                if not isinstance(item, dict):
                    continue
                action_id = str(item.get("id") or "")
                text = " ".join(str(item.get("text") or "").split())
                if not action_id or not text or any(step.id == f"action-{action_id}" for step in guide.todo):
                    continue
                guide.todo.append(GuideStep(id=f"action-{action_id}", text=text))
        elif event_type in {"action", "patch"}:
            action = data.get("action") if isinstance(data.get("action"), dict) else {}
            result = data.get("result") if isinstance(data.get("result"), dict) else {}
            action_id = str(action.get("id") or result.get("action_id") or "")
            for step in guide.todo:
                if action_id and step.id == f"action-{action_id}":
                    result_status = str(result.get("status") or "")
                    step.state = "done" if result_status == "ok" else "blocked" if result_status else "active"
                    step.evidence = str(result.get("reason") or ", ".join(result.get("changed_paths") or []))
            guide.current = message or guide.current
            changed = result.get("changed_paths") or []
            if changed:
                guide.last_change = ", ".join(str(path) for path in changed[:5])
        elif event_type == "test":
            guide.current = f"Proof: {message or data.get('name') or 'acceptance checks'}"
        elif event_type == "approval":
            options = data.get("options") or []
            titles = [str(item.get("title") or item.get("id")) for item in options if isinstance(item, dict)]
            guide.questions = [
                "Which route should I take next? " + "; ".join(titles[:4])
            ] if titles else [message]
            guide.current = "Waiting for your direction before continuing."
        elif event_type == "error":
            guide.current = f"Blocked: {message}"
            guide.questions = ["What should change about the approach before I try again?"]
        elif event_type == "result":
            status = str(data.get("status") or task.status)
            guide.current = "Mission result recorded."
            guide.result = message
            if status in {"completed", "complete"}:
                for step in guide.todo:
                    step.state = "done"
                guide.questions = []
        done = sum(1 for step in guide.todo if step.state == "done")
        guide.progress = f"{done}/{len(guide.todo)} complete"
        guide.updated_at = utc_now()

    #: A file this big cannot be safely rewritten whole by a small coder
    #: model (that is what CREATE's 6_000-byte ceiling already refuses), but
    #: it can still be read once to locate a small edit inside it.
    _EDIT_FILE_CEILING = 200_000

    #: Content that looks like an intention rather than an implementation.
    _PLACEHOLDER = re.compile(
        r"(?:code|implementation|logic|game|body|rest)\s+(?:goes\s+)?here\b|"
        r"\bTODO\b|\bFIXME\b|\bXXX\b|implement(?:ation)?\s+(?:this|me|here|pending)|"
        r"your\s+code\s+here|not\s+implemented|coming\s+soon|\.\.\.",
        re.IGNORECASE,
    )

    @staticmethod
    def _unfence(text: str) -> str:
        """Strip a ```lang fence if the model wrapped the file in one."""

        body = text.strip()
        if not body.startswith("```"):
            return text
        lines = body.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "":
            lines.pop()
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
        return "\n".join(lines) + "\n"

    def _repair_plan(self, task: TaskRecord) -> list[tuple[str, str]]:
        """Turn edits the planner could not express into edits it can.

        An 8B planner reliably knows *what* to change and can fail to emit a
        valid unified diff for it. A body-less patch on a small existing file
        becomes a rewrite request; the coder fills the body and the execution
        path turns it into a preimage-checked unified patch before applying it.
        A body-less patch on a file too large to safely rewrite whole (tui.py,
        engine.py) becomes an edit request instead; the coder fills in one or
        more search/replace blocks rather than the entire file.
        """

        if not task.plan:
            return []
        repaired: list[tuple[str, str]] = []
        for action in task.plan.actions:
            if (
                task.request.autonomy == "proposal"
                and action.kind == ActionKind.INSPECT
                and not action.argv
                and not action.pipeline
                and action.path
            ):
                try:
                    target = resolve_inside(self.workspace, action.path)
                    validate_allowed_paths([target], self.workspace, task.request.allowed_paths)
                except (OSError, ValueError):
                    continue
                if not contains_secret_path(target) and target.is_file() and target.stat().st_size <= 64_000:
                    action.argv = ["sed", "-n", "1,200p", "--", str(target.relative_to(self.workspace))]
            if action.kind != ActionKind.PATCH or action.patch.strip():
                continue
            if action.content.lstrip().startswith(("diff --git ", "--- ")):
                action.patch = action.content
                action.content = ""
                continue
            if task.request.autonomy == "proposal":
                continue
            if not action.path:
                continue
            try:
                target = (self.workspace / action.path).resolve()
                target.relative_to(self.workspace)
            except (OSError, ValueError):
                continue
            if not target.is_file():
                continue
            size = target.stat().st_size
            if size <= 6_000:
                action.kind = ActionKind.CREATE
                action.content = ""  # the coder rewrites it from the file on disk
                repaired.append((action.path, "rewrite"))
            elif size <= self._EDIT_FILE_CEILING:
                action.kind = ActionKind.EDIT
                action.content = ""  # the coder fills action.edits from the file on disk
                repaired.append((action.path, "edit"))
        return repaired

    def _prepare_proposal(self, task):
        if not task.plan:
            return "no plan was produced"
        patches = []
        preimages = {}
        selected = set(task.request.selected_options)
        for action in task.plan.actions:
            if action.option_id and action.option_id not in selected:
                continue
            if action.kind not in {ActionKind.PATCH, ActionKind.CREATE, ActionKind.EDIT}:
                continue
            paths = changed_paths(action, self.workspace)
            if action.path:
                paths.append(resolve_inside(self.workspace, action.path))
            validate_allowed_paths(paths, self.workspace, task.request.allowed_paths)
            if any(contains_secret_path(path) for path in paths):
                return "proposal targets a secret or credential path"
            if action.kind == ActionKind.PATCH and not action.patch.strip():
                if action.content.lstrip().startswith(("diff --git ", "--- ")):
                    action.patch = action.content
                    action.content = ""
                else:
                    if not action.path:
                        return "empty proposal patch has no target path"
                    target = resolve_inside(self.workspace, action.path)
                    if not target.is_file() or target.stat().st_size > 64_000:
                        return f"proposal needs a bounded existing target: {action.path}"
                    relative = str(target.relative_to(self.workspace))
                    before = sha256_file(target)
                    if relative in action.preimage_hashes and action.preimage_hashes[relative] != before:
                        return f"proposal preimage changed for {relative}"
                    action.preimage_hashes[relative] = before
                    prompt = (
                        "Return only an applicable unified diff with a/ and b/ paths, exact context "
                        "and correct hunk counts. No prose or fences. Make the smallest requested edit.\n"
                        f"FILE: {action.path}\nGOAL: {task.request.goal}\n"
                        f"CHANGE: {action.expected}\nCONSTRAINTS: {json.dumps(task.effective_constraints[:8])}\n"
                        f"CURRENT FILE:\n{target.read_text(encoding='utf-8')}"
                    )
                    action.patch = self._unfence(str(self.backend.generate(
                        prompt, role="coder", think=False, timeout=min(450, task.request.timeout)
                    )))
                    self._record_model_stats(task, "coder")
            note = self._fill_content(task, action) or self._fill_edit(task, action)
            if note.startswith("blocked:"):
                return note
            if action.kind == ActionKind.CREATE:
                if not action.path:
                    return "proposal create has no target path"
                target = resolve_inside(self.workspace, action.path)
                if target.exists():
                    return f"proposal create target already exists: {action.path}"
                if not action.content:
                    continue
                action.patch = "".join(difflib.unified_diff(
                    [], action.content.splitlines(keepends=True),
                    fromfile="/dev/null", tofile=f"b/{action.path}",
                ))
                action.kind = ActionKind.PATCH
                action.content = ""
            if action.kind == ActionKind.EDIT:
                if not action.path:
                    return "proposal edit has no target path"
                target = resolve_inside(self.workspace, action.path)
                if not target.is_file():
                    return f"proposal edit target does not exist: {action.path}"
                if not action.edits:
                    continue
                relative = str(target.relative_to(self.workspace))
                current_body = target.read_bytes().decode("utf-8")
                before = sha256_file(target)
                if relative in action.preimage_hashes and action.preimage_hashes[relative] != before:
                    return f"proposal preimage changed for {relative}"
                try:
                    new_body = apply_edit_blocks(current_body, action.edits)
                except ValueError as exc:
                    return f"proposal edit invalid for {action.path}: {exc}"
                action.patch = self._unified_patch(current_body, new_body, action.path)
                action.kind = ActionKind.PATCH
                action.edits = []
                action.preimage_hashes[relative] = before
            if action.kind != ActionKind.PATCH:
                continue
            paths = changed_paths(action, self.workspace)
            if not action.patch.strip() or not paths:
                return "proposal patch has no applicable body or affected paths"
            validate_allowed_paths(paths, self.workspace, task.request.allowed_paths)
            if any(contains_secret_path(path) for path in paths):
                return "proposal targets a secret or credential path"
            for path in paths:
                relative = str(path.relative_to(self.workspace))
                actual = sha256_file(path)
                if relative in action.preimage_hashes and action.preimage_hashes[relative] != actual:
                    return f"proposal preimage changed for {relative}"
                action.preimage_hashes[relative] = actual
                preimages[relative] = actual
            patches.append(action.patch)
        if patches:
            check = subprocess.run(
                ["git", "apply", "--check", "--whitespace=nowarn", "-"],
                input="".join(patches), cwd=self.workspace, env=safe_environment(),
                capture_output=True, text=True, timeout=60, check=False,
            )
            if check.returncode:
                return f"proposal patch check failed: {check.stderr[:1000]}"
            task.evidence.append({
                "kind": "proposal-validation", "applicable": True,
                "patches": len(patches), "preimage_hashes": preimages,
                "acceptance_checks_run": False,
            })
        return ""


    class _Argv(StrictModel):
        argv: list[str] = Field(description="the exact executable and arguments, one string per element")

    class _EditPlan(StrictModel):
        edits: list[EditBlock] = Field(
            description="search/replace blocks; each search copied verbatim from the current file"
        )

    _RUN_FILE = re.compile(
        r"\b(?:run|execute|start|launch|invoke)\b.{0,40}?(?P<file>[\w./-]+\.(?:py|sh|js|rb|pl))\b", re.IGNORECASE
    )
    _INTERPRETER_FOR = {".py": "python3", ".sh": "bash", ".js": "node", ".rb": "ruby", ".pl": "perl"}

    def _deterministic_argv(self, wish: str) -> list[str]:
        """``run fizzbuzz.py`` needs no model: the file names its interpreter."""

        match = self._RUN_FILE.search(wish)
        if not match:
            return []
        name = match.group("file")
        interpreter = self._INTERPRETER_FOR.get(Path(name).suffix.lower())
        return [interpreter, name] if interpreter else []

    @staticmethod
    def _sane_argv(argv: list[str]) -> bool:
        """A command, not prose: short tokens, one line, no inline programs."""

        if not argv or len(argv) > 24:
            return False
        if any("\n" in token or len(token) > 200 for token in argv):
            return False
        if any(token in {"|", "&&", "||", ";", ">", ">>", "<", "`"} for token in argv):
            return False
        if Path(argv[0]).name in {"python", "python3", "node", "ruby", "perl"} and any(
            token in {"-c", "-e", "--eval"} for token in argv[1:]
        ):
            return False
        return not any(marker in " ".join(argv) for marker in ("**", "ERROR", "```", "must not"))

    def _fill_argv(self, task: TaskRecord, plan: ModelPlan | None) -> list[str]:
        """Give an argv-less command step its command, in its own small generation.

        The planner (a 9B build under a JSON schema) reliably decides *what*
        a step does and unreliably fills the ``argv`` array — it describes the
        command in ``expected`` and leaves argv empty. Rejecting the whole plan
        for that omission lost three of five arena runs. A step that names a
        script gets its interpreter deterministically; any other step that
        says what it wants gets one focused call for the exact command, whose
        answer must look like a command and not like prose. A step that says
        nothing stays empty and the lint explains it. Returns log notes.
        """

        if plan is None:
            return []
        notes: list[str] = []
        for action in plan.actions:
            if action.kind not in {ActionKind.INSPECT, ActionKind.COMMAND} or action.argv or action.pipeline:
                continue
            wish = " ".join(f"{action.expected} {action.path}".split())
            if not wish:
                # A bare step: the plan's own decision, then the goal, say what it is for.
                wish = " ".join((plan.decision or plan.summary or task.request.goal).split())[:300]
            # "inspect: stats.py" — a read-only step that names an existing file
            # and nothing else means read it. No model call for that.
            if action.kind == ActionKind.INSPECT and action.path and (self.workspace / action.cwd / action.path).is_file():
                action.argv = ["cat", action.path]
                notes.append(f"step 'inspect {action.path}' → cat {action.path}")
                continue
            argv = self._deterministic_argv(wish)
            if argv:
                action.argv = argv
                notes.append(f"step '{wish[:60]}' → {' '.join(argv)}")
                continue
            prompt = f"""
One step of a plan needs its exact shell command. Return the argv array only.

WORKSPACE: {self.workspace}
GOAL: {task.request.goal}
THIS STEP MUST: {wish}
STEP KIND: {action.kind.value} ({"read-only" if action.kind == ActionKind.INSPECT else "may change files"})
FILES HERE: {", ".join(sorted(p.name for p in self.workspace.iterdir())[:40]) or "(empty)"}

Rules:
- argv is a list of short strings: executable first, then each argument separately. No shell
  syntax (no pipes, redirects, &&), no inline programs (never python3 -c), no prose, no comments.
- Only installed, ordinary tools: python3 (standard library), curl, ls, mv, mkdir, cp, cat, grep.
  No pip installs, no selenium, no third-party packages.
- Paths are relative to the workspace. Never leave the workspace.
- Logic that a single command cannot express belongs in a script the plan creates first,
  then this step runs it: ["python3", "watch.py"].
""".strip()
            try:
                # Local 9B generation needs headroom while the GPU is under load.
                filled = self.backend.generate_model(prompt, self._Argv, role="coder", timeout=240)
            except Exception as exc:
                notes.append(f"could not derive a command for '{wish[:60]}': {type(exc).__name__}")
                continue
            self._record_model_stats(task, "coder")
            argv = [str(token).strip() for token in filled.argv if str(token).strip()]
            if not self._sane_argv(argv):
                notes.append(f"rejected an unusable command for '{wish[:60]}'")
                continue
            action.argv = argv
            notes.append(f"step '{wish[:60]}' → {' '.join(argv)[:120]}")
        return notes

    @staticmethod
    def _is_technical_card(item: dict[str, Any]) -> bool:
        """Cards about a technology help the coder; cards about how to work
        (the task loop, experience hubs) only get copied into files."""

        name = str(item.get("name") or "").casefold()
        group = str(item.get("group") or item.get("category") or "").casefold()
        if group in {"core-workflow", "uncategorized", "code-quality", "research-analysis"}:
            return False
        return not any(word in name for word in ("task-loop", "workflow", "experience", "instinct", "skill-library"))

    def _cut_off(self) -> bool:
        """Did the last generation stop because it ran out of tokens?"""

        stats = getattr(self.backend, "last_stats", None)
        return isinstance(stats, dict) and str(stats.get("done_reason") or "") == "length"

    @staticmethod
    def _unified_patch(current: str, new: str, path: str) -> str:
        """A diff `git apply` accepts. Both sides end with a newline: a last
        line without one gives a hunk line without one, which git calls a
        corrupt patch (seen 2026-09-12 on a coder rewrite)."""

        lines = list(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        out: list[str] = []
        for line in lines:
            if line.endswith("\n"):
                out.append(line)
                continue
            # A final line without a newline: say so the way git does, so the
            # context matches the file on disk instead of "does not apply".
            out.append(line + "\n\\ No newline at end of file\n")
        return "".join(out)

    def _fill_content(self, task: TaskRecord, action: Action) -> str:
        """Write a stubbed file for real, in its own generation.

        A plan must carry every file body inline, so a multi-file goal cannot fit
        in one bounded generation and the planner emits "// code goes here"
        instead. Giving each file its own coder call removes that ceiling: the
        plan decides *what* the files are, the coder decides what is *in* them.
        Returns a short note for the log, or "" when nothing needed doing.
        """

        if action.kind != ActionKind.CREATE or not action.path:
            return ""
        existing = action.content or ""
        current_body = ""
        live_path: Path | None = None
        try:
            live_path = (self.workspace / action.path).resolve()
            live_path.relative_to(self.workspace)
            if live_path.is_file():
                if live_path.stat().st_size > 6_000:
                    return f"blocked: refusing to rewrite large existing file {action.path}"
                current_body = live_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            live_path = None

        if (
            live_path is not None
            and live_path.is_file()
            and existing.strip()
            and not self._PLACEHOLDER.search(existing)
        ):
            if current_body == existing:
                action.kind = ActionKind.NOTE
                action.content = f"{action.path} already contains the requested content"
                return f"{action.path}: already contains the requested content"
            patch = self._unified_patch(current_body, existing, action.path)
            if not patch:
                return f"blocked: could not derive a safe patch for {action.path}"
            action.kind = ActionKind.PATCH
            action.patch = patch
            action.content = ""
            action.preimage_hashes[str(live_path.relative_to(self.workspace))] = hashlib.sha256(
                current_body.encode("utf-8")
            ).hexdigest()
            return f"{action.path}: converted the repeat CREATE into a safe patch"

        # Deterministic, not a size guess. Regenerate only when the planner gave
        # nothing or gave an admitted stub. A short body it wrote on purpose --
        # a marker file, a tiny config -- is its decision and is left alone.
        if existing.strip() and not self._PLACEHOLDER.search(existing):
            return ""
        # An order for an empty file means empty. Never invent a body for it.
        if not existing.strip() and re.search(
            r"\b(?:empty|blank|marker|placeholder)\b", f"{action.expected} {task.request.goal}", re.IGNORECASE
        ):
            return ""

        current = (
            f"THE FILE CURRENTLY CONTAINS (fix it, keep what works):\n{current_body}"
            if current_body
            else ""
        )
        # What went wrong last time is the one thing a rewrite must know.
        # Without it the coder produced the same invalid file three times.
        failed = ""
        if task.failure or task.results or task.check_results:
            # The build error is usually in a failed *check*, not an action.
            tail = next(
                (
                    "\n".join(p for p in (result.reason, result.stdout[-700:], result.stderr[-500:]) if p)
                    for result in [
                        *reversed(task.check_results),
                        *reversed(getattr(self, "_last_failures", [])),
                        *reversed(task.results),
                    ]
                    if result.status != ActionStatus.OK and (result.stderr or result.reason or result.stdout)
                ),
                "",
            )
            detail = "\n".join(part for part in (task.failure[:600], tail) if part)
            if detail.strip():
                failed = f"WHAT FAILED LAST TIME (if it concerns this file, fix exactly this):\n{detail}"
        # Only a failure that names this file makes "the same body again" a
        # failure; when run.js broke, an unchanged add.wat is simply correct.
        failed_here = bool(failed) and Path(action.path).name in failed
        # What the earlier steps found: a research file written without the
        # search results it was meant to hold is a stub, however well phrased.
        index = getattr(self, "_action_index", {})
        gathered = "\n".join(
            f"$ {' '.join(index[result.action_id].argv)}\n{result.stdout[-1500:]}"
            for result in task.results[-10:]
            if result.status == ActionStatus.OK
            and result.stdout.strip()
            and result.action_id in index
            and index[result.action_id].argv
        )[-5000:]
        # The cards gear-up selected: the same reference the planner read.
        # A .wat file written from memory was wrong twice; written next to
        # the canonical module it is right.
        cards = "\n\n".join(
            f"CARD {item.get('name')}:\n{str(item.get('content') or '')[:2500]}"
            for item in list(getattr(self, "_gear", []))
            if item.get("content") and self._is_technical_card(item)
        )[:5000]
        prompt = f"""
Write the complete, finished contents of one file. Output the file body and nothing else:
no prose, no explanation, no markdown fence.

FILE: {action.path}
THIS FILE MUST: {action.expected or "fulfil its role in the goal below"}
OVERALL GOAL: {task.request.goal}
CONSTRAINTS: {json.dumps(task.effective_constraints[:8])}
PLANNER SKETCH (a hint only — replace it with the real thing):
{existing[:1200]}
{current}
{failed}
{("EVIDENCE GATHERED BY EARLIER STEPS (use these real results; never invent names, URLs or numbers):" + chr(10) + gathered) if gathered else ""}
{("REFERENCE CARDS — examples of correct forms for this technology. They are notes, not the file; never copy a card's text or its --- header into the file:" + chr(10) + cards) if cards else ""}

Requirements:
- Real, working, complete code. Never a stub, placeholder, TODO, or "code goes here".
- It must compile or run as written, with no missing functions you meant to add later.
- Only the standard library unless the goal names a dependency.
- If it is a program a person uses, it must actually be usable end to end.
- Tests use tiny literal data (two or three items, no punctuation) so every expected value is
  obviously right by inspection; count them before writing the assertion.
""".strip()
        try:
            # A file that just failed gets a thinking pass: the same 9B build
            # that wrote it wrong from the hip usually gets it right on reflection.
            raw = self.backend.generate(
                prompt, role="coder", think=bool(failed_here), timeout=max(300, task.request.timeout)
            )
            if self._cut_off():
                # The answer hit the token limit: a degenerate run of the same
                # token, or binary data inlined as an array (run.js, 2026-09-12).
                # A truncated file is never a file. One retry, told exactly why.
                self._emit(task, "action", f"{action.path}: the coder ran out of room; retrying shorter", {})
                raw = self.backend.generate(
                    prompt
                    + "\n\nYOUR PREVIOUS ANSWER WAS CUT OFF AT THE TOKEN LIMIT. Write a SHORT, complete file: "
                    "no inlined binary or numeric data (read files from disk instead), no repetition, "
                    "no long tables. Under 80 lines.",
                    role="coder",
                    think=False,
                    timeout=max(300, task.request.timeout),
                )
                if self._cut_off():
                    return f"blocked: the coder could not finish {action.path} within the token limit twice"
        except Exception as exc:
            return f"blocked: could not write {action.path}: {type(exc).__name__}"
        body = self._unfence(str(raw))
        # Only reject an empty answer. Length is not quality: correct, concise
        # code is routinely SHORTER than the verbose stub it replaces, and a
        # "must not shrink" rule silently kept every stub in place.
        if not body.strip():
            return f"blocked: the coder returned nothing for {action.path}"
        if live_path is not None and live_path.is_file():
            patch = self._unified_patch(current_body, body, action.path)
            if not patch and failed_here:
                return f"blocked: the coder produced the same {action.path} that just failed; the approach must change"
            if not patch:
                # The file already holds exactly what the coder would write.
                # That is the goal met, not a failure to act.
                action.kind = ActionKind.NOTE
                action.content = f"{action.path} already holds the finished content"
                return f"{action.path}: already correct, left as is"
            action.kind = ActionKind.PATCH
            action.patch = patch
            action.content = ""
            action.preimage_hashes[str(live_path.relative_to(self.workspace))] = hashlib.sha256(
                current_body.encode("utf-8")
            ).hexdigest()
            self._record_model_stats(task, "coder")
            return f"{action.path}: wrote {len(body.splitlines())} lines as a safe patch"
        action.content = body
        self._record_model_stats(task, "coder")
        return f"{action.path}: wrote {len(body.splitlines())} lines with the coder"

    def _fill_edit(self, task: TaskRecord, action: Action) -> str:
        """Give an EDIT action its search/replace blocks, in its own generation.

        The planner may name a large existing file and describe the change in
        `expected` without writing exact `edits` — or `_repair_plan` converted
        a body-less PATCH here because the file was too large to rewrite whole.
        The coder reads the real file once and returns typed search/replace
        blocks; each `search` is checked against the live file before the
        action runs, so a bad guess is caught here rather than corrupting it.
        """

        if action.kind != ActionKind.EDIT or not action.path or action.edits:
            return ""
        try:
            target = (self.workspace / action.path).resolve()
            target.relative_to(self.workspace)
        except (OSError, ValueError):
            return f"blocked: edit target outside workspace: {action.path}"
        if not target.is_file():
            return f"blocked: edit target does not exist: {action.path}"
        size = target.stat().st_size
        if size > self._EDIT_FILE_CEILING:
            return f"blocked: {action.path} is too large to edit ({size} bytes)"
        current_bytes = target.read_bytes()
        try:
            current_body = current_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return f"blocked: edit target is not UTF-8 text: {action.path}"

        failed = ""
        if task.failure or task.results or task.check_results:
            tail = next(
                (
                    "\n".join(p for p in (result.reason, result.stdout[-700:], result.stderr[-500:]) if p)
                    for result in [
                        *reversed(task.check_results),
                        *reversed(getattr(self, "_last_failures", [])),
                        *reversed(task.results),
                    ]
                    if result.status != ActionStatus.OK and (result.stderr or result.reason or result.stdout)
                ),
                "",
            )
            detail = "\n".join(part for part in (task.failure[:600], tail) if part)
            if detail.strip():
                failed = f"WHAT FAILED LAST TIME (if it concerns this file, fix exactly this):\n{detail}"

        prompt = f"""
Return one or more search/replace edits to an existing file. Each `search` must be copied
verbatim from CURRENT FILE below and must occur exactly once; `replace` is what it becomes.

FILE: {action.path}
THIS EDIT MUST: {action.expected or "fulfil its role in the goal below"}
OVERALL GOAL: {task.request.goal}
CONSTRAINTS: {json.dumps(task.effective_constraints[:8])}
{failed}

CURRENT FILE:
{current_body}

Requirements:
- `search` is copied exactly from CURRENT FILE, including indentation and line breaks. A search
  that does not match exactly, or matches more than once, is rejected and nothing is written.
- Keep each block to the smallest region that must change; never search for or replace the
  entire file.
- The result must be real, working code: no stub, placeholder, TODO, or "code goes here".
""".strip()
        try:
            filled = self.backend.generate_model(
                prompt, self._EditPlan, role="coder", timeout=max(300, task.request.timeout)
            )
        except Exception as exc:
            return f"blocked: could not derive edits for {action.path}: {type(exc).__name__}"
        self._record_model_stats(task, "coder")
        edits = [edit for edit in filled.edits if edit.search.strip()]
        if not edits:
            return f"blocked: the coder returned no usable edits for {action.path}"
        try:
            apply_edit_blocks(current_body, edits)
        except ValueError as exc:
            return f"blocked: {action.path}: {exc}"
        action.edits = edits
        action.preimage_hashes[str(target.relative_to(self.workspace))] = hashlib.sha256(
            current_bytes
        ).hexdigest()
        return f"{action.path}: wrote {len(edits)} edit block(s)"

    def _remember_written(self, result: ActionResult) -> list[dict[str, Any]]:
        """Describe what was written, and keep it for the handoff note."""

        described = self._describe_written(result)
        known = {item["path"] for item in self._written}
        self._written.extend(item for item in described if item["path"] not in known)
        return described

    def _leave_handoff(self, task: TaskRecord) -> None:
        """Leave a hidden note so the next agent here does not start blind."""

        if not self._written:
            return
        try:
            from .handoff import from_task, write

            path = write(self.workspace, from_task(self.workspace, task, self._written))
        except Exception:
            return
        if path is not None:
            self._emit(task, "action", f"left notes for the next agent in {path.name}", {})


    def _describe_written(self, result: ActionResult) -> list[dict[str, Any]]:
        """Say what each written file now *is*, not merely that a write happened.

        "6 creates, all ok" is a report about actions. The operator needs to know
        they got six stubs. A file whose body is "// code goes here" is a
        placeholder and saying so is the single most useful thing in the log.
        """

        described: list[dict[str, Any]] = []
        for raw in list(result.changed_paths or [])[:8]:
            entry: dict[str, Any] = {"path": str(raw)}
            try:
                path = (self.workspace / str(raw)).resolve()
                path.relative_to(self.workspace)  # never stat outside the workspace
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                described.append(entry)
                continue
            body = [line for line in text.splitlines() if line.strip()]
            entry["bytes"] = len(text.encode("utf-8", errors="replace"))
            entry["lines"] = len(body)
            if self._PLACEHOLDER.search(text) or len(_substantive(text)) <= 5:
                entry["placeholder"] = True
            described.append(entry)
        return described

    def _steer(self, task: TaskRecord) -> bool:
        """Apply anything the operator said. Safe points only; never mid-step.

        Returns True when the operator asked to abandon what is running. New
        restrictions join ``effective_constraints``, which is what the planner,
        coder and critic all read — so one note redirects every role at once.
        """

        if self.steering is None:
            return False
        try:
            report = self.steering() or {}
        except Exception:
            return False
        for line in report.get("lines") or []:
            self._emit(task, "steering", str(line), {"moment": task.phase.value})
        added = False
        for constraint in report.get("constraints") or []:
            text = " ".join(str(constraint).split())
            if text and text not in task.effective_constraints:
                task.effective_constraints.append(text)
                added = True
        if added:
            self.task_store.save(task)
        return bool(report.get("abort"))

    def _phase(self, task: TaskRecord, phase: Phase, message: str) -> None:
        task.phase = phase
        self.task_store.save(task)
        self._emit(task, "phase", message, {"phase": phase})

    def _emit(self, task: TaskRecord, event_type: str, message: Any, data: dict[str, Any]) -> None:
        self._update_guide(task, event_type, str(message), data)
        self.task_store.save(task)
        try:
            from .projectlog import append_project_event

            details = f"[{event_type.upper()}] {message}"
            if event_type in {"result", "error"}:
                details += f" status={task.status.value}"
                if task.lesson:
                    details += f" lesson={task.lesson}"
            append_project_event(self.workspace, task.id, details)
        except Exception:
            pass
        if getattr(self, "log_events", False):
            try:
                from .eventlog import append_global_event

                append_global_event(self.variant, self.workspace, task.id, event_type, message, data)
            except Exception:
                pass
        event_data = dict(data)
        if task.guide:
            # Attach the guide only when it actually moved. Sending the whole
            # blob with every event made it ~79% of the stream and buried the
            # one line that carried news.
            snapshot = task.guide.model_dump(mode="json")
            comparable = {key: value for key, value in snapshot.items() if key != "updated_at"}
            if comparable != self._last_guide:
                event_data["guide"] = snapshot
                self._last_guide = comparable
        self._sequence += 1
        event = XanderEvent(
            run_id=task.id,
            sequence=self._sequence,
            type=event_type,
            phase=task.phase,
            attempt=task.attempt,
            message=str(message),
            data=event_data,
        )
        if self.event_sink:
            try:
                self.event_sink(event)
            except Exception as exc:
                task.evidence.append(
                    {"kind": "warning", "phase": task.phase, "message": f"event sink failed: {exc}"[:500]}
                )

    def _leave_folder_map(self, task: TaskRecord) -> None:
        """A finished mission leaves a `.folder` map behind, once, never over the operator's."""

        if task.status != TaskStatus.COMPLETED or task.request.mode != "implement":
            return
        try:
            from .paths import agent_dir
            from .world import ensure_folder_map

            if self.workspace == agent_dir().resolve() or agent_dir().resolve() in self.workspace.parents:
                return
            written = ensure_folder_map(self.workspace, task.subject or task.request.goal[:80])
        except Exception:
            return
        if written is not None:
            task.evidence.append({"kind": "folder_map", "path": str(written)})
            self._emit(task, "action", "left a .folder map of this workspace", {"path": str(written)})

    def _result_payload(self, task: TaskRecord) -> dict[str, Any]:
        return {
            "ok": task.status == TaskStatus.COMPLETED,
            "status": task.status,
            "task_id": task.id,
            "task": task.model_dump(mode="json", by_alias=True),
            "handoff": self.task_store.handoff(task).model_dump(mode="json", by_alias=True),
        }

    def doctor(self) -> dict[str, Any]:
        backend_doctor = getattr(self.backend, "doctor", lambda: {"available": self.backend.available()})()
        snapshot = snapshot_workspace(self.workspace)
        try:
            skills = self.skills.doctor()
        except Exception as exc:
            skills = {"error": str(exc)}
        return {
            "schema": "xander.capabilities/v1",
            "workspace": str(self.workspace),
            "variant": self.variant,
            "autonomy": self.autonomy,
            "mantra": ["Analyze", "Research", "Set yourself up", "Work", "Test", "Judge/log", "Learn", "Repeat"],
            "abilities": list(ABILITIES),
            "backend": backend_doctor,
            "repository": {
                "root": snapshot.root,
                "git": snapshot.git,
                "branch": snapshot.branch,
                "dirty_count": snapshot.dirty_count,
                "dirty_sample": [
                    str(Path(path).relative_to(self.workspace))
                    for path in list(snapshot.dirty)[:20]
                    if Path(path).is_relative_to(self.workspace)
                ],
            },
            "skills": skills,
            "interfaces": ["tui", "jsonl-cli", "mcp-stdio"],
            "content_policy": neutral_intent_contract(),
        }

    def list_tasks(self) -> list[dict[str, Any]]:
        return [
            task.model_dump(mode="json", by_alias=True)
            for task in self.task_store.list()
            if task.request.workspace.expanduser().resolve(strict=False) == self.workspace
        ]

    def show_task(self, task_id: str) -> dict[str, Any]:
        task = self.task_store.load(task_id)
        if task.request.workspace.expanduser().resolve(strict=False) != self.workspace:
            raise ValueError("task belongs to a different workspace")
        return self._result_payload(task)
