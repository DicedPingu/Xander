from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import Field

from . import ABILITIES
from .backend import Backend, OllamaBackend, get_backend
from .executor import ActionExecutor, Approval
from .memory import MemoryStore
from .models import (
    AcceptanceCheck,
    Action,
    ActionKind,
    ActionResult,
    ActionStatus,
    ModelPlan,
    Phase,
    ResearchBundle,
    StrictModel,
    TaskRecord,
    TaskStatus,
    XanderEvent,
    XanderRequest,
)
from .policy import neutral_intent_contract, snapshot_workspace
from .research import Researcher
from .skills import SkillRegistry
from .tasks import TaskStore


class Analysis(StrictModel):
    subject: str
    constraints: list[str] = Field(default_factory=list)
    task_type: str = "coding"
    complexity: int = Field(default=2, ge=1, le=5)


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
            self.backend = OllamaBackend(models=self.profile.model_routing)
        else:
            self.backend = get_backend()
        self.task_store = task_store or TaskStore()
        self.skills = skill_registry or SkillRegistry()
        self.researcher = researcher or Researcher(self.workspace, self.skills)
        namespace = self.profile.memory_namespace if self.profile and self.profile.memory_namespace else variant
        self.memory = memory or MemoryStore(namespace=namespace)
        self.approve = approve
        self._sequence = 0

    def execute(
        self,
        mode: str,
        goal: str,
        constraints: Sequence[str] = (),
        acceptance_checks: Sequence[dict[str, Any] | AcceptanceCheck] = (),
        allowed_paths: Sequence[str] = (),
        timeout: int | None = None,
        caller: str = "human",
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
        )
        task = self.execute_request(request)
        return self._result_payload(task)

    def execute_request(self, request: XanderRequest) -> TaskRecord:
        workspace = request.workspace.expanduser().resolve(strict=True)
        if workspace != self.workspace:
            raise ValueError("request workspace does not match engine workspace")
        task = self.task_store.create(request)
        self._sequence = 0
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

    def resume(self, task_id: str, selected_options: Sequence[str] = ()) -> dict[str, Any]:
        task = self.task_store.load(task_id)
        if task.request.workspace.expanduser().resolve(strict=True) != self.workspace:
            raise ValueError("task belongs to a different workspace")
        task.request.autonomy = min(
            (task.request.autonomy, self.autonomy),
            key=_AUTONOMY_ORDER.__getitem__,
        )
        self.task_store.save(task)
        if task.status == TaskStatus.COMPLETED:
            return self._result_payload(task)
        if selected_options:
            task.request.selected_options = list(dict.fromkeys(selected_options))
            self.task_store.save(task)
        self._sequence = 0
        self._emit(task, "task", "task resumed", {"status": task.status})
        return self._result_payload(self._run_safely(task, resume=True))

    def _run_safely(self, task: TaskRecord, resume: bool = False) -> TaskRecord:
        try:
            return self._run(task, resume=resume)
        except KeyboardInterrupt:
            task.status = TaskStatus.INTERRUPTED
            task.failure = f"interrupted during {task.phase.value}"
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.failure = f"{task.phase.value} failed: {type(exc).__name__}: {exc}"[:2_000]
        self.task_store.save(task)
        self._emit(task, "error", task.failure, self._result_payload(task))
        return task

    def _run(self, task: TaskRecord, resume: bool = False) -> TaskRecord:
        task.status = TaskStatus.RUNNING
        if task.snapshot is None:
            self._phase(task, Phase.ANALYZE, "resolving goal, constraints, and dirty state")
            task.snapshot = snapshot_workspace(self.workspace)
            analysis = self._analyze(task)
            task.subject = analysis.subject
            profile_directives = self.profile.directives if self.profile else []
            task.effective_constraints = list(
                dict.fromkeys([*task.request.constraints, *profile_directives, *analysis.constraints])
            )
            task.evidence.append({"kind": "analysis", "task_type": analysis.task_type, "complexity": analysis.complexity})
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

        if task.request.mode in {"inspect", "research"}:
            task.status = TaskStatus.COMPLETED
            self._phase(task, Phase.JUDGE_LOG, "requested evidence bundle produced")
            task.evidence.append({"kind": "result", "verified": True, "scope": task.request.mode})
            self._phase(task, Phase.LEARN, "no durable lesson promoted for read-only work")
            self.task_store.save(task)
            self._emit(task, "result", "read-only task complete", self._result_payload(task))
            return task

        self._phase(task, Phase.SET_UP, "loading only task-relevant skills, tools, and model depth")
        selected_skills = self._select_gear(task.request.goal)
        task.evidence.append(
            {
                "kind": "gear",
                "skills": [item["name"] for item in selected_skills],
                "tools": sorted((task.research.tools if task.research else {}).keys()),
                "variant": task.request.variant,
                "skill_groups": self.profile.skill_groups if self.profile else [],
            }
        )
        self.task_store.save(task)

        previous_plan_hash = self._plan_hash(task.plan) if task.plan else ""
        completed_ids = {result.action_id for result in task.results if result.status == ActionStatus.OK}
        budget = task.request.time_budget_seconds or task.request.timeout
        deadline = time.monotonic() + budget
        max_attempts = max(3, min(12, budget // 240 + 1))
        reuse_existing_plan = resume and task.plan is not None
        while task.attempt < max_attempts:
            if time.monotonic() >= deadline:
                task.failure = f"work budget exhausted after approximately {budget} seconds"
                task.status = TaskStatus.UNVERIFIED
                break
            task.attempt += 1
            self._phase(task, Phase.WORK, f"attempt {task.attempt}: structured plan and bounded actions")
            if not reuse_existing_plan:
                task.plan = self._plan(task, selected_skills)
            reuse_existing_plan = False
            self.task_store.save(task)
            if task.plan:
                self._emit(
                    task,
                    "plan",
                    task.plan.summary,
                    {
                        "actions": len(task.plan.actions),
                        "checks": len(task.plan.acceptance_checks),
                        "options": [option.id for option in task.plan.options],
                    },
                )
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
                self._phase(task, Phase.LEARN, "no execution lesson promoted")
                self.task_store.save(task)
                self._emit(task, "result", "plan complete", self._result_payload(task))
                return task
            if task.request.autonomy == "proposal":
                inspect_results = self._execute_actions(task, completed_ids, only_inspect=True)
                task.results.extend(inspect_results)
                task.status = TaskStatus.UNVERIFIED
                task.failure = "proposal-only handoff; no mutations were applied"
                self._phase(task, Phase.JUDGE_LOG, task.failure)
                self.task_store.save(task)
                self._emit(task, "result", "proposal ready", self._result_payload(task))
                return task

            lint_failure = self._lint_plan(task.plan)
            if lint_failure:
                results = []
                blocking_failure = lint_failure
            else:
                results = self._execute_actions(task, completed_ids)
                task.results.extend(results)
                completed_ids.update(result.action_id for result in results if result.status == ActionStatus.OK)
                blocking_failure = self._blocking_failure(task, results)
            self.task_store.save(task)

            self._phase(task, Phase.TEST, "running explicit acceptance checks")
            checks = task.request.acceptance_checks or (task.plan.acceptance_checks if task.plan else [])
            task.check_results = self._run_checks(task, checks, deadline=deadline) if not blocking_failure else []
            self.task_store.save(task)

            self._phase(task, Phase.JUDGE_LOG, "independent deterministic judgment")
            success, failure = self._judge(task, checks, blocking_failure)
            if success:
                task.status = TaskStatus.COMPLETED
                task.failure = ""
                task.evidence.append({"kind": "judgment", "verified": True, "checks": len(checks)})
                self._phase(task, Phase.LEARN, "recording at most one evidence-linked lesson")
                task.lesson = self.memory.learn_from(task)
                self.task_store.save(task)
                self._emit(task, "result", "goal verified", self._result_payload(task))
                return task

            task.failure = failure
            task.status = TaskStatus.FAILED if checks else TaskStatus.UNVERIFIED
            task.evidence.append({"kind": "judgment", "verified": False, "reason": failure})
            self.task_store.save(task)
            if task.attempt >= max_attempts:
                break
            self._phase(task, Phase.REPEAT, f"new evidence requires a changed approach: {failure}")
            new_plan = self._plan(task, selected_skills, failure=failure)
            new_hash = self._plan_hash(new_plan)
            current_hash = self._plan_hash(task.plan)
            if new_hash in {current_hash, previous_plan_hash} - {""}:
                task.failure = "replanning repeated the same approach; stopped"
                break
            previous_plan_hash = current_hash
            task.plan = new_plan
            task.check_results = []
            reuse_existing_plan = True
            resume = True
            self.task_store.save(task)

        self._phase(task, Phase.LEARN, "failed or unverified runs do not create durable lessons")
        self.task_store.save(task)
        self._emit(task, "error", task.failure or "task did not verify", self._result_payload(task))
        return task

    def _analyze(self, task: TaskRecord) -> Analysis:
        fallback = Analysis(
            subject=" ".join(task.request.goal.split()[:6]),
            constraints=[],
            complexity=self._complexity(task.request.goal),
        )
        if not self.backend.available():
            return fallback
        prompt = (
            f"Goal: {task.request.goal}\n"
            f"Explicit constraints: {json.dumps(task.request.constraints)}\n"
            "Return only the requested schema. Preserve explicit constraints; do not invent preferences."
        )
        try:
            raw = self.backend.generate(prompt, role="classifier", schema=Analysis, think=False, timeout=120)
            return Analysis.model_validate_json(raw)
        except Exception as exc:
            task.evidence.append({"kind": "warning", "phase": "analyze", "message": str(exc)[:500]})
            return fallback

    def _plan(self, task: TaskRecord, selected_skills: list[dict[str, str]], failure: str = "") -> ModelPlan:
        if not self.backend.available():
            raise RuntimeError("a local model is required to create a coding plan")
        research = task.research or ResearchBundle()
        skills = "\n\n".join(f"SKILL {item['name']}:\n{item['content']}" for item in selected_skills)
        lessons = self.memory.relevant_lessons(task.request.goal)
        prior = self._failure_context(task) if task.results or failure else ""
        prompt = f"""
Create the smallest effective structured coding plan for this exact workspace.

GOAL: {task.request.goal}
MODE: {task.request.mode}
WORKSPACE: {self.workspace}
ALLOWED PATHS: {json.dumps(task.request.allowed_paths)}
CONSTRAINTS: {json.dumps(task.effective_constraints)}
STANDING DIRECTIVES: {json.dumps(self.memory.directives())}
RELEVANT VERIFIED LESSONS: {json.dumps(lessons)}
FAILURE TO CORRECT: {failure}

LOCAL EVIDENCE:
{research.local_context[:24_000]}

CURRENT DOCUMENTATION:
{research.documentation[:12_000]}

SELECTED SKILLS ONLY:
{skills[:18_000]}

PRIOR ACTION EVIDENCE:
{prior[:12_000]}

Rules:
- Return only the schema.
- Use inspect actions before uncertain edits.
- Commands are argv arrays with explicit cwd; never emit shell strings, cd commands, heredocs, or interactive editors.
- Existing files must change through a unified patch with a/ and b/ paths. CREATE is only for a path that does not exist.
- Every mutating action states its expected observable result and is blocking unless genuinely optional.
- Use depends_on for ordering. Assign a parallel_group only to independent read-only inspections that are safe to run concurrently.
- If the operator asks for ideas or a choice, return exactly the requested number of typed options, declare prerequisites/conflicts, set selection_required, and mark one honest recommendation with selected_by_default. If they asked for suggestions only, do not attach mutating actions yet.
- Write option titles and summaries in direct, natural language. Act like the operator's capable partner: make the tradeoffs concrete, recommend a next move, and avoid detached consultant filler.
- Include at least one narrow deterministic acceptance check for implementation work.
- Do not touch paths unrelated to the goal. Do not add dependencies or abstractions unless required.
""".strip()
        role = "planner" if task.request.mode == "plan" or task.attempt > 1 or self._complexity(task.request.goal) >= 4 else "coder"
        raw = self.backend.generate(prompt, role=role, schema=ModelPlan, think=False, timeout=task.request.timeout)
        plan = ModelPlan.model_validate_json(raw)
        if task.request.mode == "implement" and not plan.acceptance_checks and not task.request.acceptance_checks:
            plan.acceptance_checks = self._infer_checks(plan)
        return plan

    @staticmethod
    def _infer_checks(plan: ModelPlan) -> list[AcceptanceCheck]:
        checks = []
        for action in plan.actions:
            if action.kind == ActionKind.INSPECT and action.argv and action.acceptance_check:
                checks.append(AcceptanceCheck(name=action.acceptance_check, argv=action.argv, cwd=action.cwd))
        return checks

    def _execute_actions(self, task: TaskRecord, completed_ids: set[str], only_inspect: bool = False) -> list[ActionResult]:
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
        index = 0
        while index < len(actions):
            action = actions[index]
            if action.id in completed_ids:
                index += 1
                continue
            missing = [dependency for dependency in action.depends_on if dependency not in completed_ids]
            if missing:
                result = ActionResult(
                    action_id=action.id,
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
            result = executor.run(action)
            results.append(result)
            if result.status == ActionStatus.OK:
                completed_ids.add(result.action_id)
            event_type = "patch" if action.kind in {ActionKind.PATCH, ActionKind.CREATE} else "action"
            self._emit(task, event_type, result.status, {"result": result.model_dump(mode="json")})
            if action.blocking and result.status != ActionStatus.OK:
                break
            index += 1
        return results

    def _select_gear(self, goal: str) -> list[dict[str, str]]:
        """Quartermaster: trivial orders get at most one skill, and only a
        clearly on-topic one — a 7B planner drowns in off-topic library prose."""

        complexity = self._complexity(goal)
        limit = 1 if complexity <= 2 else 3
        max_chars = 3_000 if complexity <= 2 else 8_000
        selected = self.skills.load_selected(goal, limit=limit, max_chars=max_chars)
        if complexity > 2:
            return selected
        goal_tokens = {token for token in re.findall(r"[a-z0-9]+", goal.lower()) if len(token) > 2}
        return [
            item
            for item in selected
            if goal_tokens & set(re.findall(r"[a-z0-9]+", str(item.get("name", "")).lower()))
        ]

    @staticmethod
    def _lint_plan(plan: ModelPlan | None) -> str:
        """Reject structurally doomed plans before execution so the replan
        prompt receives a precise correction instead of executor noise."""

        if plan is None:
            return "no plan was produced"
        problems: list[str] = []
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
        missing_pipeline = [
            action.id
            for action in plan.actions
            if action.kind == ActionKind.PIPELINE
            and (not action.pipeline or any(not stage for stage in action.pipeline))
        ]
        if missing_pipeline:
            problems.append("pipeline actions without commands: " + ", ".join(missing_pipeline[:6]))
        return "invalid plan: " + "; ".join(problems) if problems else ""

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
            return bool(self.approve and self.approve(action, reason))

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
                argv=check.argv,
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
        missing = [check.name for check, action_id in zip(required, expected_ids) if action_id not in by_id]
        if missing:
            return False, "required checks did not run: " + ", ".join(missing)
        failed = [
            check.name
            for check, action_id in zip(required, expected_ids)
            if by_id[action_id].status != ActionStatus.OK
        ]
        if failed:
            return False, "required checks failed: " + ", ".join(failed)
        return True, ""

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
        if any(word in lowered for word in ("architecture", "migrate", "agent", "wasm", "security", "performance")):
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
        payload = plan.model_dump(mode="json")
        for action in payload.get("actions", []):
            action.pop("id", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _phase(self, task: TaskRecord, phase: Phase, message: str) -> None:
        task.phase = phase
        self.task_store.save(task)
        self._emit(task, "phase", message, {"phase": phase})

    def _emit(self, task: TaskRecord, event_type: str, message: Any, data: dict[str, Any]) -> None:
        self._sequence += 1
        event = XanderEvent(
            run_id=task.id,
            sequence=self._sequence,
            type=event_type,
            phase=task.phase,
            attempt=task.attempt,
            message=str(message),
            data=data,
        )
        if self.event_sink:
            try:
                self.event_sink(event)
            except Exception as exc:
                task.evidence.append(
                    {"kind": "warning", "phase": task.phase, "message": f"event sink failed: {exc}"[:500]}
                )

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
