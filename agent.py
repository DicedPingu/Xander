"""The Xander pipeline: intake → research → provision → plan → execute → learn.

Given a goal, Xander:
  1. intake   — restates the goal and extracts the operator's exact constraints
  2. research — pulls the latest from GitHub (and optionally the web)
  3. provision— wires a task-specific toolbelt (adb/fastboot/git/…) and runs
                orientation hooks so the plan is grounded in the real machine
  4. plan     — writes a concrete, ordered step list (each step a shell command
                or a check), obeying operator directives to the letter
  5. execute  — runs steps, pausing for confirmation on anything destructive
  6. learn    — records what worked / failed against the task's tags

Every stage writes its state to state/tasks/<slug>.json so a run is fully
inspectable and resumable, and so Xander literally "configs himself for the
task" on disk before working.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import DANGEROUS_PATTERNS, GEN_TIMEOUT, TASK_DIR, UNCENSORED
from memory import Memory
from research import research
from toolbelt import Toolbelt, provision, tags_for

_DANGER_RE = [re.compile(p) for p in DANGEROUS_PATTERNS]


def is_dangerous(cmd: str) -> bool:
    return any(r.search(cmd) for r in _DANGER_RE)


def slugify(text: str) -> str:
    return re.sub(r"[^\w.-]+", "-", text.lower()).strip("-")[:48] or "task"


@dataclass
class Step:
    kind: str            # "run" | "check" | "note"
    text: str            # command or note
    why: str = ""
    status: str = "pending"   # pending | ok | fail | skipped
    output: str = ""


@dataclass
class Task:
    goal: str
    subject: str = ""
    constraints: list[str] = field(default_factory=list)
    directives: list[str] = field(default_factory=list)
    toolbelt: dict = field(default_factory=dict)
    research: str = ""
    sources: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def path(self) -> Path:
        return TASK_DIR / f"{slugify(self.goal)}.json"

    def save(self) -> Path:
        p = self.path()
        payload = asdict(self)
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return p


_current_cwd = None

def run_shell(cmd: str, timeout: int = 120) -> tuple[int, str]:
    global _current_cwd
    import os
    if _current_cwd is None:
        _current_cwd = os.getcwd()
    try:
        wrapped_cmd = f"cd {_current_cwd} && {{ {cmd.rstrip(';')}; }} ; RC=$? ; echo '___CWD___' ; pwd ; exit $RC"
        r = subprocess.run(wrapped_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "")
        err = (r.stderr or "")
        
        new_cwd = _current_cwd
        if "___CWD___" in out:
            parts = out.split("___CWD___")
            cmd_out = parts[0]
            pwd_part = parts[1].strip()
            lines = pwd_part.splitlines()
            if lines:
                dir_line = lines[0].strip()
                if os.path.isdir(dir_line):
                    new_cwd = dir_line
        else:
            cmd_out = out
            
        _current_cwd = new_cwd
        combined_out = (cmd_out + err).strip()[:4000]
        return r.returncode, combined_out
    except subprocess.TimeoutExpired:
        return 124, f"[timeout after {timeout}s]"
    except Exception as exc:
        return 1, f"[error] {exc}"


class Xander:
    """Orchestrates one goal end to end. UI is injected so the core stays clean
    and testable; pass any object implementing the hooks agent.py calls."""

    def __init__(self, backend, log, ui, memory: Memory | None = None):
        self.backend = backend
        self.log = log
        self.ui = ui
        self.memory = memory or Memory()

    # ── stage 1: analyze ─────────────────────────────────────────────────────
    def analyze(self, goal: str) -> Task:
        task = Task(goal=goal, directives=self.memory.directives())
        prompt = (
            f'Operator goal: "{goal}".\n'
            "Return STRICT JSON: {\"subject\": \"2-4 word search subject\", "
            "\"constraints\": [\"explicit requirement the operator stated or clearly implied\"]}. "
            "Constraints are hard requirements only — do not invent preferences."
        )
        raw = self.backend.ask(prompt, temperature=0.2, num_predict=200, timeout=90)
        data = _loose_json(raw) or {}
        task.subject = (data.get("subject") or goal).strip()[:60]
        task.constraints = [c.strip() for c in (data.get("constraints") or []) if c.strip()][:8]
        return task

    # ── stage 2: research ────────────────────────────────────────────────────
    def research(self, task: Task) -> None:
        ctx, sources = research(task.goal, task.subject, log=self.log)
        task.research = ctx
        task.sources = sources[:12]

    # ── stage 3: gear_up ─────────────────────────────────────────────────────
    def gear_up(self, task: Task) -> Toolbelt:
        belt = provision(task.goal)
        # run orientation hooks so the plan sees the real device/machine state
        oriented = []
        for hook in belt.hooks[:6]:
            code, out = run_shell(hook, timeout=20)
            oriented.append(f"$ {hook}\n{out}")
            self.log.info("hook", cmd=hook, rc=code)
        task.toolbelt = {
            "capabilities": [c.name for c in belt.capabilities],
            "tools": belt.tools,
            "missing": belt.missing,
            "orientation": "\n\n".join(oriented),
        }
        return belt

    def work(self, task: Task, belt: Toolbelt, on_token=None) -> None:
        tags = tags_for(task.goal)
        directive_block = self.memory.directive_block()
        recall = self.memory.recall_block(tags)
        prompt = _plan_prompt(task, belt, directive_block, recall)
        raw = self.backend.ask(prompt, temperature=0.3, num_predict=8192,
                               on_token=on_token, timeout=GEN_TIMEOUT)
        steps = _parse_steps(raw)
        task.steps = steps

    # ── stage 5: execute ─────────────────────────────────────────────────────
    def execute_steps(self, task: Task, confirm, yolo: bool = False) -> None:
        """`confirm(step)` → bool, asked before any dangerous step (unless yolo)."""
        global _current_cwd
        import os
        _current_cwd = os.getcwd()
        for i, step in enumerate(task.steps):
            if step.kind == "note":
                self.ui.note(step.text)
                step.status = "ok"
                continue
            if step.kind == "write":
                self.ui.show_step(i + 1, len(task.steps), step, danger=False)
                try:
                    dest_path = Path(step.text)
                    if not dest_path.is_absolute():
                        dest_path = Path(_current_cwd) / dest_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Clean markdown code fences if present
                    content = step.output.strip()
                    if content.startswith("```"):
                        first_nl = content.find("\n")
                        if first_nl != -1:
                            content = content[first_nl + 1:]
                    if content.endswith("```"):
                        content = content[:-3].rstrip()
                        
                    dest_path.write_text(content, encoding="utf-8")
                    step.status = "ok"
                    step.output = ""
                    self.ui.step_result(step, code=0)
                    self.log.info("write", file=str(dest_path))
                except Exception as exc:
                    step.status = "fail"
                    step.output = str(exc)
                    self.ui.step_result(step, code=1)
                    self.log.info("write_fail", file=step.text, err=str(exc))
                task.save()
                continue
            danger = is_dangerous(step.text)
            self.ui.show_step(i + 1, len(task.steps), step, danger)
            if step.kind == "run" and danger and not yolo:
                if not confirm(step):
                    step.status = "skipped"
                    self.ui.skipped(step)
                    continue
            if step.kind in ("run", "check"):
                code, out = run_shell(step.text)
                step.output = out
                step.status = "ok" if code == 0 else "fail"
                self.ui.step_result(step, code)
                self.log.info("step", n=i + 1, cmd=step.text, rc=code)
                if step.kind == "check" and code != 0:
                    self.ui.note("a required check failed — stopping so you can decide")
                    break
            task.save()

    # ── stage 6: test_and_judge ──────────────────────────────────────────────
    def test_and_judge(self, task: Task) -> tuple[bool, str]:
        """Evaluate execution results. Returns (is_done, remedy_instruction)."""
        failed_steps = [s for s in task.steps if s.status == "fail"]
        if not failed_steps:
            return True, "All steps completed successfully."
        
        history = []
        for s in task.steps:
            status_str = f"[{s.status.upper()}]"
            history.append(f"{status_str} {s.kind.upper()}: {s.text}")
            if s.status == "fail" and s.output:
                history.append(f"Output:\n{s.output[:400]}")
        history_str = "\n".join(history)

        prompt = (
            f"The operator goal was: \"{task.goal}\".\n"
            f"Execution history for this iteration:\n{history_str}\n"
            "Analyze the failure. Return a response in this exact JSON format: "
            "{\"explanation\": \"short explanation of why it failed\", "
            "\"remedy\": \"a single constraint or instruction to fix it in the next iteration\"}."
        )
        raw = self.backend.ask(prompt, temperature=0.2, num_predict=300, timeout=90)
        data = _loose_json(raw) or {}
        explanation = data.get("explanation") or "Some steps failed."
        remedy = data.get("remedy") or "Retry the failed steps."
        
        self.ui.note(f"Judge output: {explanation}")
        return False, remedy

    # ── stage 6: learn ───────────────────────────────────────────────────────
    def learn(self, task: Task) -> None:
        tags = tags_for(task.goal)
        ok = [s for s in task.steps if s.status == "ok" and s.kind == "run"]
        fail = [s for s in task.steps if s.status == "fail"]
        if ok:
            self.memory.add_learning(
                tags, f'{task.subject}: worked — {ok[0].text}', worked=True)
        for s in fail[:2]:
            self.memory.add_learning(
                tags, f'{task.subject}: failed — {s.text} ({s.output[:80]})', worked=False)

        workspace_memory = Path(__file__).resolve().parent.parent / "memory"
        if workspace_memory.exists():
            today = time.strftime("%Y-%m-%d")
            daily_file = workspace_memory / f"{today}.md"
            try:
                header = f"\n\n## Learning from Xander goal: \"{task.goal}\"\n"
                entries = []
                if ok:
                    entries.append(f"- **Worked**: `{ok[0].text}`")
                if fail:
                    for s in fail[:2]:
                        entries.append(f"- **Failed**: `{s.text}` ({s.output[:80].strip()})")
                if entries:
                    if daily_file.exists():
                        content = daily_file.read_text(encoding="utf-8", errors="ignore")
                    else:
                        content = f"# Daily Note - {today}\n"
                    daily_file.write_text(content + header + "\n".join(entries), encoding="utf-8")
            except Exception:
                pass


# ── prompt + parsing helpers ─────────────────────────────────────────────────
def _plan_prompt(task: Task, belt: Toolbelt, directive_block: str, recall: str) -> str:
    tools = ", ".join(belt.tools) or "standard shell tools"
    missing = f"\nNOT available (do not use): {', '.join(belt.missing)}" if belt.missing else ""
    parts = [UNCENSORED]
    if directive_block:
        parts.append(directive_block)
    parts.append(
        f'GOAL: {task.goal}\nSUBJECT: {task.subject}'
    )
    if task.constraints:
        parts.append("OPERATOR CONSTRAINTS (hard):\n" + "\n".join(f"- {c}" for c in task.constraints))
    if recall:
        parts.append(recall)
    if task.research:
        parts.append("LATEST RESEARCH (GitHub/web):\n" + task.research[:4000])
    if task.toolbelt.get("orientation"):
        parts.append("MACHINE STATE (orientation probes just run):\n" + task.toolbelt["orientation"][:2000])
    parts.append(
        f"AVAILABLE TOOLS: {tools}{missing}\n\n"
        "Write a concrete, ordered plan to accomplish the goal on THIS machine. "
        "Output ONLY a numbered list. Each step is exactly one of:\n"
        "  RUN: <single shell command>        (an action to execute)\n"
        "  CHECK: <single shell command>       (a gate; if it fails, stop)\n"
        "  WRITE: <filepath>\n"
        "  <multiline file contents>\n"
        "  EOF                                 (write contents to filepath)\n"
        "  NOTE: <short instruction to the operator, for manual/physical steps>\n\n"
        "Example of a WRITE step:\n"
        "  5. WRITE: lib/main.dart\n"
        "  import 'package:flutter/material.dart';\n"
        "  void main() => runApp(MyApp());\n"
        "  EOF\n\n"
        "Use only available tools. Prefer real commands over prose. Keep it tight "
        "(5-12 steps). Do not add explanation outside the list."
    )
    return "\n\n".join(parts)


_STEP_RE = re.compile(r"^\s*(?:\d+[.)]\s*)?(RUN|CHECK|NOTE)\s*:\s*(.+)$", re.IGNORECASE)


def _parse_steps(raw: str) -> list[Step]:
    steps: list[Step] = []
    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        write_match = re.match(r"^(?:\d+[.)]\s*)?WRITE\s*:\s*(.+)$", line, re.IGNORECASE)
        if write_match:
            filepath = write_match.group(1).strip().strip("`").strip()
            content_lines = []
            i += 1
            while i < len(lines):
                cur_line = lines[i]
                cleaned_line = re.sub(r"^(?:\d+[.)]\s*)?", "", cur_line.strip()).strip().upper()
                if cleaned_line in ("EOF", "END_WRITE"):
                    break
                content_lines.append(cur_line)
                i += 1
            content = "\n".join(content_lines)
            steps.append(Step(kind="write", text=filepath, output=content))
            i += 1
            continue
            
        m = _STEP_RE.match(line)
        if m:
            kind = m.group(1).lower()
            text = m.group(2).strip().strip("`")
            if text:
                steps.append(Step(kind=kind, text=text))
        i += 1
    return steps


def _loose_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None
