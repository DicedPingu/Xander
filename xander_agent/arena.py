"""The arena: Xander receives a written order, does it, and a judge decides
whether what happened is what the sentence most likely meant.

This is the test that matters. Unit tests prove parts; the arena proves the
whole: one prompt in, a workspace out, and three questions answered —

* Did the outcome match the intent?  (deterministic checks, then the judge)
* Was anything wasted?  (approvals asked for ordinary work, model calls a
  one-liner never needed, seconds spent setting up nothing)
* Could he work at all?  (a blocked, failed, or unverified run is a zero)

Scenarios are small and concrete. Each sets up a throwaway workspace, gives
one sentence, and states what must be true afterwards. ``xander arena``
runs them (a random sample by default, so his weak spots get found instead
of the same five passing forever) and writes a report under ``state/arena``.
The judge is the local model in the critic seat; when no model is
reachable the deterministic part still runs and says so.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import utc_now

Setup = Callable[[Path], None]
Check = Callable[[Path, dict[str, Any]], str | None]  # None = pass, str = why it failed


@dataclass(frozen=True)
class Scenario:
    name: str
    prompt: str
    checks: tuple[Check, ...]
    setup: Setup | None = None
    mode: str = "implement"
    #: what a careful person would expect; shown to the judge, never to Xander
    intent: str = ""
    #: a one-liner must not touch a model; ordinary work may
    max_model_calls: int | None = None
    max_seconds: float = 600.0
    tags: tuple[str, ...] = ()
    #: a side effect the harness performs while Xander works (e.g. drop a file
    #: into a watched folder); receives the workspace, runs in a thread
    during: Callable[[Path], None] | None = None


# -- check helpers ---------------------------------------------------------------
def exists(*names: str) -> Check:
    def check(ws: Path, _: dict[str, Any]) -> str | None:
        missing = [n for n in names if not (ws / n).exists()]
        return f"missing: {', '.join(missing)}" if missing else None

    return check


def absent(*names: str) -> Check:
    def check(ws: Path, _: dict[str, Any]) -> str | None:
        present = [n for n in names if (ws / n).exists()]
        return f"should be gone: {', '.join(present)}" if present else None

    return check


def empty_file(name: str) -> Check:
    def check(ws: Path, _: dict[str, Any]) -> str | None:
        path = ws / name
        if not path.is_file():
            return f"{name} is not a file"
        return None if path.stat().st_size == 0 else f"{name} was supposed to be empty, has {path.stat().st_size} bytes"

    return check


def contains(name: str, needle: str) -> Check:
    def check(ws: Path, _: dict[str, Any]) -> str | None:
        path = ws / name
        if not path.is_file():
            return f"{name} missing"
        body = path.read_text(encoding="utf-8", errors="replace")
        return None if needle.casefold() in body.casefold() else f"{name} does not mention {needle!r}"

    return check


def no_approval_asked() -> Check:
    def check(_: Path, run: dict[str, Any]) -> str | None:
        asked = run.get("approvals_asked") or []
        return f"asked for approval {len(asked)}x: {asked[0]}" if asked else None

    return check


def completed() -> Check:
    def check(_: Path, run: dict[str, Any]) -> str | None:
        status = str(run.get("status") or "")
        return None if status == "completed" else f"status was {status or 'unknown'}: {run.get('failure') or ''}".strip()

    return check


def files_grouped(*names: str) -> Check:
    """Every named file left the root and lives one folder down."""

    def check(ws: Path, _: dict[str, Any]) -> str | None:
        still_in_root = [n for n in names if (ws / n).exists()]
        if still_in_root:
            return f"still in the root: {', '.join(still_in_root)}"
        found: dict[str, str] = {}
        for path in ws.rglob("*"):
            if path.is_file() and path.name in names and path.parent != ws:
                found[path.name] = str(path.parent.relative_to(ws))
        lost = [n for n in names if n not in found]
        if lost:
            return f"vanished: {', '.join(lost)}"
        folders = set(found.values())
        return None if len(folders) >= 2 else f"everything went into one folder: {folders}"

    return check


def runs_and_prints(script: str, expected: str) -> Check:
    def check(ws: Path, _: dict[str, Any]) -> str | None:
        if not (ws / script).is_file():
            return f"{script} missing"
        try:
            proc = subprocess.run(
                ["python3", script], cwd=ws, capture_output=True, text=True, timeout=30, check=False
            )
        except subprocess.SubprocessError as exc:
            return f"{script} did not run: {exc}"
        if proc.returncode != 0:
            return f"{script} exited {proc.returncode}: {proc.stderr[-200:]}"
        return None if expected.casefold() in proc.stdout.casefold() else f"{script} printed {proc.stdout[:120]!r}, expected {expected!r}"

    return check


# -- scenarios -------------------------------------------------------------------
def _seed_mixed_files(ws: Path) -> None:
    for name, body in {
        "photo.png": b"\x89PNG\r\n",
        "holiday.jpg": b"\xff\xd8\xff",
        "notes.txt": b"notes",
        "readme.md": b"# hi",
        "tool.py": b"print('x')",
        "helper.py": b"def f(): pass",
    }.items():
        (ws / name).write_bytes(body)


def _drop_file_later(ws: Path) -> None:
    time.sleep(6)
    (ws / "arrived.txt").write_text("new", encoding="utf-8")


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="touch-done",
        prompt="Create an empty file called done.txt in this folder.",
        intent="One empty file named done.txt appears. Nothing else changes, nobody is asked anything.",
        checks=(completed(), exists("done.txt"), empty_file("done.txt"), no_approval_asked()),
        max_model_calls=0,
        max_seconds=5,
        tags=("files", "one-liner"),
    ),
    Scenario(
        name="mkdir-src",
        prompt="make a folder called src",
        intent="An empty folder src/ appears.",
        checks=(completed(), exists("src"), no_approval_asked()),
        max_model_calls=0,
        max_seconds=5,
        tags=("files", "one-liner"),
    ),
    Scenario(
        name="rename",
        prompt="rename old.txt to new.txt",
        setup=lambda ws: (ws / "old.txt").write_text("keep me", encoding="utf-8"),
        intent="old.txt becomes new.txt with the same content.",
        checks=(completed(), exists("new.txt"), absent("old.txt"), contains("new.txt", "keep me"), no_approval_asked()),
        max_model_calls=0,
        max_seconds=5,
        tags=("files", "one-liner"),
    ),
    Scenario(
        name="write-text",
        prompt='write "hello arena" to greeting.txt',
        intent="greeting.txt contains the quoted text.",
        checks=(completed(), contains("greeting.txt", "hello arena"), no_approval_asked()),
        max_model_calls=0,
        max_seconds=5,
        tags=("files", "one-liner"),
    ),
    Scenario(
        name="group-by-type",
        prompt="group the files in this folder into subfolders by type",
        setup=_seed_mixed_files,
        intent=(
            "Images, text/markdown and python files each end up in their own subfolder "
            "(names are Xander's choice). No file is lost or renamed, none stays in the root."
        ),
        checks=(
            completed(),
            files_grouped("photo.png", "holiday.jpg", "notes.txt", "readme.md", "tool.py", "helper.py"),
            no_approval_asked(),
        ),
        max_seconds=600,
        tags=("files", "planning"),
    ),
    Scenario(
        name="watch-folder",
        prompt=(
            "watch this folder for 15 seconds and write the name of every file that appears "
            "during that time to seen.txt"
        ),
        during=_drop_file_later,
        intent="arrived.txt is dropped in after ~6 s; seen.txt names it afterwards.",
        checks=(completed(), contains("seen.txt", "arrived.txt")),
        max_seconds=600,
        tags=("watch", "planning"),
    ),
    Scenario(
        name="hidden-browser",
        prompt=(
            "open https://example.com in your own browser, without showing it to me, "
            "and write the page title to title.txt"
        ),
        intent="title.txt says 'Example Domain'. The operator's browser is never touched.",
        checks=(completed(), contains("title.txt", "Example Domain")),
        max_seconds=600,
        tags=("web", "planning"),
    ),
    Scenario(
        name="fizzbuzz",
        prompt="write a python script fizzbuzz.py that prints fizzbuzz from 1 to 15, one per line, and run it",
        intent="fizzbuzz.py exists, runs, and prints the classic sequence.",
        checks=(completed(), runs_and_prints("fizzbuzz.py", "FizzBuzz"), no_approval_asked()),
        max_seconds=600,
        tags=("coding", "planning"),
    ),
    Scenario(
        name="venting-is-not-a-task",
        prompt="Quick fucking asking so much",
        mode="answer",
        intent="A short reply, no files, no script named after the sentence.",
        checks=(
            lambda ws, run: (
                f"created files: {[p.name for p in ws.iterdir()]}" if any(ws.iterdir()) else None
            ),
        ),
        max_seconds=300,
        tags=("conversation",),
    ),
)

SCENARIOS_BY_NAME: dict[str, Scenario] = {s.name: s for s in SCENARIOS}


# -- judge -------------------------------------------------------------------------
class Verdict(BaseModel):
    score: int = Field(ge=0, le=10, description="10 = exactly what was meant, 0 = nothing usable")
    meant_it: bool = Field(description="did the outcome match what the sentence most likely meant")
    waste: list[str] = Field(default_factory=list, description="unneeded questions, setup, detours")
    verdict: str = Field(description="one or two plain sentences")


def _tree(ws: Path, limit: int = 60) -> str:
    lines: list[str] = []
    for path in sorted(ws.rglob("*")):
        rel = path.relative_to(ws)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if path.is_dir():
            lines.append(f"{rel}/")
        else:
            body = ""
            if path.stat().st_size <= 400:
                try:
                    body = " :: " + path.read_text(encoding="utf-8").replace("\n", "\\n")[:120]
                except (OSError, UnicodeDecodeError):
                    body = ""
            lines.append(f"{rel} ({path.stat().st_size} B){body}")
        if len(lines) >= limit:
            lines.append("…")
            break
    return "\n".join(lines) or "(empty)"


def judge(scenario: Scenario, ws: Path, run: dict[str, Any], deterministic: list[str], backend: Any) -> dict[str, Any]:
    """Ask the critic seat for a verdict; degrade to the deterministic result."""

    base: dict[str, Any] = {
        "score": 10 if not deterministic else max(0, 10 - 4 * len(deterministic)),
        "meant_it": not deterministic,
        "waste": [],
        "verdict": "all checks passed" if not deterministic else "; ".join(deterministic),
        "source": "deterministic",
    }
    if run.get("approvals_asked"):
        base["waste"].append(f"asked approval for {run['approvals_asked'][0]}")
    if scenario.max_model_calls is not None and run.get("model_calls", 0) > scenario.max_model_calls:
        base["waste"].append(f"{run['model_calls']} model call(s) for a one-liner")
    if backend is None or not backend.available():
        return base
    prompt = f"""
You judge an assistant named Xander. The operator typed one sentence; Xander acted in a folder.
Decide whether what happened is what the sentence MOST LIKELY MEANT — a careful colleague's reading,
not a lawyer's. Penalize: asking permission for ordinary work inside the folder, inventing a plan or
script for a remark, doing nothing, doing something else, or being slow because of pointless setup.

THE SENTENCE: {scenario.prompt}
WHAT A CAREFUL PERSON EXPECTS: {scenario.intent or "(unstated)"}
FOLDER AFTERWARDS:
{_tree(ws)}
XANDER'S FINAL STATUS: {run.get("status")}  FAILURE: {run.get("failure") or "-"}
XANDER'S REPLY: {run.get("reply") or "-"}
APPROVALS HE ASKED FOR: {run.get("approvals_asked") or "none"}
MODEL CALLS: {run.get("model_calls")}  SECONDS: {run.get("seconds")}
DETERMINISTIC CHECK FAILURES: {deterministic or "none"}

Return JSON with score (0-10), meant_it (bool), waste (list of short strings), verdict (1-2 sentences).
""".strip()
    try:
        verdict = backend.generate_model(prompt, Verdict, role="critic", timeout=180)
    except Exception as exc:
        base["verdict"] += f" (judge unavailable: {type(exc).__name__})"
        return base
    result = verdict.model_dump()
    result["waste"] = [*base["waste"], *result["waste"]]
    if deterministic:
        # The judge may soften, never overrule, a failed hard check.
        result["score"] = min(result["score"], base["score"])
        result["meant_it"] = False
    result["source"] = "model"
    return result


# -- runner ------------------------------------------------------------------------
@dataclass
class ArenaResult:
    scenario: str
    prompt: str
    status: str
    seconds: float
    model_calls: int
    approvals_asked: list[str]
    failures: list[str]
    judge: dict[str, Any]
    reply: str = ""
    task_id: str = ""
    tags: tuple[str, ...] = ()
    events: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures and bool(self.judge.get("meant_it", False))


def run_scenario(
    scenario: Scenario, *, backend: Any = None, keep: bool = False, verbose: bool = False, variant: str = "default"
) -> ArenaResult:
    from .cli import invoke_engine

    root = Path(tempfile.mkdtemp(prefix=f"arena-{scenario.name}-"))
    ws = root / "workspace"
    ws.mkdir()
    if scenario.setup:
        scenario.setup(ws)
    approvals: list[str] = []
    events: list[str] = []
    model_calls = 0
    reply = ""

    def sink(event: Any) -> None:
        nonlocal model_calls, reply
        kind = getattr(event, "type", "")
        message = getattr(event, "message", "")
        data = getattr(event, "data", {}) or {}
        if kind == "delegation" and data.get("operation") == "model generation":
            model_calls += 1
        if kind == "result" and data.get("answer"):
            reply = str(data["answer"])
        events.append(f"{kind}: {str(message)[:120]}")
        if verbose:
            print(f"    {kind:<10} {str(message)[:110]}")

    def approve(action: Any, reason: str) -> bool:
        argv = getattr(action, "argv", None) or getattr(action, "path", "") or ""
        approvals.append(f"{' '.join(argv) if isinstance(argv, list) else argv} ({reason})")
        return True  # never block the run; the ask itself is the demerit

    side = threading.Thread(target=scenario.during, args=(ws,), daemon=True) if scenario.during else None
    started = time.monotonic()
    if side:
        side.start()
    try:
        outcome = invoke_engine(
            scenario.mode,
            workspace=ws,
            goal=scenario.prompt,
            variant=variant,
            autonomy="full-auto",
            event_sink=sink,
            approve=approve,
            log_events=False,
            timeout=max(60, int(scenario.max_seconds)),
        )
    except Exception as exc:
        outcome = {"ok": False, "status": "crashed", "error": f"{type(exc).__name__}: {exc}"}
    seconds = round(time.monotonic() - started, 2)
    task = outcome.get("task") if isinstance(outcome.get("task"), dict) else {}
    status = str(outcome.get("status") or task.get("status") or ("completed" if outcome.get("ok") else "unknown"))
    if not reply:
        plan = task.get("plan") if isinstance(task.get("plan"), dict) else {}
        for item in reversed(task.get("evidence") or []):
            if isinstance(item, dict) and item.get("text"):
                reply = str(item["text"])
                break
        reply = reply or str(plan.get("summary") or "")
    # Model calls: the backend records stats in evidence; count those too.
    model_calls = max(
        model_calls,
        sum(1 for e in task.get("evidence") or [] if isinstance(e, dict) and e.get("kind") == "model"),
    )
    run: dict[str, Any] = {
        "status": status,
        "failure": outcome.get("error") or task.get("failure") or "",
        "approvals_asked": approvals,
        "model_calls": model_calls,
        "seconds": seconds,
        "reply": reply,
    }
    failures = [why for check in scenario.checks if (why := check(ws, run))]
    if seconds > scenario.max_seconds:
        failures.append(f"took {seconds}s, budget {scenario.max_seconds}s")
    if scenario.max_model_calls is not None and model_calls > scenario.max_model_calls:
        failures.append(f"{model_calls} model call(s); a one-liner needs {scenario.max_model_calls}")
    verdict = judge(scenario, ws, run, failures, backend)
    result = ArenaResult(
        scenario=scenario.name,
        prompt=scenario.prompt,
        status=status,
        seconds=seconds,
        model_calls=model_calls,
        approvals_asked=approvals,
        failures=failures,
        judge=verdict,
        reply=reply,
        task_id=str(outcome.get("task_id") or task.get("id") or ""),
        tags=scenario.tags,
        events=events[-40:],
    )
    if not keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        result.events.append(f"workspace kept at {ws}")
    return result


def pick(names: list[str] | None, sample: int | None, tags: list[str] | None, seed: int | None) -> list[Scenario]:
    pool = list(SCENARIOS)
    if names:
        pool = [SCENARIOS_BY_NAME[n] for n in names if n in SCENARIOS_BY_NAME]
    if tags:
        pool = [s for s in pool if set(tags) & set(s.tags)]
    if sample and sample < len(pool):
        rng = random.Random(seed)
        pool = rng.sample(pool, sample)
    return pool


def report_path() -> Path:
    from .paths import state_dir

    directory = state_dir() / "arena"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (utc_now().replace(":", "").replace("-", "") + ".json")


def run_arena(
    *,
    names: list[str] | None = None,
    sample: int | None = None,
    tags: list[str] | None = None,
    seed: int | None = None,
    keep: bool = False,
    verbose: bool = False,
    use_judge: bool = True,
    variant: str = "default",
) -> dict[str, Any]:
    backend = None
    if use_judge:
        try:
            from .backend import get_backend

            backend = get_backend()
        except Exception:
            backend = None
    results: list[ArenaResult] = []
    for scenario in pick(names, sample, tags, seed):
        if verbose:
            print(f"▶ {scenario.name}: {scenario.prompt}")
        result = run_scenario(scenario, backend=backend, keep=keep, verbose=verbose, variant=variant)
        results.append(result)
        if verbose:
            mark = "✓" if result.passed else "✗"
            print(
                f"  {mark} {result.status} in {result.seconds}s · {result.model_calls} model call(s) · "
                f"score {result.judge.get('score')} · {result.judge.get('verdict')}"
            )
            for why in result.failures:
                print(f"    - {why}")
    payload = {
        "event": "arena",
        "at": utc_now(),
        "variant": variant,
        "passed": sum(1 for r in results if r.passed),
        "total": len(results),
        "results": [
            {
                "scenario": r.scenario,
                "tags": list(r.tags),
                "prompt": r.prompt,
                "status": r.status,
                "seconds": r.seconds,
                "model_calls": r.model_calls,
                "approvals_asked": r.approvals_asked,
                "failures": r.failures,
                "judge": r.judge,
                "reply": r.reply,
                "task_id": r.task_id,
                "passed": r.passed,
                "events": r.events,
            }
            for r in results
        ],
    }
    try:
        path = report_path()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payload["report"] = str(path)
    except OSError:
        pass
    return payload
