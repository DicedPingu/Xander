"""A second opinion from outside Xander's own head.

Xander's Master and Lurker run on the same local weights he plans with, so they
share his blind spots. Gemini is a genuinely independent reviewer: different
model, different vendor, no access to his reasoning. That is the whole value —
it disagrees for reasons he could not have generated himself.

Read-only by construction. Gemini is asked for findings; it never edits the
workspace. Whether a finding becomes a change is Xander's decision, recorded
with a reason, and applied through the same guarded path as any other change.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from .models import StrictModel, utc_now

_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)
Severity = Literal["blocker", "important", "minor", "praise"]


class Finding(StrictModel):
    severity: Severity = "minor"
    subject: str = ""
    detail: str = ""
    suggestion: str = ""


class Review(StrictModel):
    ok: bool = True
    reviewer: str = "gemini"
    summary: str = ""
    findings: list[Finding] = Field(default_factory=list)
    error: str = ""
    at: str = Field(default_factory=utc_now)

    def blockers(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "blocker"]


def available() -> bool:
    return shutil.which("gemini") is not None


def _ask(prompt: str, cwd: Path, timeout: int) -> tuple[bool, str]:
    """One bounded, non-interactive call. Never raises."""

    if not available():
        return False, "gemini is not installed"
    environment = dict(os.environ)
    environment.setdefault("NO_COLOR", "1")
    try:
        completed = subprocess.run(
            ["gemini", "-p", prompt],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=environment,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return False, f"gemini did not answer within {timeout}s"
    except OSError as exc:
        return False, f"gemini could not run: {exc}"
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return False, (tail[-1] if tail else f"gemini exited {completed.returncode}")[:300]
    return True, completed.stdout


def _parse(raw: str) -> Review:
    """Gemini is chatty; find the JSON object inside whatever it said."""

    text = _FENCE.sub("", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except Exception:
        # A prose answer is still a review; keep it rather than losing it.
        body = " ".join(raw.split())[:600]
        return Review(ok=bool(body), summary=body, error="" if body else "gemini returned nothing")
    findings = []
    for item in payload.get("findings") or []:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "minor")).casefold()
        findings.append(
            Finding(
                severity=severity if severity in {"blocker", "important", "minor", "praise"} else "minor",
                subject=str(item.get("subject", ""))[:200],
                detail=str(item.get("detail", ""))[:800],
                suggestion=str(item.get("suggestion", ""))[:800],
            )
        )
    return Review(ok=True, summary=str(payload.get("summary", ""))[:600], findings=findings[:20])


_SHAPE = """
Answer with one JSON object and nothing else:
{"summary":"one sentence","findings":[{"severity":"blocker|important|minor|praise",
"subject":"what it is about","detail":"what is wrong and why it matters",
"suggestion":"the concrete change"}]}
Be specific and short. Say "blocker" only for something that is actually broken or unsafe.
""".strip()


def review_work(workspace: Path, goal: str, paths: list[str], timeout: int = 240) -> Review:
    """Review what Xander just built, against what he was asked for."""

    listing = ", ".join(paths[:20]) or "the files in this directory"
    prompt = f"""
Review the work in this directory as an independent second opinion.

WHAT WAS ASKED FOR: {goal}
FILES TO REVIEW: {listing}

Read those files. Judge whether they actually deliver what was asked. Call out
placeholders, stubs, unimplemented functions and anything that would not build
or run. State plainly if the work is incomplete.

{_SHAPE}
""".strip()
    ok, raw = _ask(prompt, workspace, timeout)
    if not ok:
        return Review(ok=False, error=raw)
    return _parse(raw)


def review_settings(workspace: Path, settings: dict[str, Any], timeout: int = 180) -> Review:
    """Ask for improvements to how Xander is configured."""

    prompt = f"""
These are the runtime settings of a local autonomous coding agent. Suggest
improvements. Prefer fewer, higher-value changes over a long list.

SETTINGS: {json.dumps(settings, indent=2)[:4000]}

Consider: are the safety bounds sensible, is anything contradictory or dead,
would a different value make it more useful without making it reckless?
Do not suggest removing a safety guard just to go faster.

{_SHAPE}
""".strip()
    ok, raw = _ask(prompt, workspace, timeout)
    if not ok:
        return Review(ok=False, error=raw)
    return _parse(raw)


def decide(review: Review, finding: Finding) -> tuple[bool, str]:
    """Xander's own call on one finding, with the reason recorded.

    He agrees with what is demonstrably about his work and concrete enough to
    act on. He declines vague advice and anything that would weaken a guard,
    because an outside reviewer has no stake in what happens at 3am.
    """

    text = f"{finding.subject} {finding.detail} {finding.suggestion}".casefold()
    weakens = any(
        phrase in text
        for phrase in ("disable the", "remove the check", "skip verification", "turn off the guard",
                       "bypass", "ignore the policy", "without approval")
    )
    if weakens:
        return False, "declined: it asks to weaken a guard, and the guard is the point"
    if not finding.suggestion.strip():
        return False, "declined: no concrete change to make"
    if finding.severity == "praise":
        return False, "no action needed"
    if finding.severity in {"blocker", "important"}:
        return True, f"agreed: {finding.severity} and actionable"
    return True, "agreed: small and concrete"
