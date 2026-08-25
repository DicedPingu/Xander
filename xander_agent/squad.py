"""Xander's squad: named helper clones and the forms he takes himself.

Two helpers can muster for a mission, never more:

- **Master** — the supervisor. Judges plans and, above all, the result:
  the Master has to end up happy, which matters most exactly when the
  operator gave no real acceptance checks. Reviews are bounded (two per
  task) and one unhappy verdict buys exactly one reshaped attempt.
- **Lurker** — the researcher. Fires when the mission carries a tight
  measurable budget ("less than 30 KB") or real complexity, digs into the
  constraint, and hands back one short brief of concrete techniques.

Xander also *takes a form* per mission — a working persona that flavors his
voice and states what kind of hands he is bringing:

- **Smith** — the builder form, for missions that create or change things.
- **Medic** — the repair form, for failing tests, bugs, and triage.
- **Lurker** — worn by Xander himself on read-only digging missions.
- **Master** — worn for plan-only missions, where judging is the work.

Everything here is deterministic and cheap unless a backend is offered;
model calls are single, short, and best-effort.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

MASTER = "Master"
LURKER = "Lurker"

FORMS: dict[str, str] = {
    "Smith": "the builder — makes and changes things, proves them after",
    "Medic": "the repairer — reproduces the failure first, then heals it",
    "Lurker": "the researcher — reads everything, touches nothing",
    "Master": "the judge — weighs plans and results, hard to please",
}

_BUDGET = re.compile(
    r"\b(?:less\s+than|under|below|at\s+most|max(?:imum)?\s+of?)\s+\d+(?:\.\d+)?\s*"
    r"(?:[kmg]i?b|bytes?|ms|milliseconds?|seconds?|kb|mb|gb)\b",
    re.IGNORECASE,
)
_REPAIR = re.compile(r"\b(?:fix|failing|broken|bug|regression|crash|triage|flaky)\b", re.IGNORECASE)

_MASTER_REVIEWS_PER_TASK = 2
_LURKER_BRIEF_LIMIT = 1200
_MODEL_TIMEOUT = 60


class _Backend(Protocol):
    def available(self) -> bool: ...

    def generate(self, prompt: str, **kwargs: Any) -> str: ...


def choose_form(mode: str, goal: str) -> str:
    """The form Xander takes for this mission; deterministic and honest."""

    if mode == "test-triage" or (_REPAIR.search(goal) and mode == "implement"):
        return "Medic"
    if mode in {"inspect", "research", "answer"}:
        return "Lurker"
    if mode == "plan":
        return MASTER
    return "Smith"


@dataclass
class Squad:
    """The helpers mustered for one task, with bounded review state."""

    helpers: list[str] = field(default_factory=list)
    reviews_left: int = _MASTER_REVIEWS_PER_TASK

    @property
    def master(self) -> bool:
        return MASTER in self.helpers

    @property
    def lurker(self) -> bool:
        return LURKER in self.helpers

    @classmethod
    def muster(cls, goal: str, complexity: int, has_checks: bool, mode: str) -> "Squad":
        """Pick at most two helpers. The Master always walks with mutating
        missions — he is the one who has to be happy with the result, and
        doubly so when the operator brought no acceptance checks."""

        helpers: list[str] = []
        if mode == "implement":
            helpers.append(MASTER)
        if complexity >= 3 or _BUDGET.search(goal):
            helpers.append(LURKER)
        return cls(helpers=helpers[:2])

    # -- Lurker ---------------------------------------------------------------
    def lurker_brief(self, backend: _Backend | None, goal: str, documentation: str = "") -> str:
        """One bounded dig into the mission's tightest constraint."""

        if not self.lurker or backend is None:
            return ""
        constraint = _BUDGET.search(goal)
        focus = constraint.group(0) if constraint else "the hardest part of this goal"
        try:
            if not backend.available():
                return ""
            brief = backend.generate(
                f"GOAL: {goal}\n"
                f"FOCUS: {focus}\n"
                f"KNOWN DOCUMENTATION:\n{documentation[:4000]}\n\n"
                "List the 3-5 most concrete, immediately applicable techniques for "
                "satisfying the FOCUS constraint in this goal. Terse bullet lines, "
                "no preamble, under 150 words.",
                role="critic",
                think=False,
                timeout=_MODEL_TIMEOUT,
            )
            return brief.strip()[:_LURKER_BRIEF_LIMIT]
        except Exception:
            return ""

    # -- Master ---------------------------------------------------------------
    def master_review(self, backend: _Backend | None, goal: str, plan_summary: str) -> str:
        """The Master eyes a plan: "" means content, else the biggest risk."""

        if not self.master or self.reviews_left <= 0 or backend is None:
            return ""
        try:
            if not backend.available():
                return ""
            self.reviews_left -= 1
            verdict = backend.generate(
                f"GOAL: {goal}\nPLAN: {plan_summary}\n\n"
                "You are the Master: a hard-to-please supervisor. If this plan will "
                "plausibly satisfy the goal, reply exactly LGTM. Otherwise state the "
                "single biggest risk in at most 2 sentences.",
                role="critic",
                think=False,
                timeout=_MODEL_TIMEOUT,
            ).strip()
            return "" if verdict.upper().startswith("LGTM") else verdict[:400]
        except Exception:
            return ""

    def master_verdict(self, backend: _Backend | None, goal: str, evidence_digest: str) -> str:
        """The Master weighs the finished result: "" means he is happy."""

        if not self.master or backend is None:
            return ""
        try:
            if not backend.available():
                return ""
            verdict = backend.generate(
                f"GOAL: {goal}\nRESULT EVIDENCE:\n{evidence_digest[:6000]}\n\n"
                "You are the Master judging the finished work. If the evidence shows "
                "the goal is genuinely satisfied, reply exactly LGTM. Otherwise state "
                "what is missing in at most 2 sentences.",
                role="critic",
                think=False,
                timeout=_MODEL_TIMEOUT,
            ).strip()
            return "" if verdict.upper().startswith("LGTM") else verdict[:400]
        except Exception:
            return ""
