"""Xander's working voice: short, relevant, first-person lines while he works.

The voice is a personality, not a log format. He says what he is about to do
and which default he picked ("making the background green, unless told
otherwise"), admits failures plainly, and audibly tires of hitting the same
wall — repetition drains him and pushes him to change the plan's shape.
Progress reports address the Master by name, because the Master does the
judging and has to end up happy.

Every line is bounded: at most one small-model call per moment with a short
timeout, a deterministic template whenever the model is down or slow, and a
hard cap on lines per task so the voice never becomes noise.
"""

from __future__ import annotations

from typing import Any, Protocol

MOMENTS = (
    "kickoff",
    "approach",
    "action",
    "setback",
    "victory",
    "progress",
    "question",
    "feedback_ack",
)

# Moments that stay audible even after the per-task cap is reached: wins,
# losses, direct questions, and acknowledgements must never be swallowed.
_ALWAYS_AUDIBLE = {"setback", "victory", "question", "feedback_ack"}

_LINE_CAPS = {"chatty": 10, "quiet": 4, "off": 0}
_MODEL_TIMEOUT = 30
_MAX_LINE = 200
_FACTUAL_MOMENTS = {
    "kickoff",
    "approach",
    "action",
    "setback",
    "victory",
    "progress",
    "question",
    "feedback_ack",
}


class _Backend(Protocol):
    def available(self) -> bool: ...

    def generate(self, prompt: str, **kwargs: Any) -> str: ...


def _clip(text: str, limit: int = _MAX_LINE) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Commentator:
    """One task's voice. Create a fresh one per task so caps and fatigue reset."""

    def __init__(
        self,
        backend: _Backend | None = None,
        memory: Any = None,
        *,
        voice: str = "chatty",
        speaker: str = "Xander",
    ) -> None:
        self.backend = backend
        self.memory = memory
        self.voice = voice if voice in _LINE_CAPS else "chatty"
        self.speaker = speaker
        self.fatigue = 0  # consecutive setbacks without a win
        self._spoken = 0
        self._model_calls = 0

    # -- public ---------------------------------------------------------------
    def say(self, moment: str, **context: Any) -> str:
        """Return one voiced line for this moment, or "" when staying quiet."""

        if self.voice == "off" or moment not in MOMENTS:
            return ""
        if self._spoken >= _LINE_CAPS[self.voice] and moment not in _ALWAYS_AUDIBLE:
            return ""
        if moment == "setback":
            self.fatigue += 1
        elif moment == "victory":
            self.fatigue = 0
        line = self._model_line(moment, context) or self._template(moment, context)
        if line:
            self._spoken += 1
        return _clip(line)

    # -- model path -----------------------------------------------------------
    def _model_line(self, moment: str, context: dict[str, Any]) -> str:
        if (
            moment in _FACTUAL_MOMENTS
            or self.voice != "chatty"
            or self.backend is None
            or self._model_calls >= 6
        ):
            return ""
        try:
            if not self.backend.available():
                return ""
            preferences = []
            if self.memory is not None and hasattr(self.memory, "preference_lines"):
                preferences = self.memory.preference_lines(limit=5)
            mood = (
                "fresh" if self.fatigue == 0
                else "focused after one setback" if self.fatigue == 1
                else "visibly tired of repeating the same failure; wants a different shape of plan"
            )
            system = (
                f"You voice {self.speaker}, a local coding agent, in one short first-person line. "
                f"Mood: {mood}. Mention the concrete choice being made (color, file, technique) "
                "when the context names one. State chosen defaults as '..., unless told otherwise'. "
                "No greetings, no emoji, no quotes, at most 140 characters. "
                + ("Operator preferences: " + "; ".join(preferences) if preferences else "")
            )
            template = self._template(moment, context)
            prompt = f"Moment: {moment}\nContext: {context}\nSay it in your own words. Fallback wording: {template}"
            self._model_calls += 1
            line = self.backend.generate(
                prompt, role="classifier", system=system, think=False, timeout=_MODEL_TIMEOUT
            )
            line = line.strip().strip('"')
            return _clip(line) if line and len(line) <= _MAX_LINE + 60 else ""
        except Exception:
            return ""

    # -- deterministic fallback ------------------------------------------------
    def _template(self, moment: str, context: dict[str, Any]) -> str:
        goal = _clip(str(context.get("goal", "")), 90)
        summary = _clip(str(context.get("summary", "")), 110)
        reason = _clip(str(context.get("reason", "")), 90)
        if moment == "kickoff":
            form = str(context.get("form", "")).strip()
            opener = f"Taking {form} form for this one. " if form and form.lower() != "xander" else ""
            return f"{opener}On it — {goal}. Looking before I touch anything."
        if moment == "approach":
            default = _clip(str(context.get("default", "")), 70)
            tail = f" Going with {default}, unless told otherwise." if default else ""
            counts = ""
            if context.get("actions") is not None:
                counts = f" {context.get('actions')} action(s), {context.get('checks', 0)} check(s)."
            return f"Here's my shape: {summary}.{counts}{tail}"
        if moment == "action":
            return f"Now: {_clip(str(context.get('expected', 'the next step')), 120)}."
        if moment == "setback":
            if self.fatigue >= 3:
                return (
                    f"That's {self.fatigue} times into the same wall — {reason}. "
                    "I'm tired of this loop; I'm changing the shape of the plan entirely."
                )
            if self.fatigue == 2:
                return f"Same wall again — {reason}. Coming at it from a different side."
            return f"That didn't hold: {reason}. Adjusting and going again."
        if moment == "victory":
            checks = context.get("checks", 0)
            paths = context.get("paths", 0)
            proof = f"{checks} check(s) green" + (f", {paths} path(s) touched" if paths else "")
            return f"Done and proven — {proof}."
        if moment == "progress":
            master = str(context.get("master", "Master"))
            attempt = context.get("attempt", 1)
            ok = context.get("actions_ok", 0)
            total = context.get("actions_total", 0)
            checks_passed = context.get("checks_passed", 0)
            checks_total = context.get("checks_total", 0)
            state = _clip(str(context.get("state", "")), 80)
            tail = f" {state}" if state else ""
            return (
                f"{master} — attempt {attempt}: {ok}/{total} action(s) landed, "
                f"{checks_passed}/{checks_total} check(s) passing.{tail}"
            )
        if moment == "question":
            options = context.get("options") or []
            default = _clip(str(context.get("default", "")), 60)
            listed = ", ".join(_clip(str(item), 40) for item in list(options)[:4])
            tail = f" My default is {default} unless you steer." if default else ""
            return f"I need a pick: {listed}.{tail}"
        if moment == "feedback_ack":
            return f"Noted — {_clip(str(context.get('text', '')), 120)}. I'll work that way from here."
        return ""
