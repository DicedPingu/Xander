"""The caliber catalog — the one place that decides which model fires in which role.

``backend.py``, ``variants.py`` and the legacy top-level ``config.py`` each used
to carry their own copy of the routing table, and they drifted: the legacy config
already knew about the q8_0 vision build while the live package still fired the
q4 one. Everything reads this module now.

Resolution is deliberately conservative. A routed tag is replaced only by a
*strictly higher precision* build of the same family and parameter size. A
re-tag of identical weights (``:8b`` and ``:8b-v2`` share a digest) is reported
as an alternative but never swapped in: reloading it costs a model eviction and
buys nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CLOUD_PREFIX = "anthropic/"

# A bare Ollama tag is q4_K_M. Rank it between q4 and q5 so an explicit q4_0
# never looks like an upgrade over the default build.
BASELINE_QUANT_BITS = 4.5

_PARAM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)b(?![\w])", re.IGNORECASE)
_QUANT_RE = re.compile(r"(?:^|[-_])q(\d+)", re.IGNORECASE)
_FLOAT_RE = re.compile(r"(?:^|[-_])(?:bf|fp|f)(16|32)(?![\w])", re.IGNORECASE)
_VERSION_RE = re.compile(r"(?:^|[-_])v(\d+)(?![\w])", re.IGNORECASE)


@dataclass(frozen=True)
class Caliber:
    """One role and the model that serves it."""

    role: str
    model: str
    purpose: str
    kind: str = "chat"


# One resident model. The card holds one ~6.5 GB build at a time and every
# swap between builds is a 10-20 s stall that reads as a freeze, so every
# chat role fires the same heretic build: Qwen3.5-based, tools + vision +
# thinking, 32k context baked into its Modelfile. The alternatives below are
# installed and catalogued for benchmarking, never swapped in mid-mission.
MAIN_MODEL = "qwenpaw-9b-heretic:latest"
ALTERNATIVE_MODELS: tuple[str, ...] = ("qwen3.8-9b-heretic:latest", "gemma4-e4b-heretic:latest")

CALIBERS: tuple[Caliber, ...] = (
    Caliber("coder", MAIN_MODEL, "precise edits, command synthesis"),
    Caliber("planner", MAIN_MODEL, "analysis, planning, replanning"),
    Caliber("classifier", MAIN_MODEL, "routing: intent detection, commentary, quick judgments"),
    Caliber("critic", MAIN_MODEL, "the squad's judgment seat"),
    Caliber(
        "embedder",
        "qwen3-embedding:0.6b",
        "vector recall; installed and catalogued, not yet wired into the engine",
        kind="embed",
    ),
)

#: Role -> model for everything the engine actually generates with.
DEFAULT_MODELS: dict[str, str] = {c.role: c.model for c in CALIBERS if c.kind == "chat"}

#: Embedders emit vectors, never prose, so they sit outside chat routing and
#: outside the abliterated-only policy that guards it.
EMBEDDER: str = next(c.model for c in CALIBERS if c.kind == "embed")

#: Tried in order when a role has no model of its own.
FALLBACK_ROLES: tuple[str, ...] = ("coder", "planner", "critic")

PURPOSES: dict[str, str] = {c.role: c.purpose for c in CALIBERS}

_REASONING_GOAL = re.compile(
    r"\b(?:architect|diagnos|investigat|migrat|refactor|review|triage|design)\w*\b",
    re.IGNORECASE,
)


def role_for_task(mode: str, goal: str = "", complexity: int = 0, retry: bool = False) -> str:
    """Choose the chat role that matches the work about to be generated."""

    if mode in {"answer", "inspect", "research"}:
        return "critic"
    if mode == "test-triage":
        return "planner"
    if mode == "plan" or retry or complexity >= 4 or _REASONING_GOAL.search(goal):
        return "planner"
    return "coder"


def route_reason(mode: str, goal: str = "", complexity: int = 0, retry: bool = False) -> str:
    role = role_for_task(mode, goal, complexity, retry)
    if role == "critic":
        return "synthesis, research, or judgment needs the deeper prose model"
    if role == "planner":
        return "planning, diagnosis, architecture, or a changed retry needs deeper reasoning"
    return "bounded implementation work benefits from the precise coding model"


def is_cloud(model: str) -> bool:
    return model.casefold().startswith(CLOUD_PREFIX)


#: Builds with refusal training removed at the weights: abliterated (huihui)
#: or heretic (p-e-w/heretic directional ablation). Both satisfy the operator's
#: local-model policy; a plain vendor build does not.
_UNCENSORED_MARKS = ("abliterat", "heretic", "uncensored")


def is_abliterated(model: str) -> bool:
    lowered = model.casefold()
    return any(mark in lowered for mark in _UNCENSORED_MARKS)


def family(model: str) -> str:
    """The repository part of a model id, without its tag."""

    return model.rsplit(":", 1)[0]


def tag(model: str) -> str:
    return model.rsplit(":", 1)[1] if ":" in model else ""


def parameter_size(model: str) -> str:
    """``8b`` from ``huihui_ai/qwen3-abliterated:8b-v2``; empty when unstated."""

    match = _PARAM_RE.search(tag(model))
    return match.group(1).lower() + "b" if match else ""


def precision_bits(model: str) -> float:
    """Weight precision implied by a tag, for ranking sibling builds."""

    label = tag(model)
    float_match = _FLOAT_RE.search(label)
    if float_match:
        return float(float_match.group(1))
    quant_match = _QUANT_RE.search(label)
    if quant_match:
        return float(quant_match.group(1))
    return BASELINE_QUANT_BITS


def _version(model: str) -> int:
    match = _VERSION_RE.search(tag(model))
    return int(match.group(1)) if match else 0


def is_sibling(model: str, candidate: str) -> bool:
    """Same family and same parameter count — a drop-in build of one model."""

    if candidate == model or is_cloud(model) or is_cloud(candidate):
        return False
    if family(candidate) != family(model):
        return False
    if parameter_size(candidate) != parameter_size(model):
        return False
    # Operator policy: a local build never leaves the abliterated set.
    return is_abliterated(candidate) == is_abliterated(model)


def siblings(model: str, installed) -> list[str]:
    return sorted(candidate for candidate in installed if is_sibling(model, candidate))


def best_installed(model: str, installed) -> str:
    """The highest-precision installed build of ``model``'s family, else ``model``."""

    if is_cloud(model):
        return model
    best, best_bits = model, precision_bits(model)
    for candidate in siblings(model, installed):
        bits = precision_bits(candidate)
        if bits > best_bits:
            best, best_bits = candidate, bits
    return best


def alternatives(model: str, installed) -> list[str]:
    """Installed siblings that are not a precision win — re-tags and rebuilds."""

    resolved = best_installed(model, installed)
    return [candidate for candidate in siblings(model, installed) if candidate != resolved]


def resolve(role: str, installed, routing: dict[str, str] | None = None) -> str:
    """The model a role should actually fire, given what is on disk."""

    table = dict(routing or DEFAULT_MODELS)
    preferred = table.get(role) or table.get("coder") or DEFAULT_MODELS["coder"]
    return best_installed(preferred, installed)


def ordered_models(role: str, installed, routing: dict[str, str] | None = None) -> list[str]:
    """Preferred model first, then the fallback roles; installed builds only.

    Falls back to the unfiltered order when nothing is installed, so the caller
    still produces a real error from the backend instead of an empty attempt.
    """

    table = dict(routing or DEFAULT_MODELS)
    ordered: list[str] = []
    for candidate_role in (role, *FALLBACK_ROLES):
        model = table.get(candidate_role)
        if not model:
            continue
        resolved = best_installed(model, installed)
        if resolved not in ordered:
            ordered.append(resolved)
    if not ordered:
        ordered.append(resolve(role, installed, table))
    # The catalogued alternatives come last: an installed heretic build beats
    # an error when the main build is missing.
    for alternative in ALTERNATIVE_MODELS:
        if alternative in set(installed) and alternative not in ordered:
            ordered.append(alternative)
    present = [model for model in ordered if is_cloud(model) or model in set(installed)]
    return present or ordered


def upgrades(routing: dict[str, str], installed) -> dict[str, dict[str, str]]:
    """Roles whose routed tag is beaten by a build already on disk."""

    report: dict[str, dict[str, str]] = {}
    for role, model in routing.items():
        better = best_installed(model, installed)
        if better != model:
            report[role] = {
                "routed": model,
                "better": better,
                "reason": f"{precision_bits(better):g}-bit build of the same {parameter_size(model) or 'family'} weights is installed",
            }
    return report
