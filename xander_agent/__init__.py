"""Public package metadata shared by Xander's human and agent interfaces."""

from __future__ import annotations

__version__ = "1.0.0"

MANTRA_PHASES: tuple[str, ...] = (
    "Analyze",
    "Research",
    "Set yourself up",
    "Work",
    "Test",
    "Judge/log",
    "Learn",
    "Repeat",
)

# Xander carries a little of every ability and deploys only what the task
# needs: llm (language judgment), agent (autonomous loop), algorithm
# (deterministic execution and judging), bot (tireless retries), oracle
# (research and current docs), quartermaster (selects minimal gear), and
# scribe (logs evidence, keeps lessons).
ABILITIES: tuple[str, ...] = (
    "llm",
    "agent",
    "algorithm",
    "bot",
    "oracle",
    "quartermaster",
    "scribe",
)

# The seven are not a flat list; they pair up, and the pairs are what a task
# actually calls on. Reasoning is language judgment checked by deterministic
# execution. Execution is an autonomous loop plus the patience to retry.
# Knowledge is what he reads in and what he writes back out. Gear stands alone:
# choosing the minimum is its own discipline, and it constrains all the others.
ABILITY_GROUPS: dict[str, tuple[str, ...]] = {
    "reasoning": ("llm", "algorithm"),
    "execution": ("agent", "bot"),
    "knowledge": ("oracle", "scribe"),
    "gear": ("quartermaster",),
}


def ability_group(ability: str) -> str:
    """Which family an ability belongs to, or "" if it is not one of his."""

    for group, members in ABILITY_GROUPS.items():
        if ability in members:
            return group
    return ""


__all__ = ["ABILITIES", "ABILITY_GROUPS", "MANTRA_PHASES", "ability_group", "__version__"]
