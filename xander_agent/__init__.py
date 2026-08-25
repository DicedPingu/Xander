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

__all__ = ["ABILITIES", "MANTRA_PHASES", "__version__"]
