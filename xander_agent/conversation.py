from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .backend import Backend, OllamaBackend, backend_for
from .calibers import role_for_task, route_reason
from .memory import MemoryStore
from .research import Researcher
from .variants import load_variant


_EVIDENCE_NAMES = {
    "AGENTS.md",
    "CAMPAIGN_LOG.md",
    "Cargo.toml",
    "MANIFEST.md",
    "README.md",
    "README.rst",
    "package.json",
    "pyproject.toml",
}
_EVIDENCE_SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "target",
    "vendor",
}


def _workspace_evidence(workspace: Path, limit: int = 18_000) -> str:
    candidates: list[Path] = []
    for path in workspace.rglob("*"):
        if not path.is_file() or path.name not in _EVIDENCE_NAMES:
            continue
        relative = path.relative_to(workspace)
        if len(relative.parts) > 3 or any(part in _EVIDENCE_SKIP_PARTS for part in relative.parts):
            continue
        candidates.append(path)

    def rank(path: Path) -> tuple[int, int, str]:
        relative = path.relative_to(workspace)
        name = path.name.casefold()
        if name.startswith("readme") and len(relative.parts) <= 2:
            priority = 0
        elif name in {"package.json", "pyproject.toml", "cargo.toml"} and len(relative.parts) <= 2:
            priority = 1
        elif name == "manifest.md":
            priority = 2
        elif name == "agents.md":
            priority = 3
        elif name == "campaign_log.md":
            priority = 4
        else:
            priority = 5
        return priority, len(relative.parts), str(relative)

    sections: list[str] = []
    remaining = limit
    for path in sorted(candidates, key=rank):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        content = content[: min(6_000, remaining)].strip()
        if not content:
            continue
        sections.append(f"SOURCE {path.relative_to(workspace)}:\n{content}")
        remaining -= len(content)
        if remaining <= 0:
            break
    return "\n\n".join(sections)


def talk(
    workspace: Path,
    variant: str,
    message: str,
    history: Sequence[dict[str, str]] = (),
    *,
    backend: Backend | None = None,
) -> dict[str, Any]:
    profile = load_variant(variant)
    managed_backend = backend is None
    brain = backend or backend_for(profile.model_routing)
    if not brain.available():
        raise RuntimeError("no configured language model is reachable")
    memory = MemoryStore(namespace=profile.memory_namespace or variant)
    recent = [
        {"role": str(turn.get("role", "user")), "content": str(turn.get("content", ""))[:2_000]}
        for turn in history[-12:]
        if str(turn.get("content", "")).strip()
    ]
    local_context = Researcher(workspace)._local_context()[:8_000]
    workspace_evidence = _workspace_evidence(workspace)
    role = role_for_task("answer", message, complexity=2)
    reason = route_reason("answer", message, complexity=2)
    prompt = (
        f"SELECTED XANDER CLONE: {variant}\n"
        f"WORKSPACE: {workspace}\n"
        f"RECENT CONVERSATION: {json.dumps(recent, ensure_ascii=False)}\n"
        f"OPERATOR PREFERENCES: {json.dumps(memory.preference_lines(), ensure_ascii=False)}\n"
        f"CLONE DIRECTIVES: {json.dumps(profile.directives, ensure_ascii=False)}\n\n"
        f"AUTHORITATIVE WORKSPACE EVIDENCE:\n{workspace_evidence or '(no project documents found)'}\n\n"
        f"LOCAL WORKSPACE INVENTORY:\n{local_context}\n\n"
        f"OPERATOR: {message}\n\n"
        "Reply as the selected Xander clone in a real conversation. Be direct and useful. "
        "Use workspace facts when relevant. Treat current README and manifest content as stronger "
        "evidence than historical campaign notes. Copy every exact command and path from a named "
        "local source above; never invent one. If the evidence does not contain an exact answer, "
        "say that it is not verified. Never claim you changed, ran, tested, or proved anything in "
        "chat. If the operator is actually asking for work, say briefly that /work runs it as "
        "autonomous work and state what outcome you understood; discussion never starts work by itself."
    )
    system = (
        "You are the conversational side of Xander. Conversation is not a task record. "
        "Answer naturally, preserve context across turns, and keep factual claims honest. "
        "Do not infer exact commands, paths, or proof artifacts from filenames alone."
    )
    try:
        text = brain.generate(
            prompt,
            role=role,
            system=system,
            think=False,
            timeout=240,
        ).strip()
    except RuntimeError as exc:
        if not managed_backend or "resource gate" not in str(exc):
            raise
        brain = OllamaBackend(models=profile.model_routing, min_available_memory_mb=900)
        role = "classifier"
        reason = "memory was tight, so conversation used the smaller language model"
        text = brain.generate(
            prompt,
            role=role,
            system=system,
            think=False,
            timeout=240,
        ).strip()
    if not text:
        raise RuntimeError("the selected language model returned an empty reply")
    stats = dict(getattr(brain, "last_stats", {}) or {})
    return {"text": text[:6_000], "role": role, "reason": reason, "stats": stats}
