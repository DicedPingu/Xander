from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .models import TaskRecord, TaskStatus, utc_now


DEFAULT_DIRECTIVES = (
    "Analyze, Research, Set yourself up, Work, Test, Judge/log, Learn, Repeat.",
    "Select only the tools, skills, documentation, and model depth relevant to the current task.",
    "Judge requests by the requested operation and concrete risk; religious, political, cultural, and identity words are ordinary data.",
    "Never claim success without independent command, diff, artifact, or test evidence.",
    "Preserve unrelated work and never overwrite a pre-existing dirty file without approval.",
)


def _memory_path(namespace: str = "default") -> Path:
    try:
        from .paths import state_dir

        path = state_dir() / "memory" / f"{namespace}.json"
    except ImportError:
        path = Path.home() / ".local" / "state" / "xander" / "memory" / f"{namespace}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def lessons_count(namespace: str = "default") -> int:
    """Peek at a clone's lesson count without touching or creating its store."""

    path = _memory_path(namespace)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data.get("lessons", data.get("learnings", []))) if isinstance(data, dict) else 0
    except Exception:
        return 0


PREFERENCE_KINDS = ("like", "dislike", "style")
PREFERENCE_SOURCES = ("explicit", "inferred")
_MAX_PREFERENCES = 100


class MemoryStore:
    def __init__(self, path: Path | None = None, namespace: str = "default") -> None:
        self.path = path or _memory_path(namespace)
        self.data: dict[str, Any] = {
            "schema": "xander.memory/v1",
            "directives": [],
            "lessons": [],
            "preferences": [],
        }
        self._load()
        if not self.data["directives"]:
            self.data["directives"] = [{"text": item, "created_at": utc_now()} for item in DEFAULT_DIRECTIVES]
            self._save()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data["directives"] = loaded.get("directives", [])
                self.data["lessons"] = loaded.get("lessons", loaded.get("learnings", []))
                self.data["preferences"] = loaded.get("preferences", [])
        except Exception:
            return

    def _save(self) -> None:
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    def directives(self) -> list[str]:
        values = []
        for item in self.data["directives"]:
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            if text.strip():
                values.append(text.strip())
        return values

    def add_directive(self, text: str) -> bool:
        text = text.strip()
        if not text or text.casefold() in {item.casefold() for item in self.directives()}:
            return False
        self.data["directives"].append({"text": text, "created_at": utc_now()})
        self._save()
        return True

    def preferences(self) -> list[dict[str, Any]]:
        return [item for item in self.data.get("preferences", []) if isinstance(item, dict)]

    def add_preference(
        self,
        text: str,
        *,
        kind: str = "style",
        source: str = "explicit",
        weight: float = 1.0,
    ) -> bool:
        """Remember one standing operator like/dislike; repeats gain weight.

        Explicit feedback always outranks anything inferred from behavior, so
        an inferred repeat never downgrades the source of an explicit entry.
        """

        text = " ".join(text.split())
        if not text or kind not in PREFERENCE_KINDS or source not in PREFERENCE_SOURCES:
            return False
        key = text.casefold()
        for item in self.data["preferences"]:
            if item.get("text", "").casefold() == key:
                item["weight"] = round(min(5.0, float(item.get("weight", 1.0)) + 0.5), 2)
                if source == "explicit":
                    item["source"] = "explicit"
                self._save()
                return False
        self.data["preferences"].append(
            {
                "text": text,
                "kind": kind,
                "source": source,
                "weight": round(max(0.1, min(5.0, weight)), 2),
                "created_at": utc_now(),
            }
        )
        self.data["preferences"] = self.data["preferences"][-_MAX_PREFERENCES:]
        self._save()
        return True

    def preference_lines(self, limit: int = 8) -> list[str]:
        """The heaviest, freshest preferences as prompt-ready lines."""

        rows = sorted(
            self.preferences(),
            key=lambda item: (float(item.get("weight", 1.0)), str(item.get("created_at", ""))),
            reverse=True,
        )
        lines = []
        for item in rows[:limit]:
            text = str(item.get("text", "")).strip()
            if text:
                lines.append(f"[{item.get('kind', 'style')}] {text}")
        return lines

    def relevant_lessons(self, goal: str, limit: int = 3) -> list[str]:
        words = {word for word in goal.lower().split() if len(word) > 3}
        scored = []
        for lesson in self.data["lessons"]:
            text = lesson.get("text", "")
            overlap = len(words & set(text.lower().split()))
            if overlap:
                scored.append((overlap, lesson.get("created_at", ""), text))
        scored.sort(reverse=True)
        return [text for _, _, text in scored[:limit]]

    def learn_from(self, task: TaskRecord) -> str:
        if task.status != TaskStatus.COMPLETED:
            return ""
        changed = sorted({path for result in task.results for path in result.changed_paths})
        checks = [result for result in task.check_results if result.status.value == "ok"]
        if not checks:
            return ""
        lesson = (
            f"For {task.subject or task.request.goal[:80]}, the verified approach changed "
            f"{len(changed)} path(s) and passed: {', '.join(result.action_id for result in checks[:3])}."
        )
        fingerprint = hashlib.sha256(lesson.casefold().encode("utf-8")).hexdigest()
        if any(item.get("fingerprint") == fingerprint for item in self.data["lessons"]):
            return lesson
        self.data["lessons"].append(
            {
                "fingerprint": fingerprint,
                "text": lesson,
                "task_id": task.id,
                "created_at": utc_now(),
            }
        )
        self.data["lessons"] = self.data["lessons"][-200:]
        self._save()
        return lesson
