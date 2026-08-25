"""Xander's memory — the part that makes him *yours*.

Two kinds of memory, both JSON-backed and loaded on every run:

  * directives  — standing orders from the operator, obeyed to the letter and
    injected verbatim into every plan/execution prompt. These are law.
  * learnings   — things Xander discovered that worked (or didn't) for a given
    kind of task, surfaced as hints next time a similar task appears.

Directives win over Xander's own judgement, always. If a directive and a
"best practice" conflict, the directive is followed and the conflict is noted.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from config import MEMORY_FILE


class Memory:
    def __init__(self, path: Path = MEMORY_FILE):
        self.path = path
        self.data = {"directives": [], "learnings": []}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                self.data["directives"] = loaded.get("directives", [])
                self.data["learnings"] = loaded.get("learnings", [])
        except Exception:
            pass  # corrupt memory should never brick the agent

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    # ── directives (law) ─────────────────────────────────────────────────────
    def add_directive(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        if any(d["text"].lower() == text.lower() for d in self.data["directives"]):
            return
        self.data["directives"].append({"text": text, "ts": time.strftime("%Y-%m-%d")})
        self._save()

    def forget_directive(self, index: int) -> str | None:
        if 0 <= index < len(self.data["directives"]):
            removed = self.data["directives"].pop(index)
            self._save()
            return removed["text"]
        return None

    def directives(self) -> list[str]:
        return [d["text"] for d in self.data["directives"]]

    def directive_block(self) -> str:
        """The verbatim law block injected into every task prompt."""
        ds = self.directives()
        if not ds:
            return ""
        lines = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(ds))
        return (
            "OPERATOR STANDING DIRECTIVES — follow these to the letter. They "
            "override your defaults and any generic best practice. If one "
            "conflicts with what you would otherwise do, obey the directive and "
            "note the conflict in one line:\n" + lines
        )

    # ── learnings (hints) ────────────────────────────────────────────────────
    def add_learning(self, tags: list[str], text: str, worked: bool = True) -> None:
        text = text.strip()
        if not text:
            return
        self.data["learnings"].append({
            "tags": [t.lower() for t in tags],
            "text": text,
            "worked": worked,
            "ts": time.strftime("%Y-%m-%d"),
        })
        self.data["learnings"] = self.data["learnings"][-500:]  # bound growth
        self._save()

    def recall(self, tags: list[str], limit: int = 6) -> list[str]:
        tagset = {t.lower() for t in tags}
        scored = []
        for item in self.data["learnings"]:
            overlap = len(tagset & set(item.get("tags", [])))
            if overlap:
                scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _, item in scored[:limit]:
            mark = "✓" if item.get("worked", True) else "✗"
            out.append(f"{mark} {item['text']}")
        return out

    def recall_block(self, tags: list[str]) -> str:
        hits = self.recall(tags)
        if not hits:
            return ""
        return "What worked (✓) or failed (✗) on similar tasks before:\n" + "\n".join(hits)
