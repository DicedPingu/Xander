from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable


CATEGORY_KEYWORDS = {
    "core-workflow": {"agent", "workflow", "memory", "planning", "orchestration", "evaluation"},
    "code-quality": {"test", "debug", "review", "refactor", "lint", "performance", "architecture"},
    "languages-frameworks": {"python", "rust", "java", "kotlin", "swift", "go", "cpp", "dotnet"},
    "mobile": {"android", "flutter", "ios", "mobile", "compose"},
    "web-ui": {"react", "vue", "angular", "svelte", "css", "frontend", "web", "ui"},
    "data-ai": {"ai", "llm", "ml", "data", "rag", "pandas", "numpy", "model"},
    "infrastructure-cloud": {"aws", "azure", "gcp", "docker", "kubernetes", "terraform", "cloud"},
    "security": {"security", "audit", "vulnerability", "penetration", "auth", "threat"},
    "product-design": {"product", "design", "ux", "accessibility", "prototype"},
    "content-growth": {"content", "seo", "marketing", "copy", "growth", "social"},
    "automation-integrations": {"automation", "mcp", "api", "integration", "zapier", "n8n"},
    "documents-media": {"pdf", "slides", "document", "audio", "video", "image"},
    "research-analysis": {"research", "analysis", "intelligence", "compare", "investigate"},
}
DAILY_NAMES = {
    "complete-task-loop",
    "agent-sort",
    "ai-agent-development",
    "code-review",
    "debugging",
    "testing",
}


def _cache_db() -> Path:
    try:
        from .paths import cache_dir

        path = cache_dir() / "skills.sqlite3"
    except ImportError:
        path = Path.home() / ".cache" / "xander" / "skills.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def skill_roots() -> list[Path]:
    home = Path.home()
    roots = [
        home / ".local" / "share" / "agent-skills" / "library",
        home / ".agents" / "skills",
        home / ".codex" / "skills",
        home / ".claude" / "skills",
    ]
    return [root for root in roots if root.exists()]


def _frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    values: dict[str, str] = {}
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def whole_tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        digest.update(str(path.relative_to(directory)).encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            continue
        digest.update(b"\0")
    return digest.hexdigest()


def classify(name: str, description: str) -> str:
    words = set(re.findall(r"[a-z0-9]+", f"{name} {description}".lower()))
    best = "uncategorized"
    score = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        overlap = len(words & keywords)
        if overlap > score:
            best, score = category, overlap
    return best


class SkillRegistry:
    def __init__(self, database: Path | None = None, roots: Iterable[Path] | None = None) -> None:
        self.database = database or _cache_db()
        self.roots = list(roots or skill_roots())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def refresh(self) -> dict[str, int]:
        records: dict[tuple[str, str], dict[str, Any]] = {}
        for root in self.roots:
            for skill_file in root.rglob("SKILL.md"):
                if any(part in {".git", "node_modules", "__pycache__"} for part in skill_file.parts):
                    continue
                try:
                    text = skill_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                metadata = _frontmatter(text)
                name = metadata.get("name") or skill_file.parent.name
                description = metadata.get("description", "")[:2000]
                tree_hash = whole_tree_hash(skill_file.parent)
                key = (name, tree_hash)
                provider = root.parent.name if root.name == "skills" else str(root)
                record = records.setdefault(
                    key,
                    {
                        "name": name,
                        "description": description,
                        "triggers": " ".join(sorted(set(re.findall(r"[a-z0-9-]+", f"{name} {description}".lower())))),
                        "category": classify(name, description),
                        "bucket": "daily" if name in DAILY_NAMES else "library",
                        "tree_hash": tree_hash,
                        "path": str(skill_file),
                        "providers": [],
                    },
                )
                record["providers"].append(provider)
        with self._connect() as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS skills;
                DROP TABLE IF EXISTS skills_fts;
                CREATE TABLE skills (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    triggers TEXT NOT NULL,
                    category TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    providers TEXT NOT NULL,
                    tree_hash TEXT NOT NULL,
                    path TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE skills_fts USING fts5(name, description, triggers, category, content='skills', content_rowid='id');
                """
            )
            for record in sorted(records.values(), key=lambda item: (item["name"], item["tree_hash"])):
                cursor = connection.execute(
                    "INSERT INTO skills(name,description,triggers,category,bucket,providers,tree_hash,path) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        record["name"],
                        record["description"],
                        record["triggers"],
                        record["category"],
                        record["bucket"],
                        json.dumps(sorted(set(record["providers"]))),
                        record["tree_hash"],
                        record["path"],
                    ),
                )
                connection.execute(
                    "INSERT INTO skills_fts(rowid,name,description,triggers,category) VALUES(?,?,?,?,?)",
                    (cursor.lastrowid, record["name"], record["description"], record["triggers"], record["category"]),
                )
        return {
            "indexed": len(records),
            "daily": sum(record["bucket"] == "daily" for record in records.values()),
            "library": sum(record["bucket"] == "library" for record in records.values()),
        }

    def ensure(self) -> None:
        if not self.database.exists():
            self.refresh()

    def search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        self.ensure()
        tokens = re.findall(r"[a-z0-9-]+", query.lower())[:12]
        if not tokens:
            return []
        expression = " OR ".join(f'"{token}"' for token in tokens)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.*, bm25(skills_fts, 5.0, 2.0, 1.0, 0.5) AS score
                FROM skills_fts JOIN skills s ON s.id = skills_fts.rowid
                WHERE skills_fts MATCH ?
                ORDER BY score, CASE s.bucket WHEN 'daily' THEN 0 ELSE 1 END, s.name
                LIMIT ?
                """,
                (expression, max(1, min(limit, 10))),
            ).fetchall()
        return [
            {
                **dict(row),
                "providers": json.loads(row["providers"]),
            }
            for row in rows
        ]

    def list(self, limit: int = 100, bucket: str | None = None) -> list[dict[str, Any]]:
        self.ensure()
        limit = max(1, min(limit, 10_000))
        with self._connect() as connection:
            if bucket:
                rows = connection.execute(
                    "SELECT * FROM skills WHERE bucket=? ORDER BY category,name LIMIT ?",
                    (bucket, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM skills ORDER BY bucket,category,name LIMIT ?",
                    (limit,),
                ).fetchall()
        return [{**dict(row), "providers": json.loads(row["providers"])} for row in rows]

    def load_selected(self, query: str, limit: int = 3, max_chars: int = 24_000) -> list[dict[str, str]]:
        selected = []
        for record in self.search(query, limit=limit):
            path = Path(record["path"])
            try:
                content = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
            except OSError:
                continue
            selected.append({"name": record["name"], "path": str(path), "content": content})
        return selected

    def doctor(self) -> dict[str, Any]:
        self.ensure()
        broken = []
        for root in self.roots:
            for path in root.rglob("*"):
                if path.is_symlink() and not path.exists():
                    broken.append(str(path))
        with self._connect() as connection:
            count = connection.execute("SELECT count(*) FROM skills").fetchone()[0]
            daily = connection.execute("SELECT count(*) FROM skills WHERE bucket='daily'").fetchone()[0]
        return {
            "database": str(self.database),
            "indexed": count,
            "daily": daily,
            "library": count - daily,
            "roots": [str(root) for root in self.roots],
            "broken_symlink_count": len(broken),
            "broken_symlink_sample": broken[:20],
        }
