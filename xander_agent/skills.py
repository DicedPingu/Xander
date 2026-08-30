from __future__ import annotations

import hashlib
import json
import os
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
        path = Path(__file__).resolve().parents[1] / "cache" / "skills.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def authored_root() -> Path:
    """Where skills Xander writes for himself live."""

    override = os.environ.get("XANDER_AUTHORED_SKILLS")
    if override:
        return Path(override).expanduser()
    from .paths import agent_dir

    return agent_dir() / "knowledge" / "skills" / "xander-authored"


_SLUG = re.compile(r"[^a-z0-9-]+")


def slugify(name: str) -> str:
    slug = _SLUG.sub("-", name.strip().lower()).strip("-")[:48]
    if not slug:
        raise ValueError("a skill needs a name with at least one letter or digit")
    return slug


def skill_roots() -> list[Path]:
    home = Path.home()
    from .paths import agent_dir, shared_skills_dir

    roots = [
        shared_skills_dir(),
        agent_dir() / "knowledge" / "skills",
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
                declared_category = metadata.get("category", "")
                tree_hash = whole_tree_hash(skill_file.parent)
                key = (name, tree_hash)
                provider = root.parent.name if root.name == "skills" else str(root)
                record = records.setdefault(
                    key,
                    {
                        "name": name,
                        "description": description,
                        "triggers": " ".join(
                            sorted(
                                set(
                                    re.findall(
                                        r"[a-z0-9-]+",
                                        f"{name} {description} {text[:16_000]}".lower(),
                                    )
                                )
                            )
                        ),
                        "category": declared_category
                        if declared_category in CATEGORY_KEYWORDS
                        else classify(name, description),
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

    def _dense_ranking(self, connection: Any, query: str, pool: int) -> list[str]:
        """Cosine ranking over stored card vectors. Empty when unavailable."""

        from . import semantic

        vectors = semantic.load_vectors(connection)
        if not vectors:
            return []
        from .calibers import EMBEDDER

        embedded = semantic.embed(
            [query],
            model=EMBEDDER,
            base_url=os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434"),
            timeout=30,
        )
        if not embedded:
            return []
        return [tree_hash for tree_hash, _ in semantic.rank(embedded[0], vectors)[:pool]]

    def search(self, query: str, limit: int = 12, *, semantic_search: bool = True) -> list[dict[str, Any]]:
        """Hybrid retrieval: BM25 and dense cosine, fused with RRF.

        Dense is strictly additive. If nothing has been embedded yet, or the
        embedder is unreachable, this returns exactly the BM25 result it always
        did.
        """

        self.ensure()
        tokens = re.findall(r"[a-z0-9-]+", query.lower())[:12]
        limit = max(1, min(limit, 100))
        pool = max(limit * 5, 50)

        with self._connect() as connection:
            lexical: list[str] = []
            if tokens:
                expression = " OR ".join(f'"{token}"' for token in tokens)
                lexical = [
                    row["tree_hash"]
                    for row in connection.execute(
                        """
                        SELECT s.tree_hash, bm25(skills_fts, 5.0, 2.0, 1.0, 0.5) AS score
                        FROM skills_fts JOIN skills s ON s.id = skills_fts.rowid
                        WHERE skills_fts MATCH ?
                        ORDER BY score, CASE s.bucket WHEN 'daily' THEN 0 ELSE 1 END, s.name
                        LIMIT ?
                        """,
                        (expression, pool),
                    ).fetchall()
                ]

            dense: list[str] = []
            if semantic_search and query.strip():
                try:
                    dense = self._dense_ranking(connection, query, pool)
                except Exception:
                    dense = []

            if not lexical and not dense:
                return []

            from .semantic import fuse

            order = fuse([r for r in (lexical, dense) if r]) if dense else lexical
            wanted = order[:limit]
            if not wanted:
                return []
            placeholders = ",".join("?" for _ in wanted)
            rows = connection.execute(
                f"SELECT * FROM skills WHERE tree_hash IN ({placeholders})",
                wanted,
            ).fetchall()

        position = {tree_hash: index for index, tree_hash in enumerate(wanted)}
        rows = sorted(rows, key=lambda row: (position.get(row["tree_hash"], len(wanted)), row["bucket"] != "daily", row["name"]))
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for row in rows:
            if row["tree_hash"] in seen:
                continue
            seen.add(row["tree_hash"])
            results.append({**dict(row), "providers": json.loads(row["providers"])})
        return results[:limit]

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

    def assemble(
        self,
        query: str,
        *,
        preferred_groups: Iterable[str] = (),
        context_budget: int = 18_000,
        namespace: str = "default",
    ) -> list[dict[str, str]]:
        """Build layered gear by relevance and context cost, never a skill count.

        One general workflow skill anchors every mission. Task matches then fill
        the remaining context budget and retain their category as a research
        hub. The number of skills is an outcome of relevance and size.
        """

        budget = max(2_000, context_budget)
        query_tokens = {
            token
            for token in re.findall(r"[a-z0-9-]+", query.casefold())
            if len(token) > 3 and token not in {"folder", "good", "have", "here", "make", "working"}
        }
        preferred = set(preferred_groups)
        candidates: list[dict[str, Any]] = []
        daily = self.list(limit=100, bucket="daily")
        anchor = next((item for item in daily if item["name"] == "complete-task-loop"), None)
        if anchor:
            candidates.append(anchor)
        for item in self.search(query, limit=100):
            experience_prefix = "xander-experience-"
            if item["name"].startswith(experience_prefix) and not item["name"].startswith(
                f"{experience_prefix}{slugify(namespace)}-"
            ):
                continue
            words = set(
                re.findall(
                    r"[a-z0-9-]+",
                    f"{item['name']} {item['description']} {item['category']} {item.get('triggers', '')}".casefold(),
                )
            )
            overlap = len(query_tokens & words)
            if not overlap and item["category"] not in preferred:
                continue
            item = {**item, "relevance": overlap + (1 if item["category"] in preferred else 0)}
            candidates.append(item)
        candidates.sort(
            key=lambda item: (
                item["name"] != "complete-task-loop",
                -int(item.get("relevance", 0)),
                item.get("score", 0),
                item["name"],
            )
        )

        selected: list[dict[str, str]] = []
        seen: set[str] = set()
        remaining = budget
        for record in candidates:
            key = str(record.get("tree_hash") or record["path"])
            if key in seen or remaining < 500:
                continue
            try:
                content = Path(record["path"]).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not content.strip() or len(content) > remaining:
                continue
            selected.append(
                {
                    "name": str(record["name"]),
                    "path": str(record["path"]),
                    "content": content,
                    "group": str(record.get("category") or "uncategorized"),
                }
            )
            seen.add(key)
            remaining -= len(content)
        return selected

    # -- self-authoring --------------------------------------------------------
    def author(
        self,
        name: str,
        description: str,
        body: str,
        *,
        category: str | None = None,
        root: Path | None = None,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Write a new SKILL.md of Xander's own and index it immediately.

        This is the ability to make abilities: whatever he learns the hard
        way can become a skill the quartermaster hands out next time.
        """

        slug = slugify(name)
        description = " ".join(description.split())[:2000]
        if not description:
            raise ValueError("a skill needs a description — it is what search matches on")
        if not body.strip():
            raise ValueError("a skill needs a body")
        target = (root or authored_root()) / slug
        skill_file = target / "SKILL.md"
        if skill_file.exists() and not replace:
            raise ValueError(f"skill already exists: {slug} (pass replace=True to overwrite)")
        target.mkdir(parents=True, exist_ok=True)
        document = (
            "---\n"
            f"name: {slug}\n"
            f"description: {json.dumps(description)}\n"
            + (f"category: {category}\n" if category in CATEGORY_KEYWORDS else "")
            + "author: xander\n"
            "---\n\n"
            f"{body.strip()}\n"
        )
        skill_file.write_text(document, encoding="utf-8")
        authored = target.parent
        if authored not in self.roots and not any(
            authored == existing or authored.is_relative_to(existing) for existing in self.roots
        ):
            self.roots.append(authored)
        counts = self.refresh()
        return {
            "name": slug,
            "path": str(skill_file),
            "category": category if category in CATEGORY_KEYWORDS else classify(slug, description),
            "indexed": counts["indexed"],
        }

    def record_experience(
        self,
        group: str,
        lesson: str,
        *,
        namespace: str = "default",
        root: Path | None = None,
    ) -> dict[str, Any] | None:
        """Fold one evidence-linked lesson into a reusable grouped skill hub."""

        lesson = " ".join(lesson.split())[:800]
        if not lesson:
            return None
        slug = slugify(f"xander-experience-{namespace}-{group or 'general'}")
        target_root = root or authored_root()
        skill_file = target_root / slug / "SKILL.md"
        lessons: list[str] = []
        if skill_file.exists():
            try:
                lessons = [
                    line[2:].strip()
                    for line in skill_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    if line.startswith("- ")
                ]
            except OSError:
                lessons = []
        if lesson.casefold() not in {item.casefold() for item in lessons}:
            lessons.append(lesson)
        lessons = lessons[-30:]
        body = (
            "# Evidence-linked experience hub\n\n"
            "Use these observations as prior evidence, then verify them against the current workspace.\n\n"
            + "\n".join(f"- {item}" for item in lessons)
        )
        return self.author(
            slug,
            f"Grouped evidence from the {namespace} clone's verified {group or 'general'} missions.",
            body,
            category=group if group in CATEGORY_KEYWORDS else None,
            root=target_root,
            replace=True,
        )

    def embed(self) -> dict[str, Any]:
        """Build or top up the dense index. Safe to re-run; only new cards cost."""

        from .calibers import EMBEDDER
        from .semantic import backfill

        return backfill(
            self,
            model=EMBEDDER,
            base_url=os.environ.get("OLLAMA_API_BASE", "http://127.0.0.1:11434"),
        )

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
