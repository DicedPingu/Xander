from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .models import ResearchBundle
from .skills import SkillRegistry


TOOL_HINTS = {
    "python": {"python", "pyproject", "pytest", "pydantic", "textual", "mcp"},
    "uv": {"python", "uv", "pyproject", "pydantic", "textual", "mcp"},
    "pytest": {"python", "test", "pytest"},
    "cargo": {"rust", "cargo", "wasm", "wasmtime", "wit", "wasi"},
    "rustc": {"rust", "cargo", "wasm", "wasmtime", "wit", "wasi"},
    "wasm-tools": {"wasm", "component", "wit", "wasi"},
    "git": {"git", "repo", "repository", "patch", "commit", "diff"},
    "gh": {"github", "pull request", "repository", "existing solution"},
    "node": {"javascript", "typescript", "node", "npm", "react", "web"},
    "npm": {"javascript", "typescript", "node", "npm", "react", "web"},
    "flutter": {"flutter", "dart", "android"},
    "adb": {"adb", "android", "device", "apk"},
    "ollama": {"ollama", "model", "llm", "agent"},
}
DOC_LIBRARIES = {
    "textual",
    "pydantic",
    "mcp",
    "model context protocol",
    "wasmtime",
    "wasi",
    "wit-bindgen",
    "ollama",
    "flutter",
    "react",
    "fastapi",
    "django",
    "rust",
}
MANIFESTS = (
    "pyproject.toml",
    "uv.lock",
    "Cargo.toml",
    "Cargo.lock",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "pubspec.yaml",
    "go.mod",
    "AGENTS.md",
)


def _run(argv: list[str], cwd: Path, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            argv,
            124,
            stdout=_as_text(exc.stdout),
            stderr=f"timed out after {timeout}s",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, stdout="", stderr=str(exc))


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def _redact(text: str) -> str:
    text = re.sub(r"(?i)(token|password|secret|api[_-]?key)(\s*[=:]\s*)\S+", r"\1\2[REDACTED]", text)
    return text[:24_000]


class Researcher:
    def __init__(self, workspace: Path, skills: SkillRegistry | None = None) -> None:
        self.workspace = workspace.expanduser().resolve(strict=True)
        self.skills = skills or SkillRegistry()

    def gather(self, goal: str, subject: str = "") -> ResearchBundle:
        query = f"{subject} {goal}".strip()
        local = self._local_context()
        tools = self._relevant_tools(query)
        skills = [
            {key: value for key, value in record.items() if key in {"name", "description", "category", "bucket", "path", "score"}}
            for record in self.skills.search(query, limit=3)
        ]
        documentation, doc_sources, warnings = self._context7(query)
        web_digest, web_sources = self._web_search(query)
        if web_digest:
            documentation = (documentation + "\n\nWeb findings:\n" + web_digest).strip()
            doc_sources = [*doc_sources, *web_sources]
        mcp, mcp_warnings = self._mcp_inventory(query)
        warnings.extend(mcp_warnings)
        if mcp:
            local += "\n\nRelevant MCP inventory:\n" + mcp
        sources = list(doc_sources)
        github, github_sources, github_warnings = self._existing_solutions(query)
        warnings.extend(github_warnings)
        if github:
            local += "\n\nExisting-solutions preflight:\n" + github
            sources.extend(github_sources)
        return ResearchBundle(
            local_context=local[:40_000],
            documentation=documentation[:32_000],
            skills=skills,
            tools=tools,
            sources=list(dict.fromkeys(sources)),
            warnings=warnings,
        )

    def _local_context(self) -> str:
        sections = [f"Workspace: {self.workspace}"]
        git = _run(["git", "status", "--short", "--branch", "--", "."], self.workspace)
        if git.returncode == 0:
            sections.append("Git status:\n" + _redact(git.stdout[:12_000]))
        files = _run(["rg", "--files"], self.workspace)
        if files.returncode == 0:
            names = files.stdout.splitlines()
            sections.append("Repository files (bounded):\n" + "\n".join(names[:500]))
            if len(names) > 500:
                sections.append(f"... {len(names) - 500} additional files omitted")
        for name in MANIFESTS:
            path = self.workspace / name
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            sections.append(f"{name} (bounded):\n{_redact(content[:12_000])}")
        return "\n\n".join(sections)

    def _relevant_tools(self, query: str) -> dict[str, str]:
        lowered = query.lower()
        relevant: dict[str, str] = {}
        for tool, hints in TOOL_HINTS.items():
            if not any(hint in lowered for hint in hints):
                continue
            executable = shutil.which(tool)
            if executable:
                relevant[tool] = executable
        for baseline in ("rg", "git"):
            executable = shutil.which(baseline)
            if executable:
                relevant.setdefault(baseline, executable)
        return dict(list(relevant.items())[:12])

    def _context7(self, query: str) -> tuple[str, list[str], list[str]]:
        executable = shutil.which("ctx7")
        if not executable:
            return "", [], ["Context7 CLI is unavailable; used local truth only"]
        lowered = query.lower()
        candidates = [library for library in DOC_LIBRARIES if library in lowered][:2]
        if not candidates:
            return "", [], []
        sections: list[str] = []
        sources: list[str] = []
        warnings: list[str] = []
        for library in candidates:
            resolve = _run([executable, "library", library, f"{library} current API setup"], self.workspace, timeout=30)
            if resolve.returncode != 0:
                warnings.append(f"Context7 could not resolve {library}")
                continue
            library_id = self._library_id(resolve.stdout)
            if not library_id:
                warnings.append(f"Context7 returned no library id for {library}")
                continue
            docs = _run([executable, "docs", library_id, f"current setup API testing guidance for {library}"], self.workspace, timeout=45)
            if docs.returncode == 0 and docs.stdout.strip():
                sections.append(f"Context7 {library_id}:\n{docs.stdout[:16_000]}")
                sources.append(f"context7:{library_id}")
            else:
                warnings.append(f"Context7 docs failed for {library_id}")
        return "\n\n".join(sections), sources, warnings

    def _web_search(self, query: str) -> tuple[str, list[str]]:
        """Bounded online lookup; set XANDER_OFFLINE=1 to keep him home."""

        if os.environ.get("XANDER_OFFLINE"):
            return "", []
        try:
            from . import web

            rows = web.search(query)
        except Exception:
            return "", []
        if not rows:
            return "", []
        return web.digest(rows), [row["url"] for row in rows if row.get("url")][:6]

    @staticmethod
    def _library_id(text: str) -> str:
        match = re.search(r"(?m)(/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", text)
        return match.group(1) if match else ""

    def _mcp_inventory(self, query: str) -> tuple[str, list[str]]:
        keywords = set(re.findall(r"[a-z0-9-]+", query.lower()))
        matches: list[str] = []
        warnings: list[str] = []
        probes: list[tuple[str, list[str]]] = []
        for executable, args in (("codex", ["mcp", "list"]), ("claude", ["mcp", "list"])):
            path = shutil.which(executable)
            if not path:
                continue
            probes.append((executable, [path, *args]))

        def probe(item: tuple[str, list[str]]) -> tuple[str, subprocess.CompletedProcess[str]]:
            executable, argv = item
            return executable, _run(argv, self.workspace, timeout=5)

        with ThreadPoolExecutor(max_workers=max(1, len(probes)), thread_name_prefix="xander-mcp") as pool:
            results = list(pool.map(probe, probes)) if probes else []
        for executable, run in results:
            if run.returncode == 124:
                warnings.append(f"{executable} MCP inventory timed out; continued without it")
                continue
            if run.returncode != 0:
                continue
            for line in run.stdout.splitlines():
                if any(keyword in line.lower() for keyword in keywords if len(keyword) > 3):
                    matches.append(f"{executable}: {_redact(line)}")
        return "\n".join(matches[:12]), warnings

    def _existing_solutions(self, query: str) -> tuple[str, list[str], list[str]]:
        if not shutil.which("gh"):
            return "", [], []
        auth = _run(["gh", "auth", "status"], self.workspace, timeout=10)
        if auth.returncode == 124:
            return "", [], ["GitHub preflight timed out; continued with local truth"]
        if auth.returncode != 0:
            return "", [], []
        terms = [word for word in re.findall(r"[a-z0-9-]+", query.lower()) if len(word) > 3]
        public_query = " ".join(terms[:5])
        if not public_query:
            return "", [], []
        run = _run(
            ["gh", "search", "repos", public_query, "--limit", "3", "--sort", "updated", "--json", "fullName,description,updatedAt,url,isArchived"],
            self.workspace,
            timeout=25,
        )
        if run.returncode == 124:
            return "", [], ["GitHub repository search timed out; continued with local truth"]
        if run.returncode != 0:
            return "", [], []
        try:
            repos = [repo for repo in json.loads(run.stdout) if not repo.get("isArchived")]
        except Exception:
            return "", [], []
        lines = []
        sources = []
        for repo in repos[:3]:
            lines.append(f"- {repo.get('fullName')}: {repo.get('description') or ''} ({str(repo.get('updatedAt', ''))[:10]})")
            if repo.get("url"):
                sources.append(repo["url"])
        return "\n".join(lines), sources, []
