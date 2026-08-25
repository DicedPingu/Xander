"""Research — Xander finding the latest before he acts.

Primary source is GitHub via the authenticated `gh` CLI (works offline-of-web
but online-to-GitHub, and it's already logged in). Xander searches repos and
code for the task's subject, pulls the most relevant project descriptions and
READMEs, and hands that fresh context to the planner so the plan reflects how
people actually do the thing today — not stale training data.

Web search is exposed as a pluggable hook (`web_search`) that degrades to a
no-op when no engine is configured, so Xander never hangs on the network.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


def _gh(args: list[str], timeout: int = 25) -> str:
    if not shutil.which("gh"):
        return ""
    try:
        r = subprocess.run(["gh", *args], capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def github_repos(query: str, limit: int = 6) -> list[dict]:
    out = _gh([
        "search", "repos", query, "--limit", str(limit), "--sort", "stars",
        "--json", "fullName,description,stargazersCount,updatedAt,url",
    ])
    try:
        return json.loads(out) if out.strip() else []
    except Exception:
        return []


def github_code(query: str, limit: int = 4) -> list[dict]:
    out = _gh([
        "search", "code", query, "--limit", str(limit),
        "--json", "repository,path,url",
    ])
    try:
        return json.loads(out) if out.strip() else []
    except Exception:
        return []


def readme(full_name: str, max_chars: int = 2500) -> str:
    """Fetch a repo README via the GitHub API (base64 in JSON)."""
    out = _gh(["api", f"repos/{full_name}/readme", "--jq", ".content"])
    if not out.strip():
        return ""
    import base64
    try:
        text = base64.b64decode(out.strip()).decode("utf-8", "replace")
        return text[:max_chars]
    except Exception:
        return ""


def find_mcp_servers(subject: str) -> list[dict]:
    paths = [
        Path.home() / ".config" / "Code" / "User" / "mcp.json",
        Path.home() / ".config" / "Code - Insiders" / "User" / "mcp.json",
        Path.home() / ".config" / "Claude" / "claude_desktop_config.json",
    ]
    matched = []
    subject_lower = subject.lower()
    for p in paths:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                servers = data.get("servers", {})
                for name, cfg in servers.items():
                    cfg_str = json.dumps(cfg).lower()
                    if subject_lower in name.lower() or subject_lower in cfg_str:
                        matched.append({
                            "name": name,
                            "type": cfg.get("type", "stdio"),
                            "command": cfg.get("command", ""),
                            "args": cfg.get("args", []),
                            "url": cfg.get("url", "")
                        })
            except Exception:
                pass
    return matched


def find_local_skills_and_plugins(subject: str) -> list[dict]:
    search_paths = [
        Path.home() / ".gemini" / "config" / "plugins",
        Path.home() / ".gemini" / "antigravity-cli" / "skills",
    ]
    matched = []
    subject_lower = subject.lower()
    for base_path in search_paths:
        if not base_path.exists():
            continue
        try:
            for item in base_path.iterdir():
                if item.is_dir():
                    matches = subject_lower in item.name.lower()
                    desc = ""
                    
                    skill_files = list(item.glob("**/SKILL.md"))
                    if skill_files:
                        skill_file = skill_files[0]
                        try:
                            content = skill_file.read_text(encoding="utf-8", errors="ignore")
                            m = re.search(r"description:\s*[\"']?(.*?)[\"']?\n", content)
                            if m:
                                desc = m.group(1).strip()
                            if subject_lower in content.lower():
                                matches = True
                        except Exception:
                            pass
                    
                    plugin_json = item / "plugin.json"
                    if plugin_json.exists():
                        try:
                            pdata = json.loads(plugin_json.read_text(encoding="utf-8"))
                            desc = pdata.get("description", desc)
                            if subject_lower in pdata.get("name", "").lower() or subject_lower in desc.lower():
                                matches = True
                        except Exception:
                            pass
                    
                    if matches:
                        matched.append({
                            "name": item.name,
                            "path": str(item),
                            "description": desc or "No description available.",
                            "type": "plugin" if "plugins" in str(base_path) else "skill"
                        })
        except Exception:
            pass
    return matched


def find_cheat_sheets(subject: str) -> list[dict]:
    matched = []
    workspace_root = Path(__file__).resolve().parent.parent
    try:
        import re
        words = [w.strip() for w in re.split(r"\W+", subject.lower()) if w.strip()]
        stop_words = {"setup", "project", "build", "create", "initialize", "modular", "modularly"}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        if not keywords:
            keywords = [subject.lower()]
            
        for p in workspace_root.glob("*.md"):
            content = p.read_text(encoding="utf-8", errors="ignore").lower()
            filename = p.name.lower()
            if any(k in filename or k in content for k in keywords):
                matched.append({
                    "name": f"{p.name} (workspace doc/cheat sheet)",
                    "path": str(p),
                    "summary": f"Local documentation file: {p.name}"
                })
    except Exception:
        pass
    return matched


def web_search(query: str, limit: int = 5) -> list[dict]:
    """Pluggable web search. Returns [] unless an engine is wired in.

    Wire a local engine here (SearXNG, Kagi, Brave API, …) if you want Xander to
    read the open web too. Kept a no-op by default so research never blocks.
    """
    return []


def research(goal: str, subject: str, log=None) -> tuple[str, list[str]]:
    """Gather fresh context for the goal. Returns (context_block, source_urls)."""
    sources: list[str] = []
    lines: list[str] = []

    # 1. Local resources research (MCP servers, plugins/skills, cheat sheets)
    mcp_servers = find_mcp_servers(subject)
    if mcp_servers:
        lines.append("## Local MCP Servers matched")
        for s in mcp_servers:
            details = f"cmd: {s['command']} {' '.join(s['args'])}" if s['command'] else f"url: {s['url']}"
            lines.append(f"- {s['name']} ({s['type']}) — {details}")
            if s.get("url"):
                sources.append(s["url"])

    local_sp = find_local_skills_and_plugins(subject)
    if local_sp:
        lines.append("\n## Local Plugins & Skills matched")
        for sp in local_sp:
            lines.append(f"- {sp['name']} ({sp['type']}) — {sp['description']} (path: {sp['path']})")

    cheats = find_cheat_sheets(subject)
    if cheats:
        lines.append("\n## Local Cheat Sheets matched")
        for c in cheats:
            lines.append(f"- {c['name']} (path: {c['path']})")
            try:
                content = Path(c['path']).read_text(encoding="utf-8", errors="ignore")
                lines.append(f"\n### Contents of {c['name']}:\n{content}\n")
            except Exception as e:
                lines.append(f"[error reading cheat sheet: {e}]")

    # 2. GitHub projects
    repos = github_repos(subject)
    if repos:
        lines.append(("\n" if lines else "") + "## GitHub projects (most-starred, recent)")
        for r in repos:
            name = r.get("fullName", "")
            stars = r.get("stargazersCount", 0)
            desc = (r.get("description") or "").strip()
            updated = (r.get("updatedAt") or "")[:10]
            lines.append(f"- {name} ★{stars} ({updated}) — {desc}")
            if r.get("url"):
                sources.append(r["url"])
        top = repos[0].get("fullName", "")
        if top:
            rd = readme(top)
            if rd:
                lines.append(f"\n## README excerpt — {top}\n{rd}")
        if log:
            log.info("researched github", repos=len(repos), top=top)

    code = github_code(subject)
    if code:
        lines.append("\n## Code references")
        for c in code:
            repo = (c.get("repository") or {}).get("fullName", "")
            path = c.get("path", "")
            lines.append(f"- {repo}/{path}")
            if c.get("url"):
                sources.append(c["url"])

    web = web_search(subject)
    if web:
        lines.append("\n## Web")
        for item in web:
            lines.append(f"- {item.get('title', '')} — {item.get('url', '')}")
            if item.get("url"):
                sources.append(item["url"])

    return ("\n".join(lines).strip(), sources)
