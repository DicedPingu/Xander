"""Task-aware provisioning — Xander wiring himself up for the job at hand.

Given a goal like "how do I use magisk properly", Xander decides which local
tools the task needs (adb, fastboot, git, …), checks which are actually present,
and assembles a *toolbelt*: the concrete CLIs he's allowed to drive plus the
"hooks" (ready-made probe commands) that get him oriented — e.g. `adb devices`,
`fastboot devices` for Android work.

This is deterministic and offline: pattern → capability. The model then plans
using only the tools that provisioning confirmed exist, so it never invents a
command the machine can't run.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Capability:
    name: str
    tools: list[str]                 # CLIs this capability wants
    hooks: list[str] = field(default_factory=list)   # orientation probes
    keywords: list[str] = field(default_factory=list)
    note: str = ""


# The capability catalog. Order matters only for display; matching is by keyword.
CATALOG: list[Capability] = [
    Capability(
        "android-device", ["adb", "fastboot"],
        hooks=["adb devices -l", "adb get-state", "fastboot devices"],
        keywords=["magisk", "adb", "fastboot", "root", "twrp", "recovery", "flash",
                  "bootloader", "android", "apk", "lineage", "kernelsu", "odin",
                  "unlock", "sideload", "ota", "gsi"],
        note="Android device / rooting work. Device must be in the right mode "
             "(adb: booted+USB-debug; fastboot: bootloader).",
    ),
    Capability(
        "git-repo", ["git", "gh"],
        hooks=["git status -sb", "git remote -v"],
        keywords=["git", "commit", "branch", "merge", "rebase", "pull request",
                  "clone", "repo", "github"],
    ),
    Capability(
        "containers", ["docker", "podman", "kubectl"],
        hooks=["docker ps", "docker images"],
        keywords=["docker", "container", "image", "compose", "kubernetes", "k8s", "pod"],
    ),
    Capability(
        "python", ["python3", "uv", "pip", "pytest"],
        hooks=["python3 --version", "uv --version"],
        keywords=["python", "pip", "venv", "uv", "pytest", "poetry", "wheel", "pypi"],
    ),
    Capability(
        "packaging", ["apt", "dpkg", "pacman", "dnf", "flatpak", "snap"],
        hooks=[],
        keywords=["install", "package", "apt", "pacman", "dnf", "dependency", "repo"],
    ),
    Capability(
        "networking", ["ip", "ss", "curl", "dig", "nmap", "ping"],
        hooks=["ip -brief addr", "ip route"],
        keywords=["network", "ip", "dns", "port", "firewall", "curl", "socket",
                  "vpn", "wifi", "nmap", "ssh", "tunnel"],
    ),
    Capability(
        "gpu-ml", ["nvidia-smi", "ollama", "python3"],
        hooks=["nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader",
               "ollama ps"],
        keywords=["cuda", "gpu", "nvidia", "torch", "tensorflow", "ollama", "llm",
                  "model", "inference", "vram"],
    ),
    Capability(
        "filesystem", ["rsync", "find", "du", "df", "tar"],
        hooks=["df -h"],
        keywords=["backup", "rsync", "disk", "mount", "partition", "sync", "archive", "tar"],
    ),
    Capability(
        "systemd", ["systemctl", "journalctl"],
        hooks=[],
        keywords=["service", "daemon", "systemd", "unit", "boot", "journal", "logs"],
    ),
]


@dataclass
class Toolbelt:
    goal: str
    capabilities: list[Capability]
    available: dict[str, str]        # tool -> path
    missing: list[str]

    @property
    def tools(self) -> list[str]:
        return sorted(self.available)

    @property
    def hooks(self) -> list[str]:
        """Orientation probes whose lead CLI is actually installed."""
        out = []
        for cap in self.capabilities:
            for h in cap.hooks:
                if h.strip() and h.split() and h.split()[0] in self.available:
                    out.append(h)
        return out

    def installation_suggestions(self) -> list[str]:
        suggestions = []
        for t in self.missing:
            if t == "git":
                if shutil.which("apt"):
                    suggestions.append("sudo apt install -y git")
                elif shutil.which("pacman"):
                    suggestions.append("sudo pacman -S git")
            elif t == "uv":
                suggestions.append("curl -LsSf https://astral.sh/uv/install.sh | sh")
            elif t in ("node", "npm", "npx"):
                if shutil.which("apt"):
                    suggestions.append("sudo apt install -y nodejs npm")
            elif "mcp" in t:
                suggestions.append(f"npx -y {t}")
        return suggestions

    def summary(self) -> str:
        caps = ", ".join(c.name for c in self.capabilities) or "general"
        missing_str = ""
        if self.missing:
            missing_str = f" · missing: {', '.join(self.missing)}"
            sugs = self.installation_suggestions()
            if sugs:
                missing_str += f" (suggested: {'; '.join(sugs)})"
        return f"{caps} · {len(self.available)} tools ready" + missing_str


def load_catalog() -> list[Capability]:
    catalog = list(CATALOG)  # Start with built-ins
    cap_dir = Path(__file__).resolve().parent / "capabilities"
    if cap_dir.exists():
        for f in cap_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                items = data if isinstance(data, list) else [data]
                for item in items:
                    catalog.append(Capability(
                        name=item["name"],
                        tools=item["tools"],
                        hooks=item.get("hooks", []),
                        keywords=item.get("keywords", []),
                        note=item.get("note", "")
                    ))
            except Exception:
                pass
    return catalog


def provision(goal: str) -> Toolbelt:
    """Match the goal to capabilities, then probe which tools exist."""
    text = goal.lower()
    matched: list[Capability] = []
    for cap in load_catalog():
        if any(kw in text for kw in cap.keywords):
            matched.append(cap)

    wanted: list[str] = []
    for cap in matched:
        for t in cap.tools:
            if t not in wanted:
                wanted.append(t)
    # A general goal still gets a baseline toolbelt so Xander isn't empty-handed.
    if not wanted:
        wanted = ["bash", "ls", "cat", "grep", "find", "curl", "git"]

    available: dict[str, str] = {}
    missing: list[str] = []
    for t in wanted:
        path = shutil.which(t)
        if path:
            available[t] = path
        else:
            missing.append(t)

    return Toolbelt(goal=goal, capabilities=matched, available=available, missing=missing)


def tags_for(goal: str) -> list[str]:
    """Tags used to file/recall learnings for this kind of task."""
    text = goal.lower()
    tags = [c.name for c in load_catalog() if any(kw in text for kw in c.keywords)]
    return tags or ["general"]
