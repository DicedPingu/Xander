from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .calibers import best_installed, is_cloud


_WASM = re.compile(r"\b(?:wasm|webassembly|web\s+assembly|wasi|wit)\b", re.IGNORECASE)
_WEB = re.compile(r"\b(?:browser|web|site|frontend|javascript|typescript|html)\b", re.IGNORECASE)
_PYTHON = re.compile(r"\b(?:python|pytest|pydantic|textual|mcp)\b", re.IGNORECASE)
_CONTAINER = re.compile(r"\b(?:container|docker|podman|sandbox|isolat(?:e|ion)|untrusted|risky|malware)\b", re.IGNORECASE)


@dataclass(frozen=True)
class BrainReadiness:
    """Whether the calibers a mission will reason with are actually reachable."""

    reachable: bool
    routing: dict[str, str] = field(default_factory=dict)
    installed: tuple[str, ...] = ()
    unusable_roles: dict[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def ready(self) -> bool:
        return self.reachable and not self.unusable_roles

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "reachable": self.reachable,
            "routing": dict(self.routing),
            "installed": list(self.installed),
            "unusable_roles": dict(self.unusable_roles),
            "note": self.note,
        }


def read_brain(report: dict[str, Any]) -> BrainReadiness:
    """Normalise an Ollama or Hybrid ``doctor()`` payload into a verdict.

    A role counts as usable when the best installed build of its family is on
    disk, so a precision upgrade satisfies the check the routed tag would fail.
    """

    local = report.get("local") if isinstance(report.get("local"), dict) else report
    cloud_roles = set(report.get("cloud_roles") or ())
    routing = {str(k): str(v) for k, v in (local.get("routing") or {}).items()}
    installed = tuple(str(name) for name in (local.get("installed_models") or ()))
    reachable = bool(report.get("available", local.get("available", False)))

    unusable: dict[str, str] = {}
    for role, model in routing.items():
        if role in cloud_roles or is_cloud(model):
            continue
        resolved = best_installed(model, installed)
        if resolved not in installed:
            unusable[role] = model
    return BrainReadiness(reachable=reachable, routing=routing, installed=installed, unusable_roles=unusable)


def probe_brain(source: Callable[[], dict[str, Any]] | dict[str, Any] | None) -> BrainReadiness | None:
    """Read a backend doctor without ever letting a probe failure raise."""

    if source is None:
        return None
    try:
        report = source() if callable(source) else source
    except Exception as exc:  # a dead backend is a verdict, not a crash
        return BrainReadiness(reachable=False, note=f"backend probe failed: {exc}")
    if not isinstance(report, dict):
        return BrainReadiness(reachable=False, note="backend probe returned no report")
    return read_brain(report)


@dataclass(frozen=True)
class ToolNeed:
    name: str
    executables: tuple[str, ...]
    purpose: str
    install: tuple[str, ...]


@dataclass(frozen=True)
class ToolReadiness:
    query: str
    needs: tuple[ToolNeed, ...]
    available: dict[str, str]
    missing: tuple[str, ...]
    container_runtime: str = ""
    container_requested: bool = False
    brain: BrainReadiness | None = None

    @property
    def ready(self) -> bool:
        return not self.missing

    @property
    def install_suggestions(self) -> list[str]:
        missing = set(self.missing)
        suggestions: list[str] = []
        for need in self.needs:
            if need.name in missing:
                suggestions.extend(need.install)
        return list(dict.fromkeys(suggestions))

    def blocker(self) -> str:
        if not self.missing:
            return ""
        details = []
        for need in self.needs:
            if need.name in self.missing:
                details.append(f"{need.name} ({need.purpose}; choose one of: {', '.join(need.executables)})")
        message = "required mission tooling is missing: " + "; ".join(details)
        if self.install_suggestions:
            message += ". Install or enable one of: " + " | ".join(self.install_suggestions)
        return message + ". Do not claim completion until the preflight and acceptance checks pass."

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "query": self.query,
            "required": [
                {"name": need.name, "executables": list(need.executables), "purpose": need.purpose}
                for need in self.needs
            ],
            "available": dict(self.available),
            "missing": list(self.missing),
            "install_suggestions": self.install_suggestions,
            "container_runtime": self.container_runtime,
            "container_requested": self.container_requested,
            "brain": self.brain.as_dict() if self.brain else {},
        }


def _need(name: str, executables: Iterable[str], purpose: str, install: Iterable[str]) -> ToolNeed:
    return ToolNeed(name, tuple(executables), purpose, tuple(install))


def assess(
    goal: str,
    workspace: Path | None = None,
    commands: Iterable[Iterable[str]] = (),
    *,
    brain: Callable[[], dict[str, Any]] | dict[str, Any] | None = None,
) -> ToolReadiness:
    query = goal
    if workspace is not None:
        for filename in ("Cargo.toml", "package.json", "index.html", "pyproject.toml"):
            if (workspace / filename).is_file():
                query += f" {filename}"

    needs: list[ToolNeed] = []
    if _WASM.search(query):
        needs.extend(
            [
                _need("Rust WASM compiler", ("cargo", "rustc"), "compile Rust to WebAssembly", ("sudo apt-get install -y rustc cargo",)),
                _need("WASM validator", ("wasm-tools", "wasm-validate"), "validate the final binary", ("cargo install wasm-tools --locked", "sudo apt-get install -y wabt")),
            ]
        )
        if _WEB.search(query) or (workspace is not None and (workspace / "index.html").is_file()):
            needs.append(
                _need(
                    "browser WASM bridge",
                    ("wasm-bindgen", "wasm-pack", "wat2wasm"),
                    "produce browser-loadable bindings or a binary from WAT",
                    ("cargo install wasm-bindgen-cli --locked", "cargo install wasm-pack --locked", "sudo apt-get install -y wabt"),
                )
            )
            needs.append(
                _need("JavaScript runtime", ("node", "npm"), "run browser-side behavior checks", ("sudo apt-get install -y nodejs npm",))
            )
    if _PYTHON.search(query):
        needs.append(_need("Python test runner", ("python3", "pytest"), "run the Python test suite", ("uv sync --group dev",)))
    if _CONTAINER.search(query):
        needs.append(
            _need(
                "container runtime",
                ("podman", "docker"),
                "run risky or untrusted checks with containment",
                ("sudo apt-get install -y podman", "sudo apt-get install -y docker.io"),
            )
        )

    known = {executable for need in needs for executable in need.executables}
    for command in commands:
        argv = list(command)
        if not argv or Path(argv[0]).name in known:
            continue
        executable = Path(argv[0]).name
        needs.append(
            _need(
                f"acceptance command: {executable}",
                (executable,),
                "execute an operator-supplied required check",
                (f"install or enable `{executable}` before resuming",),
            )
        )

    available: dict[str, str] = {}
    missing: list[str] = []
    for need in needs:
        paths = {tool: shutil.which(tool) for tool in need.executables}
        found = next((path for path in paths.values() if path), None)
        if found:
            available.update({tool: path for tool, path in paths.items() if path})
        else:
            missing.append(need.name)

    container_runtime = next((tool for tool in ("podman", "docker") if tool in available), "")

    # The brain is part of the world too: a mission must not pass preflight when
    # the calibers it will reason with are unreachable or not on disk.
    brain_status = probe_brain(brain)
    if brain_status is not None and not brain_status.ready:
        if not brain_status.reachable:
            purpose = "serve the local calibers every mission step reasons with"
            install = ("ollama serve",)
        else:
            roles = ", ".join(sorted(brain_status.unusable_roles))
            purpose = f"provide a usable model for the {roles} role(s)"
            install = tuple(f"ollama pull {model}" for model in sorted(set(brain_status.unusable_roles.values())))
        needs.append(_need("language model backend", ("ollama",), purpose, install))
        missing.append("language model backend")

    return ToolReadiness(
        query=query,
        needs=tuple(needs),
        available=available,
        missing=tuple(missing),
        container_runtime=container_runtime,
        container_requested=bool(_CONTAINER.search(query)),
        brain=brain_status,
    )
