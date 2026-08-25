"""Versioned Xander profiles and portable, hash-verified bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import __version__
from .paths import ensure_runtime_dirs, migration_dir, variants_dir

VARIANT_SCHEMA = "xander.variant/v1"
BUNDLE_SCHEMA = "xander.variant-bundle/v1"
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,47}$")
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_MAX_BUNDLE_BYTES = 128 * 1024 * 1024


class VariantError(ValueError):
    """A profile or bundle failed validation."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class VariantProfile(BaseModel):
    """Everything that may differ between independent Xander clones."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["xander.variant/v1"] = Field(
        default=VARIANT_SCHEMA, alias="schema", serialization_alias="schema"
    )
    name: str
    version: str = "1.0.0"
    engine_requirement: str = f"=={__version__}"
    parent: str | None = None
    created_at: str = Field(default_factory=_now)
    model_routing: dict[str, str] = Field(
        default_factory=lambda: {
            "coder": "huihui_ai/qwen2.5-coder-abliterate:7b",
            "planner": "huihui_ai/qwen3-abliterated:8b",
            "classifier": "huihui_ai/qwen2.5-vl-abliterated:3b",
            "critic": "huihui_ai/qwen3-abliterated:8b",
        }
    )
    skill_groups: list[str] = Field(default_factory=lambda: ["core-workflow", "code-quality"])
    autonomy: Literal["full-auto", "supervised", "proposal-only"] = "full-auto"
    capabilities: list[str] = Field(default_factory=list)
    theme: str = "amber"
    directives: list[str] = Field(default_factory=list)
    memory_namespace: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        if not _NAME_RE.fullmatch(value):
            raise ValueError("use 1-48 lowercase letters, digits, or hyphens; start with a letter")
        return value

    @field_validator("version")
    @classmethod
    def valid_version(cls, value: str) -> str:
        if not _VERSION_RE.fullmatch(value):
            raise ValueError("version must be semantic, for example 1.2.0")
        return value

    @field_validator("model_routing")
    @classmethod
    def only_abliterated_models(cls, value: dict[str, str]) -> dict[str, str]:
        """Operator policy: route only to abliterated model builds."""

        rejected = [model for model in value.values() if "abliterat" not in model.casefold()]
        if rejected:
            raise ValueError(
                "model routing only accepts abliterated builds; rejected: " + ", ".join(sorted(set(rejected)))
            )
        return value

    @field_validator("engine_requirement")
    @classmethod
    def valid_engine_requirement(cls, value: str) -> str:
        try:
            SpecifierSet(value)
        except InvalidSpecifier as exc:
            raise ValueError("engine requirement must be a PEP 440 version specifier") from exc
        return value

    def as_json(self) -> str:
        return self.model_dump_json(by_alias=True, indent=2) + "\n"


def default_variant() -> VariantProfile:
    return VariantProfile(name="default", memory_namespace="default")


def _profile_path(name: str) -> Path:
    if not _NAME_RE.fullmatch(name):
        raise VariantError(f"invalid variant name: {name!r}")
    return variants_dir() / f"{name}.json"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            pass


def save_variant(profile: VariantProfile, *, replace: bool = False) -> Path:
    ensure_runtime_dirs()
    path = _profile_path(profile.name)
    if path.exists() and not replace:
        raise VariantError(f"variant already exists: {profile.name}")
    _atomic_write(path, profile.as_json())
    return path


def load_variant(name: str) -> VariantProfile:
    path = _profile_path(name)
    if name == "default" and not path.exists():
        return default_variant()
    try:
        return VariantProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VariantError(f"variant not found: {name}") from exc
    except Exception as exc:
        raise VariantError(f"invalid variant {name}: {exc}") from exc


def list_variants() -> list[VariantProfile]:
    ensure_runtime_dirs()
    found: dict[str, VariantProfile] = {"default": load_variant("default")}
    for path in sorted(variants_dir().glob("*.json")):
        profile = load_variant(path.stem)
        found[profile.name] = profile
    return [found[name] for name in sorted(found)]


def clone_variant(
    name: str,
    *,
    from_name: str = "default",
    version: str | None = None,
    engine_requirement: str | None = None,
    replace: bool = False,
) -> VariantProfile:
    source = load_variant(from_name)
    if engine_requirement and _VERSION_RE.fullmatch(engine_requirement):
        engine_requirement = f"=={engine_requirement}"
    profile = source.model_copy(deep=True)
    profile.name = name
    profile.parent = source.name
    profile.version = version or source.version
    profile.engine_requirement = engine_requirement or source.engine_requirement
    profile.created_at = _now()
    profile.memory_namespace = name
    profile.metadata = {**source.metadata, "cloned_from_version": source.version}
    profile = VariantProfile.model_validate(profile.model_dump(by_alias=True))
    save_variant(profile, replace=replace)
    return profile


ARMY_RANKS = ("leader", "captain", "sergeant", "trooper")


def army() -> list[dict[str, Any]]:
    """The chain of command: every clone with its rank, lineage, and lessons.

    Rank follows clone depth — the root profile leads, direct clones are
    captains, their clones sergeants, everything deeper a trooper.
    """

    from .memory import lessons_count

    profiles = {profile.name: profile for profile in list_variants()}

    def depth(profile: VariantProfile) -> int:
        level, seen, current = 0, {profile.name}, profile
        while current.parent and current.parent in profiles and current.parent not in seen:
            seen.add(current.parent)
            level += 1
            current = profiles[current.parent]
        return level

    rows = []
    for profile in profiles.values():
        level = depth(profile)
        namespace = profile.memory_namespace or profile.name
        rows.append(
            {
                "name": profile.name,
                "rank": ARMY_RANKS[min(level, len(ARMY_RANKS) - 1)],
                "depth": level,
                "parent": profile.parent,
                "version": profile.version,
                "autonomy": profile.autonomy,
                "memory_namespace": namespace,
                "lessons": lessons_count(namespace),
            }
        )
    rows.sort(key=lambda row: (row["depth"], row["name"]))
    return rows


def remove_variant(name: str) -> Path:
    if name == "default":
        raise VariantError("the default variant cannot be removed")
    source = _profile_path(name)
    if not source.exists():
        raise VariantError(f"variant not found: {name}")
    destination = migration_dir() / f"removed-variant-{name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.move(source, destination)
    return destination


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _engine_members() -> dict[str, bytes]:
    root = Path(__file__).resolve().parent.parent
    lock = root / "uv.lock"
    project = root / "pyproject.toml"
    uv = shutil.which("uv")
    if uv is None:
        raise VariantError("cannot include the engine: uv is not installed")
    with tempfile.TemporaryDirectory(prefix="xander-wheel-") as temporary:
        completed = subprocess.run(
            [uv, "build", "--wheel", "--out-dir", temporary],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-1200:]
            raise VariantError(f"engine wheel build failed: {detail}")
        wheels = sorted(Path(temporary).glob("*.whl"))
        if len(wheels) != 1:
            raise VariantError("engine build did not produce exactly one wheel")
        members = {f"engine/{wheels[0].name}": wheels[0].read_bytes()}
    if project.exists():
        members["engine/pyproject.toml"] = project.read_bytes()
    if lock.exists():
        members["engine/uv.lock"] = lock.read_bytes()
    return members


def export_variant(name: str, output: Path, *, include_engine: bool = False) -> Path:
    profile = load_variant(name)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    members: dict[str, bytes] = {"variant.json": profile.as_json().encode()}
    lock = Path(__file__).resolve().parent.parent / "uv.lock"
    if lock.exists():
        members["engine/uv.lock"] = lock.read_bytes()
    if include_engine:
        members.update(_engine_members())
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "name": profile.name,
        "version": profile.version,
        "engine_requirement": profile.engine_requirement,
        "created_at": _now(),
        "include_engine": include_engine,
        "members": {member: _sha256(content) for member, content in sorted(members.items())},
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            for member, content in sorted(members.items()):
                archive.writestr(member, content)
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return output


def import_variant(bundle: Path, *, replace: bool = False) -> VariantProfile:
    bundle = bundle.expanduser().resolve()
    if bundle.stat().st_size > _MAX_BUNDLE_BYTES:
        raise VariantError("bundle exceeds the 128 MiB limit")
    try:
        with zipfile.ZipFile(bundle) as archive:
            if "manifest.json" not in archive.namelist():
                raise VariantError("bundle has no manifest.json")
            manifest = json.loads(archive.read("manifest.json"))
            if manifest.get("schema") != BUNDLE_SCHEMA:
                raise VariantError("unsupported bundle schema")
            expected = manifest.get("members")
            if not isinstance(expected, dict) or "variant.json" not in expected:
                raise VariantError("bundle manifest has no variant profile")
            declared = set(expected)
            actual = set(archive.namelist()) - {"manifest.json"}
            if declared != actual:
                raise VariantError("bundle members do not match the manifest")
            expanded_size = sum(archive.getinfo(member).file_size for member in declared)
            if expanded_size > _MAX_BUNDLE_BYTES:
                raise VariantError("expanded bundle exceeds the 128 MiB limit")
            payloads: dict[str, bytes] = {}
            total = 0
            for member, digest in expected.items():
                if member.startswith("/") or ".." in Path(member).parts:
                    raise VariantError(f"unsafe bundle member: {member}")
                data = archive.read(member)
                total += len(data)
                if total > _MAX_BUNDLE_BYTES:
                    raise VariantError("expanded bundle exceeds the 128 MiB limit")
                if _sha256(data) != digest:
                    raise VariantError(f"checksum mismatch: {member}")
                payloads[member] = data
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise VariantError(f"invalid variant bundle: {exc}") from exc
    try:
        profile = VariantProfile.model_validate_json(payloads["variant.json"])
    except Exception as exc:
        raise VariantError(f"invalid bundled variant: {exc}") from exc
    if any(
        manifest.get(field) != getattr(profile, field)
        for field in ("name", "version", "engine_requirement")
    ):
        raise VariantError("bundle manifest does not match its variant profile")
    if Version(__version__) not in SpecifierSet(profile.engine_requirement):
        raise VariantError(
            f"variant requires Xander {profile.engine_requirement}; this engine is {__version__}"
        )
    save_variant(profile, replace=replace)
    return profile
