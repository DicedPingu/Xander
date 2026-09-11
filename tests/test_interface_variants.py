import json
import zipfile
from pathlib import Path

import pytest

from xander_agent.variants import (
    VariantError,
    clone_variant,
    export_variant,
    import_variant,
    list_variants,
    load_variant,
    remove_variant,
)


def _xdg(monkeypatch, root: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(root / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))


def test_clone_export_and_import_are_independent_and_verified(tmp_path: Path, monkeypatch) -> None:
    _xdg(monkeypatch, tmp_path / "source")
    profile = clone_variant("ambusher", version="2.3.0", engine_requirement="1.0.0")
    assert profile.parent == "default"
    assert profile.memory_namespace == "ambusher"
    assert profile.version == "2.3.0"
    assert profile.engine_requirement == "==1.0.0"

    bundle = export_variant("ambusher", tmp_path / "ambusher.xanderbundle")
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema"] == "xander.variant-bundle/v1"
        assert manifest["members"]["variant.json"]

    _xdg(monkeypatch, tmp_path / "destination")
    imported = import_variant(bundle)
    assert imported == profile
    assert load_variant("ambusher").version == "2.3.0"
    assert [item.name for item in list_variants()] == ["ambusher", "default"]


def test_remove_quarantines_instead_of_deleting(tmp_path: Path, monkeypatch) -> None:
    _xdg(monkeypatch, tmp_path)
    clone_variant("temporary")
    quarantined = remove_variant("temporary")
    assert quarantined.exists()
    with pytest.raises(VariantError, match="not found"):
        load_variant("temporary")


def test_import_rejects_a_checksum_mismatch(tmp_path: Path, monkeypatch) -> None:
    _xdg(monkeypatch, tmp_path / "source")
    clone_variant("pathfinder")
    bundle = export_variant("pathfinder", tmp_path / "pathfinder.xanderbundle")
    corrupted = tmp_path / "corrupted.xanderbundle"
    with zipfile.ZipFile(bundle) as source, zipfile.ZipFile(corrupted, "w") as destination:
        for name in source.namelist():
            data = source.read(name)
            destination.writestr(name, b"{}" if name == "variant.json" else data)

    _xdg(monkeypatch, tmp_path / "destination")
    with pytest.raises(VariantError, match="checksum mismatch"):
        import_variant(corrupted)


def test_import_rejects_an_incompatible_engine(tmp_path: Path, monkeypatch) -> None:
    _xdg(monkeypatch, tmp_path / "source")
    clone_variant("future", engine_requirement=">=99")
    bundle = export_variant("future", tmp_path / "future.xanderbundle")

    _xdg(monkeypatch, tmp_path / "destination")
    with pytest.raises(VariantError, match="requires Xander"):
        import_variant(bundle)


def test_model_routing_only_accepts_abliterated_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _xdg(monkeypatch, tmp_path)
    from xander_agent.variants import VariantProfile

    from xander_agent.calibers import is_abliterated

    profile = VariantProfile(name="allowed")  # defaults are heretic builds
    assert all(is_abliterated(model) for model in profile.model_routing.values())

    with pytest.raises(ValueError, match="abliterated"):
        VariantProfile(name="blocked", model_routing={"coder": "qwen2.5-coder:7b"})


def test_army_ranks_follow_clone_lineage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _xdg(monkeypatch, tmp_path)
    from xander_agent.cli import army_payload
    from xander_agent.variants import army, clone_variant

    clone_variant("alpha")                        # captain, cloned from the leader
    clone_variant("bravo", from_name="alpha")     # sergeant
    clone_variant("charlie", from_name="bravo")   # trooper

    rows = {row["name"]: row for row in army()}
    assert rows["default"]["rank"] == "leader" and rows["default"]["depth"] == 0
    assert rows["alpha"]["rank"] == "captain" and rows["alpha"]["parent"] == "default"
    assert rows["bravo"]["rank"] == "sergeant"
    assert rows["charlie"]["rank"] == "trooper" and rows["charlie"]["depth"] == 3
    assert all(row["lessons"] == 0 for row in rows.values())
    assert [row["depth"] for row in army()] == sorted(row["depth"] for row in army())

    report = army_payload()
    assert report["leader"] == "default"
    assert report["size"] == 4
    assert all("tasks_won" in row for row in report["ranks"])
