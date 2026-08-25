"""Hooks stay broad until a mission narrows them; skills can make skills."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xander_agent.hooks import Hook, HookBook
from xander_agent.skills import SkillRegistry, slugify


# -- hooks ---------------------------------------------------------------------
def test_a_hook_without_a_match_is_always_relevant() -> None:
    assert Hook(constraint="never touch main").relevant("literally anything") is True


def test_a_matching_hook_arms_and_others_stay_on_the_shelf(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    path.write_text(
        json.dumps(
            [
                {"match": "css|style|background", "constraint": "keep the palette accessible"},
                {"match": "database|sql", "constraint": "never drop a table"},
                {"constraint": "preserve unrelated dirty files"},
            ]
        ),
        encoding="utf-8",
    )
    book = HookBook(path=path)
    constraints = book.constraints_for("make the background green")
    assert constraints == ["keep the palette accessible", "preserve unrelated dirty files"]
    assert "never drop a table" not in constraints


def test_notes_fire_only_at_their_moment(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    path.write_text(
        json.dumps(
            [
                {"on": "victory", "note": "tag the release"},
                {"on": "analyze", "note": "check the changelog first"},
            ]
        ),
        encoding="utf-8",
    )
    book = HookBook(path=path)
    assert book.notes_for("ship it", "analyze") == ["check the changelog first"]
    assert book.notes_for("ship it", "victory") == ["tag the release"]


def test_a_broken_regex_degrades_to_substring_matching(tmp_path: Path) -> None:
    hook = Hook(match="green(", constraint="c")
    assert hook.relevant("make it green(ish)") is True
    assert hook.relevant("make it blue") is False


def test_add_persists_and_rejects_empty_hooks(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    book = HookBook(path=path)
    book.add(on="test", match="pytest", note="run the whole suite, not one file")
    assert len(HookBook(path=path).hooks) == 1
    with pytest.raises(ValueError, match="constraint or a note"):
        book.add(match="anything")


def test_a_malformed_hook_file_is_ignored_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    path.write_text("{not json at all", encoding="utf-8")
    assert HookBook(path=path).hooks == []


# -- skills that create skills --------------------------------------------------
def test_slugify_normalizes_and_rejects_empty() -> None:
    assert slugify("Shrink a Web Bundle!") == "shrink-a-web-bundle"
    with pytest.raises(ValueError):
        slugify("!!!")


def test_author_writes_an_indexed_skill(tmp_path: Path) -> None:
    root = tmp_path / "authored"
    registry = SkillRegistry(database=tmp_path / "skills.sqlite3", roots=[])
    record = registry.author(
        "Shrink a web bundle",
        "Techniques for getting a static page under a hard KB budget",
        "# Shrink a web bundle\n\nInline everything, drop frameworks, minify.",
        root=root,
    )

    skill_file = Path(record["path"])
    assert skill_file.name == "SKILL.md"
    assert skill_file.parent.name == "shrink-a-web-bundle"
    text = skill_file.read_text(encoding="utf-8")
    assert text.startswith("---\nname: shrink-a-web-bundle\n")
    assert "author: xander" in text
    assert "Inline everything" in text

    found = registry.search("shrink bundle budget")
    assert any(item["name"] == "shrink-a-web-bundle" for item in found), (
        "an authored skill must be searchable immediately"
    )


def test_author_refuses_to_clobber_without_replace(tmp_path: Path) -> None:
    root = tmp_path / "authored"
    registry = SkillRegistry(database=tmp_path / "skills.sqlite3", roots=[])
    registry.author("dup", "first description", "body one", root=root)
    with pytest.raises(ValueError, match="already exists"):
        registry.author("dup", "second description", "body two", root=root)
    registry.author("dup", "second description", "body two", root=root, replace=True)
    assert "body two" in (root / "dup" / "SKILL.md").read_text(encoding="utf-8")


def test_author_requires_a_description_and_body(tmp_path: Path) -> None:
    registry = SkillRegistry(database=tmp_path / "skills.sqlite3", roots=[])
    with pytest.raises(ValueError, match="description"):
        registry.author("name", "   ", "body", root=tmp_path / "a")
    with pytest.raises(ValueError, match="body"):
        registry.author("name", "description", "   ", root=tmp_path / "a")
