from pathlib import Path

from xander_agent.skills import SkillRegistry


def _skill(root: Path, slug: str, name: str, description: str, body: str) -> None:
    directory = root / slug
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_gear_is_layered_and_context_bounded_instead_of_count_capped(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _skill(root, "loop", "complete-task-loop", "workflow planning evaluation", "general workflow")
    for index in range(8):
        _skill(
            root,
            f"wasm-{index}",
            f"wasm-helper-{index}",
            f"WebAssembly game technique {index}",
            f"specialist guidance {index}",
        )
    registry = SkillRegistry(database=tmp_path / "skills.sqlite3", roots=[root])
    registry.refresh()

    gear = registry.assemble(
        "make a WebAssembly game",
        preferred_groups=("core-workflow", "code-quality"),
        context_budget=20_000,
    )

    assert gear[0]["name"] == "complete-task-loop"
    assert len(gear) > 3
    assert sum(len(item["content"]) for item in gear) <= 20_000
    assert all(item["group"] for item in gear)


def test_experience_lessons_are_grouped_deduplicated_and_reindexed(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    registry = SkillRegistry(database=tmp_path / "skills.sqlite3", roots=[root])
    registry.record_experience("web-ui", "The browser smoke test caught a broken reset.", root=root)
    registry.record_experience("web-ui", "The browser smoke test caught a broken reset.", root=root)
    registry.record_experience("web-ui", "Keyboard play also needs verification.", root=root)

    hub = root / "xander-experience-default-web-ui" / "SKILL.md"
    content = hub.read_text(encoding="utf-8")
    assert content.count("broken reset") == 1
    assert "Keyboard play" in content
    fresh = SkillRegistry(database=tmp_path / "skills.sqlite3", roots=[root])
    matching = fresh.assemble("browser broken reset", context_budget=10_000)
    experience = next(
        item for item in matching if item["name"] == "xander-experience-default-web-ui"
    )
    assert experience["group"] == "web-ui"


def test_clone_experience_hubs_stay_namespaced_until_explicitly_promoted(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    registry = SkillRegistry(database=tmp_path / "skills.sqlite3", roots=[root])
    registry.record_experience(
        "code-quality", "Alpha uses a pytest fixture teardown.", namespace="alpha", root=root
    )
    registry.record_experience(
        "code-quality", "Bravo uses a browser teardown.", namespace="bravo", root=root
    )

    alpha = registry.assemble("pytest fixture teardown", namespace="alpha", context_budget=10_000)
    bravo = registry.assemble("pytest fixture teardown", namespace="bravo", context_budget=10_000)

    assert any(item["name"] == "xander-experience-alpha-code-quality" for item in alpha)
    assert not any("bravo" in item["name"] for item in alpha)
    assert not any("alpha" in item["name"] for item in bravo)
