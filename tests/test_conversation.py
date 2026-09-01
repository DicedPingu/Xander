from pathlib import Path

from xander_agent.conversation import talk


class StubBrain:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.last_stats = {"model": "critic-test", "tokens": 12}

    def available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append((prompt, kwargs))
        return "The Rust crate owns the game state."


def test_conversation_uses_the_selected_clone_and_prior_turns_without_a_task(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XANDER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("XANDER_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(
        "xander_agent.conversation.Researcher._local_context",
        lambda self: "Cargo.toml and package.json are present",
    )
    brain = StubBrain()

    result = talk(
        tmp_path,
        "default",
        "What owns the game state?",
        [{"role": "user", "content": "We are discussing the WASP project."}],
        backend=brain,
    )

    prompt, request = brain.calls[0]
    assert result["text"] == "The Rust crate owns the game state."
    assert result["stats"]["model"] == "critic-test"
    assert "SELECTED XANDER CLONE: default" in prompt
    assert "We are discussing the WASP project." in prompt
    assert "Cargo.toml and package.json are present" in prompt
    assert request["role"] == "critic"
    assert not (tmp_path / "state" / "tasks").exists()


def test_conversation_includes_nested_current_project_evidence(tmp_path: Path, monkeypatch) -> None:
    assignment = tmp_path / "assignment-001"
    assignment.mkdir()
    (assignment / "README.md").write_text(
        "# Current build\n\nThe exact validation command is `npm run validate`.\n",
        encoding="utf-8",
    )
    (assignment / "package.json").write_text(
        '{"scripts":{"validate":"npm run test && wasm-tools validate dist/game.wasm"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "xander_agent.conversation.Researcher._local_context",
        lambda self: "assignment-001/README.md",
    )
    brain = StubBrain()

    talk(tmp_path, "default", "How is this project validated?", backend=brain)

    prompt, _ = brain.calls[0]
    assert "SOURCE assignment-001/README.md" in prompt
    assert "npm run validate" in prompt
    assert "SOURCE assignment-001/package.json" in prompt
    assert "never invent one" in prompt


def test_conversation_uses_compact_model_only_for_managed_resource_gate(
    tmp_path: Path, monkeypatch
) -> None:
    class GatedBrain(StubBrain):
        def generate(self, prompt: str, **kwargs) -> str:
            raise RuntimeError("resource gate: only 1200 MiB memory available")

    compact = StubBrain()
    compact.last_stats = {"model": "compact-test", "tokens": 8}
    monkeypatch.setattr("xander_agent.conversation.backend_for", lambda models: GatedBrain())
    monkeypatch.setattr(
        "xander_agent.conversation.OllamaBackend",
        lambda **kwargs: compact,
    )

    result = talk(tmp_path, "default", "Hello Xander")

    assert result["text"] == "The Rust crate owns the game state."
    assert result["role"] == "classifier"
    assert result["stats"]["model"] == "compact-test"
    assert "smaller language model" in result["reason"]
    assert compact.calls[0][1]["role"] == "classifier"
