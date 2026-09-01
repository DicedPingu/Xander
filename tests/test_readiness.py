from pathlib import Path

from xander_agent.readiness import assess


def test_browser_wasm_preflight_reports_missing_bridge(monkeypatch, tmp_path: Path) -> None:
    present = {"cargo": "/usr/bin/cargo", "rustc": "/usr/bin/rustc", "wasm-tools": "/usr/bin/wasm-tools", "node": "/usr/bin/node", "npm": "/usr/bin/npm"}
    monkeypatch.setattr("xander_agent.readiness.shutil.which", present.get)

    readiness = assess("build a browser WebAssembly site", tmp_path)

    assert not readiness.ready
    assert "browser WASM bridge" in readiness.missing
    assert any("wasm-bindgen" in suggestion for suggestion in readiness.install_suggestions)
    assert "Do not claim completion" in readiness.blocker()


def test_container_preflight_requires_runtime(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("xander_agent.readiness.shutil.which", lambda _tool: None)

    readiness = assess("run risky third-party code in a container", tmp_path)

    assert readiness.container_requested
    assert readiness.container_runtime == ""
    assert "container runtime" in readiness.missing


def test_acceptance_commands_are_preflight_requirements(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("xander_agent.readiness.shutil.which", lambda _tool: None)

    readiness = assess("small task", tmp_path, [["wasm-bindgen", "--version"]])

    assert "acceptance command: wasm-bindgen" in readiness.missing


def test_preflight_without_a_probe_keeps_the_old_behaviour(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("xander_agent.readiness.shutil.which", lambda tool: f"/usr/bin/{tool}")

    readiness = assess("small task", tmp_path)

    assert readiness.ready
    assert readiness.as_dict()["brain"] == {}


def test_an_unreachable_backend_blocks_the_mission(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("xander_agent.readiness.shutil.which", lambda tool: f"/usr/bin/{tool}")

    def dead_backend() -> dict:
        raise OSError("connection refused")

    readiness = assess("small task", tmp_path, brain=dead_backend)

    assert not readiness.ready
    assert "language model backend" in readiness.missing
    assert "ollama serve" in readiness.install_suggestions
    assert "connection refused" in readiness.brain.note


def test_a_routed_model_that_is_not_on_disk_blocks_the_mission(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("xander_agent.readiness.shutil.which", lambda tool: f"/usr/bin/{tool}")
    report = {
        "available": True,
        "routing": {"coder": "huihui_ai/absent-abliterated:7b"},
        "installed_models": [],
    }

    readiness = assess("small task", tmp_path, brain=report)

    assert not readiness.ready
    assert readiness.brain.unusable_roles == {"coder": "huihui_ai/absent-abliterated:7b"}
    assert "ollama pull huihui_ai/absent-abliterated:7b" in readiness.install_suggestions


def test_a_better_installed_build_satisfies_a_stale_route(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("xander_agent.readiness.shutil.which", lambda tool: f"/usr/bin/{tool}")
    report = {
        "available": True,
        "routing": {"classifier": "huihui_ai/qwen2.5-vl-abliterated:3b"},
        "installed_models": ["huihui_ai/qwen2.5-vl-abliterated:3b-instruct-q8_0"],
    }

    readiness = assess("small task", tmp_path, brain=report)

    assert readiness.ready
    assert readiness.brain.unusable_roles == {}


def test_cloud_routed_roles_do_not_need_a_local_build(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("xander_agent.readiness.shutil.which", lambda tool: f"/usr/bin/{tool}")
    report = {
        "available": True,
        "cloud_roles": ["critic"],
        "local": {
            "available": True,
            "routing": {"critic": "anthropic/claude-opus-5", "coder": "huihui_ai/qwen2.5-coder-abliterate:7b"},
            "installed_models": ["huihui_ai/qwen2.5-coder-abliterate:7b"],
        },
    }

    readiness = assess("small task", tmp_path, brain=report)

    assert readiness.ready
    assert readiness.brain.reachable
