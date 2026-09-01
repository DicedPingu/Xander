from pathlib import Path

from xander_agent.research import Researcher


def test_web_assembly_phrase_discovers_installed_wasm_tools(
    tmp_path: Path, monkeypatch
) -> None:
    installed = {
        "cargo": "/usr/bin/cargo",
        "rustc": "/usr/bin/rustc",
        "wat2wasm": "/usr/bin/wat2wasm",
        "wasm-ld": "/usr/bin/wasm-ld",
        "rg": "/usr/bin/rg",
        "git": "/usr/bin/git",
    }
    monkeypatch.setattr("xander_agent.research.shutil.which", installed.get)

    tools = Researcher(tmp_path)._relevant_tools("make a game in Web Assembly")

    assert tools["wat2wasm"] == "/usr/bin/wat2wasm"
    assert tools["cargo"] == "/usr/bin/cargo"
    assert tools["rustc"] == "/usr/bin/rustc"
