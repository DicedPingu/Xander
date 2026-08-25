import json
from pathlib import Path

from xander_agent.cli import main


def _xdg(monkeypatch, root: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(root / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))


def test_json_mode_emits_only_jsonl(tmp_path: Path, monkeypatch, capsys) -> None:
    _xdg(monkeypatch, tmp_path)
    assert main(["doctor", "--json", "--workspace", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines
    decoded = [json.loads(line) for line in lines]
    assert all(item["schema"] == "xander.cli/v1" for item in decoded)
    assert decoded[-1]["event"] == "doctor"


def test_legacy_goal_is_routed_to_run(monkeypatch, tmp_path: Path, capsys) -> None:
    _xdg(monkeypatch, tmp_path)

    class FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def execute(self, **kwargs):
            return {
                "task": {"id": "task-1", "status": "completed"},
                "handoff": {"status": "completed"},
                "goal": kwargs["goal"],
            }

    monkeypatch.setattr("xander_agent.cli._load_engine", lambda: FakeEngine)
    assert main(["--json", "make", "a", "widget"]) == 0
    payloads = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert payloads[-1]["result"]["goal"] == "make a widget"


def test_variant_cli_accepts_trailing_json_flag(tmp_path: Path, monkeypatch, capsys) -> None:
    _xdg(monkeypatch, tmp_path)
    assert main(["variant", "clone", "critic", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["event"] == "variant.clone"
    assert payload["result"]["name"] == "critic"


def test_legacy_module_keeps_the_askar_launcher_callable() -> None:
    import xander

    assert callable(xander.main)
