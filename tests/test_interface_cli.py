import json
from pathlib import Path

from xander_agent.cli import main


def _xdg(monkeypatch, root: Path) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(root / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))


def test_json_mode_emits_only_jsonl(tmp_path: Path, monkeypatch, capsys) -> None:
    _xdg(monkeypatch, tmp_path)
    assert main(["doctor", "--json", "--workspace", str(tmp_path)]) in {0, 1}
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


def test_research_cli_mode_is_read_only_and_reaches_the_engine(monkeypatch, tmp_path: Path, capsys) -> None:
    _xdg(monkeypatch, tmp_path)
    captured: dict[str, object] = {}
    engine_options: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            engine_options.update(kwargs)
            self.kwargs = kwargs

        def execute(self, **kwargs):
            captured.update(kwargs)
            return {"task": {"status": "completed"}, "handoff": {"status": "completed"}}

    monkeypatch.setattr("xander_agent.cli._load_engine", lambda: FakeEngine)
    assert main(["--json", "research", "compare", "safe", "routes"]) == 0
    assert captured["mode"] == "research"
    assert engine_options["autonomy"] == "proposal-only"
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["event"] == "result"


def test_bare_xander_launches_the_tui_with_arguments_it_actually_accepts(monkeypatch, tmp_path):
    """The launch path had no test, so a renamed parameter reached users as a
    TypeError on the most common command of all: plain `xander`."""

    import inspect

    from xander_agent import cli, tui

    seen: dict = {}
    # Capture the REAL signature before patching; binding against the stand-in
    # would accept anything and the test would pass while the bug is present.
    real_signature = inspect.signature(tui.run_tui)

    def fake_run_tui(**kwargs):
        # Fail here rather than at runtime if cli passes something run_tui lost.
        real_signature.bind(**kwargs)
        seen.update(kwargs)

    monkeypatch.setattr(cli, "run_tui", fake_run_tui, raising=False)
    monkeypatch.setattr(tui, "run_tui", fake_run_tui)

    parser = cli.build_parser()
    args = parser.parse_args(["--workspace", str(tmp_path)])
    assert args.command is None

    assert cli._run_command(args, cli.Emitter()) == 0
    assert seen["workspace"] == tmp_path.resolve()
