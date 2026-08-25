from pathlib import Path

from xander_agent import ABILITIES
from xander_agent.narrator import PHASE_ABILITY, Narrator, abilities_line


def make_narrator(tmp_path: Path) -> Narrator:
    return Narrator(variant="testling", log_dir=tmp_path)


def test_every_ability_is_exercised_somewhere_in_the_mantra() -> None:
    assert len(ABILITIES) == 7
    for core in ("llm", "agent", "algorithm", "bot"):
        assert core in ABILITIES
    assert set(PHASE_ABILITY.values()) == set(ABILITIES)
    assert abilities_line().count("·") == len(ABILITIES) - 1


def test_phase_event_narrates_with_glyph_ability_and_message(tmp_path: Path) -> None:
    channel, line = make_narrator(tmp_path).narrate(
        {"type": "phase", "phase": "research", "message": "local truth first"}
    )
    assert channel == "run"
    assert "»" in line
    assert "research" in line
    assert "(oracle)" in line
    assert "local truth first" in line


def test_events_route_to_their_specialty_channels(tmp_path: Path) -> None:
    narrator = make_narrator(tmp_path)
    assert narrator.narrate({"type": "test", "message": "x"})[0] == "tests"
    assert narrator.narrate({"type": "patch", "message": "x"})[0] == "diff"
    assert narrator.narrate({"type": "plan", "message": "x"})[0] == "plan"
    assert narrator.narrate({"type": "research", "message": "x"})[0] == "research"
    assert narrator.narrate({"type": "action", "message": "x"})[0] == "run"


def test_digest_keeps_only_the_readable_fields(tmp_path: Path) -> None:
    narrator = make_narrator(tmp_path)
    _, line = narrator.narrate(
        {
            "type": "test",
            "phase": "test",
            "message": "ok",
            "data": {"name": "pytest -q", "result": {"returncode": 0, "status": "ok"}},
        }
    )
    assert "check=pytest -q" in line
    assert "rc=0" in line
    _, plan_line = narrator.narrate(
        {"type": "plan", "message": "two-step fix", "data": {"actions": 2, "checks": 1, "options": []}}
    )
    assert "actions=2" in plan_line and "checks=1" in plan_line


def test_untrusted_message_content_is_markup_escaped(tmp_path: Path) -> None:
    _, line = make_narrator(tmp_path).narrate({"type": "action", "message": "[red]sneaky[/red]"})
    assert "\\[red]" in line


def test_plain_twin_lines_land_in_central_and_variant_logs(tmp_path: Path) -> None:
    narrator = make_narrator(tmp_path)
    narrator.narrate({"type": "phase", "phase": "work", "message": "editing one file"})
    central = (tmp_path / "xander.log").read_text(encoding="utf-8")
    personal = (tmp_path / "testling.log").read_text(encoding="utf-8")
    for content in (central, personal):
        assert "[PHASE] work:editing one file" in content
        assert "[dim]" not in content
        assert "[testling]" in content


def test_summarize_reports_verdict_evidence_and_lesson(tmp_path: Path) -> None:
    narrator = make_narrator(tmp_path)
    lines = narrator.summarize(
        {
            "ok": True,
            "status": "completed",
            "task_id": "t-123",
            "task": {
                "attempt": 1,
                "results": [{"changed_paths": ["src/a.py", "src/b.py"]}],
                "check_results": [{"status": "ok"}, {"status": "ok"}],
                "lesson": "narrow checks caught the regression",
            },
        }
    )
    text = "\n".join(lines)
    assert "✓" in lines[0] and "t-123" in lines[0]
    assert "changed 2 path(s)" in text
    assert "2/2 passed" in text
    assert "lesson" in text
    assert "xander tasks show t-123" in text


def test_summarize_surfaces_the_blocker_on_failure(tmp_path: Path) -> None:
    lines = make_narrator(tmp_path).summarize(
        {"ok": False, "status": "failed", "task_id": "t-9", "task": {"attempt": 3, "failure": "required checks failed: build"}}
    )
    text = "\n".join(lines)
    assert "✗" in lines[0]
    assert "blocked:" in text and "required checks failed" in text


def test_control_characters_cannot_forge_log_lines(tmp_path: Path) -> None:
    narrator = make_narrator(tmp_path)
    _, line = narrator.narrate(
        {
            "type": "approval",
            "message": "pick one",
            "data": {"options": ["alpha\ninjected", "beta\x1b[31m"], "selected": []},
        }
    )
    assert "\n" not in line and "\x1b" not in line
    content = (tmp_path / "xander.log").read_text(encoding="utf-8")
    assert "\x1b" not in content
    assert all(entry.startswith("[") for entry in content.splitlines())
