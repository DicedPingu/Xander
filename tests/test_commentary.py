"""The voice: personality with bounds — offline templates, caps, and fatigue."""

from __future__ import annotations

from pathlib import Path

from xander_agent.commentary import Commentator
from xander_agent.memory import MemoryStore


def test_offline_templates_speak_without_a_backend() -> None:
    voice = Commentator()
    line = voice.say("kickoff", goal="tic-tac-toe site under 30 KB")
    assert "tic-tac-toe site under 30 KB" in line
    assert line.startswith("On it")


def test_kickoff_announces_the_form_taken() -> None:
    voice = Commentator(speaker="Xander")
    line = voice.say("kickoff", goal="fix the failing tests", form="Medic")
    assert line.startswith("Taking Medic form")


def test_approach_states_the_default_with_override_phrasing() -> None:
    voice = Commentator()
    line = voice.say(
        "approach",
        summary="paint the landing page",
        default="a green background",
        actions=3,
        checks=1,
    )
    assert "a green background, unless told otherwise" in line


def test_fatigue_builds_over_repeated_setbacks_and_resets_on_victory() -> None:
    voice = Commentator()
    first = voice.say("setback", reason="tests failed")
    second = voice.say("setback", reason="tests failed")
    third = voice.say("setback", reason="tests failed")
    assert "didn't hold" in first
    assert "Same wall again" in second
    assert "tired" in third and "3 times" in third
    voice.say("victory", checks=2, paths=1)
    assert voice.fatigue == 0
    assert "didn't hold" in voice.say("setback", reason="new problem")


def test_progress_report_addresses_the_master() -> None:
    voice = Commentator()
    line = voice.say(
        "progress",
        master="Master",
        attempt=2,
        actions_ok=3,
        actions_total=5,
        checks_passed=0,
        checks_total=2,
        state="patch applied, tests next",
    )
    assert line.startswith("Master — attempt 2: 3/5")
    assert "patch applied, tests next" in line


def test_line_cap_silences_chatter_but_not_setbacks_or_victories() -> None:
    voice = Commentator(voice="quiet")  # cap of 4
    for _ in range(4):
        assert voice.say("action", expected="poke a file")
    assert voice.say("action", expected="one poke too many") == ""
    assert voice.say("setback", reason="still audible") != ""
    assert voice.say("victory", checks=1) != ""


def test_voice_off_says_nothing() -> None:
    voice = Commentator(voice="off")
    assert voice.say("kickoff", goal="anything") == ""
    assert voice.say("victory", checks=3) == ""


def test_feedback_ack_echoes_the_preference() -> None:
    voice = Commentator()
    line = voice.say("feedback_ack", text="less consultant filler")
    assert "less consultant filler" in line


class ExplodingBackend:
    def available(self) -> bool:
        return True

    def generate(self, prompt, **kwargs) -> str:
        raise RuntimeError("model melted")


def test_backend_failure_falls_back_to_templates(tmp_path: Path) -> None:
    memory = MemoryStore(path=tmp_path / "memory.json")
    voice = Commentator(backend=ExplodingBackend(), memory=memory)
    line = voice.say("kickoff", goal="survive a melting model")
    assert "survive a melting model" in line
