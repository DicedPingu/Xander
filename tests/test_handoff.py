from pathlib import Path

from xander_agent.handoff import FILENAME, compose, write


def test_the_note_is_hidden_and_readable(tmp_path: Path) -> None:
    path = write(tmp_path, "# hello\n")
    assert path is not None and path.name.startswith(".")
    assert path.name == FILENAME
    assert path.read_text(encoding="utf-8") == "# hello\n"
    # Readable by others: a note nobody can open defeats the purpose.
    assert oct(path.stat().st_mode)[-3:] == "644"


def test_it_cannot_be_written_outside_the_workspace(tmp_path: Path) -> None:
    assert write(tmp_path, "x", filename="../escaped.md") is None
    assert not (tmp_path.parent / "escaped.md").exists()


def test_a_placeholder_is_called_out_where_a_reader_will_see_it(tmp_path: Path) -> None:
    body = compose(
        workspace=tmp_path,
        goal="three tic tac toe games",
        status="unverified",
        files=[
            {"path": "C++/main.cpp", "lines": 80},
            {"path": "Rust/main.rs", "lines": 3, "placeholder": True},
        ],
        commands=["g++ -o ttt C++/main.cpp", "./ttt"],
        open_problems=["the C++ build fails: all_of not declared"],
        lesson="include <algorithm> for all_of",
        task_id="t-1",
    )
    assert "three tic tac toe games" in body
    assert "g++ -o ttt C++/main.cpp" in body
    assert "all_of not declared" in body
    # The whole point: a future agent must not mistake a stub for an implementation.
    assert "still a placeholder" in body
    assert "Rust/main.rs" in body
    assert "include <algorithm>" in body


def test_no_known_problems_still_warns_rather_than_reassures(tmp_path: Path) -> None:
    body = compose(workspace=tmp_path, goal="g", status="completed", files=[], commands=[], open_problems=[])
    assert "Verify before trusting that" in body


def test_backticks_cannot_break_out_of_the_markdown(tmp_path: Path) -> None:
    body = compose(
        workspace=tmp_path,
        goal="a goal with ``` a fence and `code`",
        status="ok",
        files=[{"path": "a`b.py"}],
        commands=["echo `whoami`"],
        open_problems=[],
    )
    assert "```bash" in body  # our own fence survives
    assert body.count("```") == 2  # and nothing injected another one
