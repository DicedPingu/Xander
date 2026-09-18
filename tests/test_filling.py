"""The plan decides what the files are; the coder decides what is in them."""

from pathlib import Path
from typing import cast

import pytest

from xander_agent.engine import Engine, _substantive
from xander_agent.backend import Backend
from xander_agent.executor import ActionExecutor
from xander_agent.models import Action, ActionKind, EditBlock, ModelPlan, TaskRecord, XanderRequest
from xander_agent.policy import snapshot_workspace


class _Coder:
    def __init__(self, body: str = "int main(){return 0;}\n") -> None:
        self.body = body
        self.calls: list[str] = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return self.body


class _EditCoder:
    def __init__(self, edits: list[EditBlock]) -> None:
        self.edits = edits
        self.calls: list[str] = []

    def generate_model(self, prompt, model_type, **kwargs):
        self.calls.append(prompt)
        return model_type(edits=self.edits)


def _engine(tmp_path: Path, coder: _Coder | _EditCoder) -> Engine:
    engine = Engine.__new__(Engine)
    engine.workspace = tmp_path.resolve()
    engine.backend = cast(Backend, coder)
    engine._record_model_stats = lambda *a, **k: None
    return engine


def _task(tmp_path: Path) -> TaskRecord:
    request = XanderRequest(mode="implement", workspace=tmp_path, goal="build a game")
    return TaskRecord(id="t1", request=request)


def test_include_lines_are_code_not_comments() -> None:
    # Counting `#include` as a comment made C++ files look empty, which is how
    # a stub slipped through twice.
    assert _substantive("#include <x>\nint main(){}\n") == ["#include <x>", "int main(){}"]
    assert _substantive("# a python comment\nx = 1\n") == ["x = 1"]
    assert _substantive("// just a note\n") == []


@pytest.mark.parametrize(
    "content",
    ["", "   \n", "// C++ code goes here\n", "int main(){\n // Game logic here\n}\n", "# TODO: implement\n"],
)
def test_an_empty_or_stubbed_file_is_written_by_the_coder(tmp_path: Path, content: str) -> None:
    coder = _Coder()
    engine = _engine(tmp_path, coder)
    action = Action(kind=ActionKind.CREATE, path="main.cpp", content=content, expected="a working game")

    note = engine._fill_content(_task(tmp_path), action)
    assert note and "main.cpp" in note
    assert action.content == coder.body
    assert len(coder.calls) == 1


def test_content_the_planner_meant_is_never_discarded(tmp_path: Path) -> None:
    """A deliberately short file is a decision, not a stub."""

    coder = _Coder()
    engine = _engine(tmp_path, coder)
    action = Action(kind=ActionKind.CREATE, path="marker.txt", content="b", expected="a marker")

    assert engine._fill_content(_task(tmp_path), action) == ""
    assert action.content == "b"
    assert coder.calls == []


def test_repeat_create_with_the_same_body_is_a_safe_noop(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("ready\n", encoding="utf-8")
    engine = _engine(tmp_path, _Coder())
    task = _task(tmp_path)
    action = Action(kind=ActionKind.CREATE, path="marker.txt", content="ready\n", expected="the marker is ready")

    note = engine._fill_content(task, action)
    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert "already contains" in note
    assert action.kind == ActionKind.NOTE
    assert result.status.value == "ok"
    assert (tmp_path / "marker.txt").read_text(encoding="utf-8") == "ready\n"


def test_repeat_create_with_new_body_becomes_a_safe_patch(tmp_path: Path) -> None:
    (tmp_path / "marker.txt").write_text("old\n", encoding="utf-8")
    engine = _engine(tmp_path, _Coder())
    task = _task(tmp_path)
    action = Action(kind=ActionKind.CREATE, path="marker.txt", content="new\n", expected="the marker is new")

    note = engine._fill_content(task, action)
    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert "safe patch" in note
    assert action.kind == ActionKind.PATCH
    assert result.status.value == "ok"
    assert (tmp_path / "marker.txt").read_text(encoding="utf-8") == "new\n"


def test_a_markdown_fence_is_stripped(tmp_path: Path) -> None:
    coder = _Coder("```cpp\nint main(){}\n```")
    engine = _engine(tmp_path, coder)
    action = Action(kind=ActionKind.CREATE, path="main.cpp", content="", expected="x")

    engine._fill_content(_task(tmp_path), action)
    assert action.content.strip() == "int main(){}"
    assert "```" not in action.content


def test_the_coder_is_shown_the_file_it_is_fixing(tmp_path: Path) -> None:
    (tmp_path / "main.cpp").write_text("int main(){ all_of(); }\n", encoding="utf-8")
    coder = _Coder()
    engine = _engine(tmp_path, coder)
    action = Action(kind=ActionKind.CREATE, path="main.cpp", content="", expected="fix the build")

    engine._fill_content(_task(tmp_path), action)
    assert "all_of()" in coder.calls[0]
    assert action.kind == ActionKind.PATCH
    assert action.patch.startswith("--- a/main.cpp")


def test_repaired_existing_file_is_applied_as_a_safe_patch(tmp_path: Path) -> None:
    (tmp_path / "main.cpp").write_text("int main(){ all_of(); }\n", encoding="utf-8")
    engine = _engine(tmp_path, _Coder("int main(){ return 0; }\n"))
    task = _task(tmp_path)
    task.plan = ModelPlan(
        summary="fix the build",
        actions=[
            Action(
                id="p1",
                kind=ActionKind.PATCH,
                path="main.cpp",
                expected="replace the broken call",
            )
        ],
    )

    assert engine._repair_plan(task) == [("main.cpp", "rewrite")]
    action = task.plan.actions[0]
    note = engine._fill_content(task, action)
    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert "safe patch" in note
    assert action.kind == ActionKind.PATCH
    assert result.status.value == "ok"
    assert (tmp_path / "main.cpp").read_text(encoding="utf-8") == "int main(){ return 0; }\n"


def test_an_unexpressable_patch_becomes_a_rewrite(tmp_path: Path) -> None:
    """The 8B planner knows what to change and cannot emit a valid diff for it.
    Rejecting the plan throws away a correct diagnosis over formatting."""

    (tmp_path / "main.cpp").write_text("int main(){}\n", encoding="utf-8")
    engine = _engine(tmp_path, _Coder())
    task = _task(tmp_path)
    task.plan = ModelPlan(
        summary="fix the build",
        actions=[Action(id="p1", kind=ActionKind.PATCH, path="main.cpp", patch="", expected="add the include")],
    )

    assert engine._repair_plan(task) == [("main.cpp", "rewrite")]
    action = task.plan.actions[0]
    assert action.kind == ActionKind.CREATE and action.content == ""


def test_a_real_diff_and_a_missing_file_are_left_alone(tmp_path: Path) -> None:
    engine = _engine(tmp_path, _Coder())
    task = _task(tmp_path)
    (tmp_path / "real.py").write_text("x = 1\n", encoding="utf-8")
    task.plan = ModelPlan(
        summary="two patches",
        actions=[
            Action(id="good", kind=ActionKind.PATCH, path="real.py",
                   patch="--- a/real.py\n+++ b/real.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"),
            Action(id="absent", kind=ActionKind.PATCH, path="nowhere.py", patch=""),
        ],
    )

    assert engine._repair_plan(task) == []
    assert task.plan.actions[0].kind == ActionKind.PATCH
    assert task.plan.actions[1].kind == ActionKind.PATCH


def test_it_cannot_be_tricked_into_reading_outside_the_workspace(tmp_path: Path) -> None:
    engine = _engine(tmp_path, _Coder())
    task = _task(tmp_path)
    task.plan = ModelPlan(
        summary="escape",
        actions=[Action(id="bad", kind=ActionKind.PATCH, path="../../../etc/passwd", patch="")],
    )
    assert engine._repair_plan(task) == []
    assert task.plan.actions[0].kind == ActionKind.PATCH


def test_an_unexpressable_patch_on_a_large_file_becomes_an_edit(tmp_path: Path) -> None:
    """A body-less patch used to be left broken ("no unified patch body") when
    the target was too large to rewrite whole. It becomes an edit request
    instead, so the coder only has to locate and change one region."""

    big = "line\n" * 3000  # well over the 6_000-byte rewrite ceiling
    (tmp_path / "big.py").write_text(big, encoding="utf-8")
    engine = _engine(tmp_path, _Coder())
    task = _task(tmp_path)
    task.plan = ModelPlan(
        summary="fix one line",
        actions=[Action(id="p1", kind=ActionKind.PATCH, path="big.py", patch="", expected="change one line")],
    )

    assert engine._repair_plan(task) == [("big.py", "edit")]
    action = task.plan.actions[0]
    assert action.kind == ActionKind.EDIT
    assert action.edits == []


def test_fill_edit_writes_a_verified_search_replace_block(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    coder = _EditCoder([EditBlock(search="b = 2", replace="b = 20")])
    engine = _engine(tmp_path, coder)
    task = _task(tmp_path)
    action = Action(id="e1", kind=ActionKind.EDIT, path="big.py", expected="update b")

    note = engine._fill_edit(task, action)
    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert "1 edit block" in note
    assert result.status.value == "ok"
    assert (tmp_path / "big.py").read_text(encoding="utf-8") == "a = 1\nb = 20\nc = 3\n"
    assert coder.calls and "big.py" in coder.calls[0]


def test_fill_edit_rejects_a_search_that_does_not_match(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("a = 1\n", encoding="utf-8")
    coder = _EditCoder([EditBlock(search="nowhere", replace="x")])
    engine = _engine(tmp_path, coder)
    action = Action(id="e1", kind=ActionKind.EDIT, path="big.py", expected="update b")

    note = engine._fill_edit(_task(tmp_path), action)

    assert note.startswith("blocked:")
    assert action.edits == []
    assert (tmp_path / "big.py").read_text(encoding="utf-8") == "a = 1\n"


def test_fill_edit_leaves_planner_supplied_edits_alone(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("a = 1\n", encoding="utf-8")
    coder = _EditCoder([EditBlock(search="should not be called", replace="x")])
    engine = _engine(tmp_path, coder)
    action = Action(
        id="e1",
        kind=ActionKind.EDIT,
        path="big.py",
        edits=[EditBlock(search="a = 1", replace="a = 2")],
        expected="update a",
    )

    assert engine._fill_edit(_task(tmp_path), action) == ""
    assert coder.calls == []
    assert action.edits[0].replace == "a = 2"


def test_fill_edit_preserves_crlf_and_hashes_exact_bytes(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_bytes(b"a = 1\r\nb = 2\r\n")
    coder = _EditCoder([EditBlock(search="a = 1", replace="a = 3")])
    engine = _engine(tmp_path, coder)
    action = Action(id="e1", kind=ActionKind.EDIT, path="a.py", expected="update a")

    note = engine._fill_edit(_task(tmp_path), action)
    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert "1 edit block" in note
    assert result.status.value == "ok", result.reason
    assert target.read_bytes() == b"a = 3\r\nb = 2\r\n"
