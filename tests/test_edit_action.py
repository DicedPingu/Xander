"""The EDIT action: search/replace blocks for one region of an existing file,
without a unified diff and without rewriting the whole file. Added because a
body-less PATCH on a large file (tui.py, engine.py) had no way to apply at
all -- a 9B planner cannot reliably emit a diff for a multi-thousand-line
file, and CREATE's rewrite-whole path refuses anything over 6_000 bytes.
"""

from pathlib import Path

import pytest

from xander_agent.engine import Engine
from xander_agent.executor import ActionExecutor
from xander_agent.models import Action, ActionKind, ActionStatus, EditBlock, ModelPlan, Risk
from xander_agent.policy import apply_edit_blocks, classify_risk, changed_paths, snapshot_workspace


# --- apply_edit_blocks (the pure function shared by executor and engine) ---


def test_apply_edit_blocks_replaces_a_unique_match() -> None:
    result = apply_edit_blocks("a = 1\nb = 2\n", [EditBlock(search="a = 1", replace="a = 10")])
    assert result == "a = 10\nb = 2\n"


def test_apply_edit_blocks_applies_multiple_blocks_in_order() -> None:
    result = apply_edit_blocks(
        "a = 1\nb = 2\n",
        [EditBlock(search="a = 1", replace="a = 10"), EditBlock(search="b = 2", replace="b = 20")],
    )
    assert result == "a = 10\nb = 20\n"


def test_apply_edit_blocks_rejects_a_missing_search() -> None:
    with pytest.raises(ValueError, match="not found"):
        apply_edit_blocks("a = 1\n", [EditBlock(search="z = 9", replace="z = 90")])


def test_apply_edit_blocks_rejects_an_ambiguous_search() -> None:
    with pytest.raises(ValueError, match="matches 2 times"):
        apply_edit_blocks("x\nx\n", [EditBlock(search="x", replace="y")])


def test_apply_edit_blocks_lets_a_later_block_see_an_earlier_blocks_result() -> None:
    # Block 2 searches for text that only exists after block 1 has run.
    result = apply_edit_blocks("a\n", [EditBlock(search="a", replace="b"), EditBlock(search="b", replace="c")])
    assert result == "c\n"


# --- policy: risk and changed_paths ---


def test_edit_is_medium_risk_by_default() -> None:
    action = Action(kind=ActionKind.EDIT, path="a.py", edits=[EditBlock(search="x", replace="y")])
    assert classify_risk(action) == Risk.MEDIUM


def test_edit_changed_paths_is_the_target_file(tmp_path: Path) -> None:
    action = Action(kind=ActionKind.EDIT, path="src/a.py", edits=[EditBlock(search="x", replace="y")])
    assert changed_paths(action, tmp_path) == [(tmp_path / "src/a.py").resolve()]


# --- executor: the actual file mutation ---


def test_executor_applies_an_edit(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("greeting = 'hi'\n", encoding="utf-8")
    action = Action(
        kind=ActionKind.EDIT,
        path="a.py",
        edits=[EditBlock(search="greeting = 'hi'", replace="greeting = 'hello'")],
        expected="change the greeting",
    )

    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert result.status == ActionStatus.OK
    assert result.changed_paths == ["a.py"]
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "greeting = 'hello'\n"


def test_executor_refuses_a_search_that_does_not_match(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    action = Action(kind=ActionKind.EDIT, path="a.py", edits=[EditBlock(search="y = 2", replace="y = 3")])

    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert result.status == ActionStatus.FAILED
    assert "not found" in result.reason
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_executor_refuses_an_ambiguous_search(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    action = Action(kind=ActionKind.EDIT, path="a.py", edits=[EditBlock(search="x = 1", replace="x = 2")])

    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert result.status == ActionStatus.FAILED
    assert "ambiguous" in result.reason
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\nx = 1\n"


def test_executor_refuses_to_edit_a_missing_file(tmp_path: Path) -> None:
    action = Action(kind=ActionKind.EDIT, path="nowhere.py", edits=[EditBlock(search="x", replace="y")])

    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert result.status == ActionStatus.FAILED
    assert "does not exist" in result.reason


def test_executor_refuses_an_edit_action_with_no_blocks(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    action = Action(kind=ActionKind.EDIT, path="a.py")

    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert result.status == ActionStatus.FAILED
    assert "no search/replace" in result.reason


def test_executor_refuses_an_edit_whose_preimage_changed(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    action = Action(
        kind=ActionKind.EDIT,
        path="a.py",
        edits=[EditBlock(search="x = 1", replace="x = 2")],
        preimage_hashes={"a.py": "stale-hash-does-not-match"},
    )

    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="full-auto").run(action)

    assert result.status == ActionStatus.FAILED
    assert "preimage" in result.reason
    assert target.read_text(encoding="utf-8") == "x = 1\n"


def test_edit_action_requires_approval_under_supervised_autonomy(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    action = Action(kind=ActionKind.EDIT, path="a.py", edits=[EditBlock(search="x = 1", replace="x = 2")])

    result = ActionExecutor(tmp_path, snapshot_workspace(tmp_path), autonomy="supervised").run(action)

    assert result.status == ActionStatus.BLOCKED
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 1\n"


# --- engine: plan-lint and phantom-target validation ---


def test_lint_rejects_an_edit_action_without_a_path() -> None:
    plan = ModelPlan(summary="x", actions=[Action(id="e1", kind=ActionKind.EDIT)])
    assert "edit actions without a target path" in Engine._lint_plan(plan)


def test_lint_accepts_an_edit_action_with_a_path_and_no_edits_yet() -> None:
    # `edits` may still be empty at lint time: the coder fills it later, the
    # same way CREATE's `content` is allowed to be empty until the coder runs.
    plan = ModelPlan(
        summary="x",
        actions=[Action(id="e1", kind=ActionKind.EDIT, path="a.py", expected="change something")],
        acceptance_checks=[],
    )
    assert Engine._lint_plan(plan, require_execution=True) == ""
