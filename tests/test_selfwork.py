from pathlib import Path

import pytest

from xander_agent.selfwork import (
    Guard,
    SelfPolicy,
    SelfStore,
    changed_since,
    _hash_tree,
    improve_code,
    is_protected,
    tune_models,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A miniature Xander: something to edit, something unrelated and dirty."""

    (tmp_path / "xander_agent").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "xander_agent" / "narrator.py").write_text("original\n", encoding="utf-8")
    (tmp_path / "xander_agent" / "policy.py").write_text("guardrail\n", encoding="utf-8")
    (tmp_path / "xander_agent" / "unrelated_dirty.py").write_text("precious\n", encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the operator's real config. One of these tests wrote a fake
    model into the live default variant and broke coder routing for real."""

    for name in ("XANDER_CONFIG_DIR", "XANDER_STATE_DIR", "XANDER_CACHE_DIR"):
        monkeypatch.setenv(name, str(tmp_path / "sealed"))


@pytest.fixture
def store(tmp_path: Path) -> SelfStore:
    return SelfStore(root=tmp_path / "config")


def test_the_guardrails_are_off_limits() -> None:
    policy = SelfPolicy()
    assert is_protected("xander_agent/policy.py", policy)
    assert is_protected("xander_agent/selfwork.py", policy)
    assert is_protected("tests/test_learn.py", policy)
    assert is_protected("pyproject.toml", policy)
    # Ordinary implementation is fair game.
    assert not is_protected("xander_agent/narrator.py", policy)


def test_guard_restores_only_what_it_watched(repo: Path) -> None:
    policy = SelfPolicy()
    guard = Guard(repo, policy)
    guard.watch([repo / "xander_agent" / "narrator.py"])

    (repo / "xander_agent" / "narrator.py").write_text("edited\n", encoding="utf-8")
    (repo / "xander_agent" / "unrelated_dirty.py").write_text("edited too\n", encoding="utf-8")

    guard.restore()
    assert (repo / "xander_agent" / "narrator.py").read_text() == "original\n"
    # Unrelated dirty work is not in the backup set and must be left alone.
    assert (repo / "xander_agent" / "unrelated_dirty.py").read_text() == "edited too\n"


def test_guard_deletes_files_that_did_not_exist_before(repo: Path) -> None:
    guard = Guard(repo, SelfPolicy())
    new = repo / "xander_agent" / "invented.py"
    guard.watch([new])
    new.write_text("brand new\n", encoding="utf-8")
    guard.restore()
    assert not new.exists()


def test_changed_since_sees_edits_additions_and_deletions(repo: Path) -> None:
    before = _hash_tree(repo)
    (repo / "xander_agent" / "narrator.py").write_text("edited\n", encoding="utf-8")
    (repo / "xander_agent" / "added.py").write_text("new\n", encoding="utf-8")
    (repo / "xander_agent" / "unrelated_dirty.py").unlink()

    moved = changed_since(repo, before)
    assert "xander_agent/narrator.py" in moved
    assert "xander_agent/added.py" in moved
    assert "xander_agent/unrelated_dirty.py" in moved


def _invoke_that_edits(repo: Path, relative: str, content: str = "changed\n"):
    def invoke(mode: str, **kwargs: object) -> dict:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "status": "completed", "task_id": "t1", "task": {}, "handoff": {}}

    return invoke


def test_a_change_that_passes_verification_is_kept(repo: Path, store: SelfStore) -> None:
    record = store.load()
    record.policy.verify = ["true"]  # stands in for a green suite
    store.save(record)

    change = improve_code(
        repo, "make narration clearer", store=store, invoke=_invoke_that_edits(repo, "xander_agent/narrator.py")
    )
    assert change.kept and change.verified
    assert (repo / "xander_agent" / "narrator.py").read_text() == "changed\n"


def test_a_change_that_fails_verification_is_reverted(repo: Path, store: SelfStore) -> None:
    record = store.load()
    record.policy.verify = ["false"]  # stands in for a red suite
    store.save(record)

    change = improve_code(
        repo, "a regression", store=store, invoke=_invoke_that_edits(repo, "xander_agent/narrator.py")
    )
    assert not change.kept and not change.verified
    assert "reverted" in change.detail
    # The file is byte-for-byte what it was.
    assert (repo / "xander_agent" / "narrator.py").read_text() == "original\n"


def test_editing_his_own_guardrail_is_reverted_without_even_verifying(
    repo: Path, store: SelfStore
) -> None:
    record = store.load()
    # Verification would pass, so only the protected-path rule can stop this.
    record.policy.verify = ["true"]
    store.save(record)

    change = improve_code(
        repo, "loosen the policy", store=store, invoke=_invoke_that_edits(repo, "xander_agent/policy.py")
    )
    assert not change.kept
    assert "protected" in change.detail
    assert (repo / "xander_agent" / "policy.py").read_text() == "guardrail\n"


def test_an_oversized_change_is_reverted(repo: Path, store: SelfStore) -> None:
    record = store.load()
    record.policy.verify = ["true"]
    record.policy.max_changed_files = 2
    store.save(record)

    def invoke(mode: str, **kwargs: object) -> dict:
        for index in range(5):
            (repo / "xander_agent" / f"sprawl{index}.py").write_text("x\n", encoding="utf-8")
        return {"ok": True, "status": "completed", "task_id": "t1", "task": {}, "handoff": {}}

    change = improve_code(repo, "rewrite everything", store=store, invoke=invoke)
    assert not change.kept and "limit is 2" in change.detail
    assert not (repo / "xander_agent" / "sprawl0.py").exists()


def test_propose_mode_never_applies(repo: Path, store: SelfStore) -> None:
    record = store.load()
    record.policy.code = "propose"
    store.save(record)

    change = improve_code(
        repo, "suggest something", store=store, invoke=_invoke_that_edits(repo, "xander_agent/narrator.py")
    )
    assert not change.kept and "proposal only" in change.detail


def test_self_coding_can_be_switched_off(repo: Path, store: SelfStore) -> None:
    record = store.load()
    record.policy.code = "off"
    store.save(record)
    calls: list[str] = []

    def invoke(mode: str, **kwargs: object) -> dict:
        calls.append(mode)
        return {}

    change = improve_code(repo, "anything", store=store, invoke=invoke)
    assert "off" in change.summary
    assert calls == []  # the engine is never even reached


def test_history_records_what_was_kept_and_what_was_undone(repo: Path, store: SelfStore) -> None:
    record = store.load()
    record.policy.verify = ["true"]
    store.save(record)
    improve_code(repo, "good change", store=store, invoke=_invoke_that_edits(repo, "xander_agent/a.py"))

    record = store.load()
    record.policy.verify = ["false"]
    store.save(record)
    improve_code(repo, "bad change", store=store, invoke=_invoke_that_edits(repo, "xander_agent/b.py"))

    from xander_agent.selfwork import status

    report = status(store)
    assert report["changes_kept"] == 1
    assert report["changes_reverted"] == 1
    assert [item["summary"] for item in report["recent"]] == ["good change", "bad change"]


def test_model_tuning_can_be_switched_off(tmp_path: Path, store: SelfStore) -> None:
    record = store.load()
    record.policy.models = "off"
    store.save(record)
    changes = tune_models(tmp_path, store=store)
    assert len(changes) == 1 and "off" in changes[0].summary


def test_an_invalid_policy_value_is_rejected_not_silently_stored(store: SelfStore) -> None:
    # Plain assignment does not validate a Literal, so the CLI round-trips
    # through model_validate. A typo must never leave the policy undefined.
    record = store.load()
    candidate = record.policy.model_dump(mode="json")
    candidate["code"] = "banana"
    with pytest.raises(Exception):
        SelfPolicy.model_validate(candidate)

    candidate["code"] = "propose"
    assert SelfPolicy.model_validate(candidate).code == "propose"


def test_any_is_the_default_so_he_can_fetch_what_he_is_missing() -> None:
    assert SelfPolicy().models == "any"


def _fake_backend(monkeypatch, installed: list[str], pulls: list[str]):
    """A backend that starts with `installed` and gains whatever is pulled."""

    class Fake:
        def installed_models(self):
            return list(installed)

        def pull(self, model, timeout=0):
            pulls.append(model)
            installed.append(model)
            return True, "success"

        def ask(self, *a, **k):
            return "ok"

    monkeypatch.setattr("xander_agent.backend.get_backend", lambda: Fake())
    return Fake


def test_any_fetches_a_missing_catalogue_build(tmp_path, store, monkeypatch):
    from xander_agent.calibers import DEFAULT_MODELS

    record = store.load()
    record.policy.models = "any"
    store.save(record)

    missing = DEFAULT_MODELS["coder"]
    pulls: list[str] = []
    _fake_backend(monkeypatch, [m for m in DEFAULT_MODELS.values() if m != missing], pulls)

    tune_models(tmp_path, store=store, probe=lambda model: True)
    assert missing in pulls


def test_installed_mode_never_downloads_anything(tmp_path, store, monkeypatch):
    from xander_agent.calibers import DEFAULT_MODELS

    record = store.load()
    record.policy.models = "installed"
    store.save(record)

    missing = DEFAULT_MODELS["coder"]
    pulls: list[str] = []
    _fake_backend(monkeypatch, [m for m in DEFAULT_MODELS.values() if m != missing], pulls)

    tune_models(tmp_path, store=store, probe=lambda model: True)
    assert pulls == []


def test_he_cannot_fetch_a_model_nobody_authorised(tmp_path, store, monkeypatch):
    """`any` means "the catalogue plus what you named", not the whole internet."""

    record = store.load()
    record.policy.models = "any"
    store.save(record)

    pulls: list[str] = []
    _fake_backend(monkeypatch, [], pulls)

    from xander_agent.variants import default_variant, save_variant

    profile = default_variant()
    profile.model_routing = {**profile.model_routing, "coder": "huihui_ai/something-abliterated:70b"}
    save_variant(profile, replace=True)  # writes into the sealed tmp config

    tune_models(tmp_path, store=store, probe=lambda model: True)
    assert "huihui_ai/something-abliterated:70b" not in pulls
