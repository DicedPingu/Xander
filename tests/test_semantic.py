"""Dense retrieval must help when it can and vanish when it cannot."""

from __future__ import annotations

import sqlite3
from array import array

import pytest

from xander_agent import semantic


def test_normalize_makes_cosine_a_dot_product() -> None:
    vector = semantic.normalize([3.0, 4.0])
    assert pytest.approx(sum(value * value for value in vector), abs=1e-6) == 1.0
    assert pytest.approx(sum(a * b for a, b in zip(vector, vector)), abs=1e-6) == 1.0


def test_normalize_survives_a_zero_vector() -> None:
    # An embedder returning zeros must not raise ZeroDivisionError mid-backfill.
    assert list(semantic.normalize([0.0, 0.0])) == [0.0, 0.0]


def test_rrf_rewards_agreement_between_retrievers() -> None:
    lexical = ["a", "b", "c"]
    dense = ["c", "a", "d"]
    fused = semantic.fuse([lexical, dense])
    # "a" is 1st and 2nd; "c" is 3rd and 1st. Both beat items only one saw.
    assert fused[:2] == ["a", "c"]
    assert set(fused) == {"a", "b", "c", "d"}


def test_rrf_ignores_empty_rankings() -> None:
    assert semantic.fuse([[], ["a", "b"], []]) == ["a", "b"]


def test_rank_skips_vectors_of_a_different_dimension() -> None:
    query = semantic.normalize([1.0, 0.0])
    vectors = {
        "same": semantic.normalize([1.0, 0.0]),
        "orthogonal": semantic.normalize([0.0, 1.0]),
        "wrong-dim": semantic.normalize([1.0, 0.0, 0.0]),
    }
    ranked = semantic.rank(query, vectors)
    assert [key for key, _ in ranked] == ["same", "orthogonal"]
    assert pytest.approx(ranked[0][1], abs=1e-6) == 1.0


def test_embed_returns_none_when_the_embedder_is_unreachable() -> None:
    # A dead port, so search falls back to BM25 instead of raising.
    assert semantic.embed(["x"], model="none", base_url="http://127.0.0.1:1", timeout=2) is None


def test_embed_of_nothing_is_nothing() -> None:
    assert semantic.embed([], model="none") == []


def test_vectors_round_trip_through_sqlite(tmp_path) -> None:
    connection = sqlite3.connect(tmp_path / "v.sqlite3")
    connection.row_factory = sqlite3.Row
    semantic.ensure_schema(connection)
    original = semantic.normalize([0.1, 0.2, 0.3])
    connection.execute(
        "INSERT INTO skill_vectors(tree_hash,dim,vec) VALUES(?,?,?)",
        ("hash-1", len(original), original.tobytes()),
    )
    loaded = semantic.load_vectors(connection)
    assert list(loaded) == ["hash-1"]
    assert pytest.approx(list(loaded["hash-1"]), abs=1e-6) == list(original)


def test_load_vectors_is_empty_before_the_table_exists(tmp_path) -> None:
    connection = sqlite3.connect(tmp_path / "empty.sqlite3")
    connection.row_factory = sqlite3.Row
    assert semantic.load_vectors(connection) == {}


def test_card_text_joins_what_it_is_with_when_to_use_it() -> None:
    row = {"name": "k8s-triage", "description": "debug pods", "triggers": "CrashLoopBackOff"}
    text = semantic.card_text(row)
    assert "k8s-triage" in text and "CrashLoopBackOff" in text


def test_card_text_tolerates_missing_fields() -> None:
    assert semantic.card_text({"name": "only-name"}) == "only-name"


def test_failure_signature_collapses_the_hello_n_dot_py_loop() -> None:
    """The observed crash: one wall, dodged by renaming the file each attempt."""

    from xander_agent.engine import Engine

    first = Engine._failure_signature("create refuses to overwrite an existing path: hello2.py")
    later = Engine._failure_signature("create refuses to overwrite an existing path: hello8.py")
    assert first == later

    different = Engine._failure_signature("acceptance check failed: pytest returned 1")
    assert different != first


def test_failure_signature_is_case_and_whitespace_stable() -> None:
    from xander_agent.engine import Engine

    assert Engine._failure_signature("  Create Refuses  ") == Engine._failure_signature("create refuses")


def test_the_breaker_trips_below_the_attempt_limit() -> None:
    from xander_agent.engine import Engine

    # Whatever the attempt budget is, three identical setbacks must end it,
    # otherwise the 12-attempt run in the bug report happens again.
    assert Engine.REPEATED_FAILURE_LIMIT <= 3


# --- knowledge sources -------------------------------------------------------

def test_abilities_are_grouped_exhaustively_and_without_overlap() -> None:
    from xander_agent import ABILITIES, ABILITY_GROUPS

    members = [name for group in ABILITY_GROUPS.values() for name in group]
    assert sorted(members) == sorted(ABILITIES)
    assert len(members) == len(set(members)), "an ability must belong to exactly one group"


def test_ability_group_lookup() -> None:
    from xander_agent import ability_group

    assert ability_group("oracle") == "knowledge"
    assert ability_group("quartermaster") == "gear"
    assert ability_group("not-an-ability") == ""


def test_github_provider_parses_a_search_payload() -> None:
    import json as _json

    from xander_agent import web

    payload = _json.dumps(
        {
            "items": [
                {
                    "full_name": "owner/repo",
                    "html_url": "https://github.com/owner/repo",
                    "stargazers_count": 42,
                    "pushed_at": "2026-08-30T12:00:00Z",
                    "open_issues_count": 7,
                    "license": {"spdx_id": "MIT"},
                    "language": "Python",
                    "description": "a thing",
                }
            ]
        }
    )
    rows = web.github("thing", get=lambda url, timeout=8: payload)
    assert rows[0]["title"] == "owner/repo"
    assert rows[0]["provider"] == "github"
    # Health numbers, not just a star count, so a stale repo is visible as stale.
    for marker in ("42", "2026-08-30", "open-issues 7", "MIT"):
        assert marker in rows[0]["snippet"]


def test_github_provider_is_empty_when_the_network_fails() -> None:
    from xander_agent import web

    def boom(url: str, timeout: int = 8) -> str:
        raise OSError("no network")

    assert web.github("anything", get=boom) == []


def test_github_code_needs_a_token(monkeypatch) -> None:
    from xander_agent import web

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert web.github_code("anything") == []


def test_github_readme_rejects_a_non_repo_string() -> None:
    from xander_agent import web

    assert web.github_readme("not-a-repo") == ""


def test_github_is_registered_as_a_provider() -> None:
    from xander_agent.web import PROVIDERS

    assert "github" in PROVIDERS and "github_code" in PROVIDERS


def test_abilities_are_grouped_without_gaps_or_strays() -> None:
    from xander_agent import ABILITIES, ABILITY_GROUPS, ability_group

    grouped = [name for members in ABILITY_GROUPS.values() for name in members]
    assert sorted(grouped) == sorted(ABILITIES)
    assert len(grouped) == len(set(grouped)), "an ability must belong to exactly one group"
    assert ability_group("oracle") == "knowledge"
    assert ability_group("not-an-ability") == ""


def test_github_provider_parses_a_search_payload() -> None:
    import json as _json

    from xander_agent import web

    payload = _json.dumps(
        {
            "items": [
                {
                    "full_name": "octo/thing",
                    "html_url": "https://github.com/octo/thing",
                    "stargazers_count": 12,
                    "pushed_at": "2026-08-30T00:00:00Z",
                    "open_issues_count": 3,
                    "license": {"spdx_id": "MIT"},
                    "language": "Python",
                    "description": "does a thing",
                }
            ]
        }
    )
    rows = web.github("thing", get=lambda url, timeout=8: payload)
    assert rows[0]["title"] == "octo/thing"
    assert rows[0]["provider"] == "github"
    # Health signals, not just a star count, so a dead repo reads as dead.
    assert "2026-08-30" in rows[0]["snippet"] and "MIT" in rows[0]["snippet"]


def test_github_provider_is_empty_when_github_is_unreachable() -> None:
    from xander_agent import web

    def boom(url: str, timeout: int = 8) -> str:
        raise OSError("no network")

    assert web.github("thing", get=boom) == []


def test_github_readme_needs_owner_slash_name() -> None:
    from xander_agent import web

    assert web.github_readme("not-a-repo") == ""
