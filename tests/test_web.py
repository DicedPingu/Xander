"""Online abilities: many providers, all optional, never fatal."""

from __future__ import annotations

import json

from xander_agent import web


def test_duckduckgo_parses_titles_urls_and_snippets() -> None:
    page = """
    <div class="result">
      <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fone">First &amp; best</a>
      <a class="result__snippet">A <b>tiny</b> summary</a>
    </div>
    <div class="result">
      <a class="result__a" href="https://direct.example/two">Second</a>
      <a class="result__snippet">More words</a>
    </div>
    """
    rows = web.duckduckgo("anything", get=lambda url, timeout: page)
    assert [row["title"] for row in rows] == ["First & best", "Second"]
    assert rows[0]["url"] == "https://example.com/one", "the /l/?uddg= wrapper must be unwrapped"
    assert rows[1]["url"] == "https://direct.example/two"
    assert rows[0]["snippet"] == "A tiny summary"
    assert all(row["provider"] == "duckduckgo" for row in rows)


def test_wikipedia_reads_the_opensearch_shape() -> None:
    payload = json.dumps(
        ["wasm", ["WebAssembly"], ["A binary instruction format"], ["https://en.wikipedia.org/wiki/WebAssembly"]]
    )
    rows = web.wikipedia("wasm", get=lambda url, timeout: payload)
    assert rows == [
        {
            "title": "WebAssembly",
            "url": "https://en.wikipedia.org/wiki/WebAssembly",
            "snippet": "A binary instruction format",
            "provider": "wikipedia",
        }
    ]


def test_pypi_summarizes_a_package() -> None:
    payload = json.dumps(
        {"info": {"name": "textual", "version": "8.2.8", "summary": "TUIs", "package_url": "https://pypi.org/project/textual/"}}
    )
    rows = web.pypi("textual", get=lambda url, timeout: payload)
    assert rows[0]["title"] == "textual 8.2.8"
    assert rows[0]["provider"] == "pypi"


def test_stackoverflow_reports_score_and_answered() -> None:
    payload = json.dumps(
        {"items": [{"title": "Why is my &amp; escaped?", "link": "https://stackoverflow.com/q/1", "score": 7, "is_answered": True}]}
    )
    rows = web.stackoverflow("escaping", get=lambda url, timeout: payload)
    assert rows[0]["title"] == "Why is my & escaped?"
    assert "score 7" in rows[0]["snippet"]


def test_every_provider_returns_empty_when_the_network_dies() -> None:
    def explode(url: str, timeout: int) -> str:
        raise OSError("no route to host")

    for provider in web.PROVIDERS.values():
        assert provider("anything", get=explode) == []
    assert web.fetch_page("https://example.com", get=explode) == ""
    assert web.search("anything", get=explode) == []


def test_fetch_page_strips_markup_and_refuses_non_http() -> None:
    page = "<html><script>bad()</script><body><h1>Title</h1><p>Body text</p></body></html>"
    text = web.fetch_page("https://example.com", get=lambda url, timeout: page)
    assert text == "Title Body text"
    assert "bad()" not in text
    assert web.fetch_page("file:///etc/passwd", get=lambda url, timeout: "secret") == ""


def test_search_fans_out_and_deduplicates_by_url() -> None:
    same = json.dumps(["q", ["Dup"], ["desc"], ["https://example.com/same"]])
    page = '<a class="result__a" href="https://example.com/same">Dup</a>'

    def get(url: str, timeout: int) -> str:
        return page if "duckduckgo" in url else same

    rows = web.search("dup", providers=("duckduckgo", "wikipedia"), get=get)
    assert len(rows) == 1, "the same URL from two providers collapses to one row"


def test_digest_is_bounded_and_attributed() -> None:
    rows = [{"title": "T", "url": "https://u", "snippet": "S", "provider": "wikipedia"}]
    assert web.digest(rows) == "- [wikipedia] T — S (https://u)"
    assert len(web.digest([{"title": "x" * 5000, "provider": "web", "url": ""}])) <= 2_400
