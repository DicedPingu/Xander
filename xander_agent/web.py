"""Online search abilities: many small ways to look things up, none required.

Xander is offline-first; the network is a bonus, never a dependency. Every
provider here is stdlib-only, hard-timeouted, and returns an empty result
instead of raising when the world is unreachable. Providers:

- ``duckduckgo``  — general web search (HTML endpoint, no key)
- ``wikipedia``   — encyclopedia lookup (opensearch JSON API)
- ``pypi``        — Python package lookup (JSON API)
- ``stackoverflow`` — Q&A search (StackExchange JSON API)
- ``github``      — repository search (JSON API; reads GITHUB_TOKEN/GH_TOKEN if set)
- ``github_code`` — code search across public repos (needs a token; empty without one)
- ``fetch_page``  — pull one URL and strip it to readable text

``search()`` fans out across providers and folds everything into one
bounded, source-attributed digest for research or a Lurker brief.
"""

from __future__ import annotations

import gzip
import os
import html
import json
import re
import urllib.parse
import urllib.request
from typing import Callable

_TIMEOUT = 8
_USER_AGENT = "xander-agent/1.0 (+local coding agent)"
_MAX_RESULTS = 5
_MAX_PAGE_CHARS = 12_000

Fetcher = Callable[[str, int], str]


def _http_get(url: str, timeout: int = _TIMEOUT) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_000)
        if response.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", "replace")


_TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.IGNORECASE | re.DOTALL)
_DDG_RESULT = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_DDG_SNIPPET = re.compile(
    r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<text>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)


def _clean(text: str) -> str:
    return html.unescape(" ".join(_TAG.sub(" ", text).split()))


def _ddg_url(raw_href: str) -> str:
    """DuckDuckGo wraps result links as /l/?uddg=<encoded-url>."""

    parsed = urllib.parse.urlparse(raw_href)
    if parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        return target or raw_href
    return raw_href


def duckduckgo(query: str, *, get: Fetcher = _http_get, limit: int = _MAX_RESULTS) -> list[dict[str, str]]:
    try:
        page = get("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query), _TIMEOUT)
    except Exception:
        return []
    results = []
    snippets = [_clean(match.group("text")) for match in _DDG_SNIPPET.finditer(page)]
    for index, match in enumerate(_DDG_RESULT.finditer(page)):
        if len(results) >= limit:
            break
        results.append(
            {
                "title": _clean(match.group("title")),
                "url": _ddg_url(match.group("href")),
                "snippet": snippets[index] if index < len(snippets) else "",
                "provider": "duckduckgo",
            }
        )
    return results


def wikipedia(query: str, *, get: Fetcher = _http_get, limit: int = 3) -> list[dict[str, str]]:
    try:
        raw = get(
            "https://en.wikipedia.org/w/api.php?action=opensearch&format=json&limit="
            f"{limit}&search=" + urllib.parse.quote_plus(query),
            _TIMEOUT,
        )
        _, titles, descriptions, urls = json.loads(raw)
    except Exception:
        return []
    return [
        {"title": title, "url": url, "snippet": description, "provider": "wikipedia"}
        for title, description, url in zip(titles, descriptions, urls)
    ]


def pypi(package: str, *, get: Fetcher = _http_get) -> list[dict[str, str]]:
    name = re.sub(r"[^A-Za-z0-9._-]", "", package.strip())
    if not name:
        return []
    try:
        raw = get(f"https://pypi.org/pypi/{name}/json", _TIMEOUT)
        info = json.loads(raw).get("info", {})
    except Exception:
        return []
    if not info.get("name"):
        return []
    return [
        {
            "title": f"{info['name']} {info.get('version', '')}".strip(),
            "url": info.get("package_url", f"https://pypi.org/project/{name}/"),
            "snippet": (info.get("summary") or "")[:300],
            "provider": "pypi",
        }
    ]


def stackoverflow(query: str, *, get: Fetcher = _http_get, limit: int = 3) -> list[dict[str, str]]:
    try:
        raw = get(
            "https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance"
            "&site=stackoverflow&pagesize=" + str(limit) + "&q=" + urllib.parse.quote_plus(query),
            _TIMEOUT,
        )
        items = json.loads(raw).get("items", [])
    except Exception:
        return []
    return [
        {
            "title": html.unescape(item.get("title", "")),
            "url": item.get("link", ""),
            "snippet": f"score {item.get('score', 0)}, answered: {item.get('is_answered', False)}",
            "provider": "stackoverflow",
        }
        for item in items[:limit]
    ]


def _github_fetcher() -> Fetcher:
    """GitHub unauthenticated is 10 searches/minute; a token makes it 30.

    Xander should use the operator's token when there is one and still work
    when there is not, so this returns a plain fetcher rather than failing.
    """

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        return _http_get

    def get(url: str, timeout: int = _TIMEOUT) -> str:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(2_000_000).decode("utf-8", "replace")

    return get


def github(query: str, *, get: Fetcher | None = None, limit: int = _MAX_RESULTS) -> list[dict[str, str]]:
    """Search repositories, best-starred first.

    Stars are a weak signal and a stale one, so the snippet carries the numbers
    that say whether a repo is actually alive — pushed date, open issues,
    license — instead of leaving a ranking to be mistaken for a recommendation.
    """

    get = get or _github_fetcher()
    try:
        page = get(
            "https://api.github.com/search/repositories?sort=stars&order=desc&per_page="
            + str(limit)
            + "&q="
            + urllib.parse.quote_plus(query),
            _TIMEOUT,
        )
        items = json.loads(page).get("items", [])
    except Exception:
        return []
    rows = []
    for item in items[:limit]:
        licence = (item.get("license") or {}).get("spdx_id") or "no license"
        rows.append(
            {
                "title": str(item.get("full_name", "?")),
                "url": str(item.get("html_url", "")),
                "snippet": (
                    f"{item.get('stargazers_count', 0)}star "
                    f"pushed {str(item.get('pushed_at', ''))[:10]} "
                    f"open-issues {item.get('open_issues_count', 0)} "
                    f"{licence} {item.get('language') or ''} "
                    f"- {item.get('description') or ''}"
                ).strip()[:400],
                "provider": "github",
            }
        )
    return rows


def github_code(query: str, *, get: Fetcher | None = None, limit: int = 3) -> list[dict[str, str]]:
    """Search code. GitHub requires auth here, so without a token this is empty."""

    if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")):
        return []
    get = get or _github_fetcher()
    try:
        page = get(
            "https://api.github.com/search/code?per_page="
            + str(limit)
            + "&q="
            + urllib.parse.quote_plus(query),
            _TIMEOUT,
        )
        items = json.loads(page).get("items", [])
    except Exception:
        return []
    return [
        {
            "title": f"{(item.get('repository') or {}).get('full_name', '?')}: {item.get('path', '?')}",
            "url": str(item.get("html_url", "")),
            "snippet": str(item.get("path", ""))[:200],
            "provider": "github_code",
        }
        for item in items[:limit]
    ]


def github_readme(repo: str, *, get: Fetcher | None = None, max_chars: int = _MAX_PAGE_CHARS) -> str:
    """Read a repo's README as text. `repo` is "owner/name".

    This is the step that turns a search hit into knowledge: a ranked list says
    a repo exists, the README says whether it does what the task needs.
    """

    if "/" not in repo:
        return ""
    get = get or _github_fetcher()
    for branch in ("HEAD",):
        try:
            page = get(f"https://api.github.com/repos/{repo}/readme", _TIMEOUT)
            payload = json.loads(page)
        except Exception:
            return ""
        content = payload.get("content") or ""
        if payload.get("encoding") == "base64" and content:
            import base64

            try:
                return base64.b64decode(content).decode("utf-8", "replace")[:max_chars]
            except Exception:
                return ""
    return ""


def fetch_page(url: str, *, get: Fetcher = _http_get, max_chars: int = _MAX_PAGE_CHARS) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    try:
        return _clean(get(url, _TIMEOUT))[:max_chars]
    except Exception:
        return ""


PROVIDERS: dict[str, Callable[..., list[dict[str, str]]]] = {
    "duckduckgo": duckduckgo,
    "wikipedia": wikipedia,
    "pypi": pypi,
    "stackoverflow": stackoverflow,
    "github": github,
    "github_code": github_code,
}


def search(
    query: str,
    *,
    providers: tuple[str, ...] = ("duckduckgo", "wikipedia"),
    get: Fetcher = _http_get,
) -> list[dict[str, str]]:
    """Fan out one query across providers; failures just mean fewer rows."""

    results: list[dict[str, str]] = []
    for name in providers:
        provider = PROVIDERS.get(name)
        if provider is None:
            continue
        results.extend(provider(query, get=get))
    seen: set[str] = set()
    unique = []
    for row in results:
        key = row.get("url") or row.get("title", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(row)
    return unique[: _MAX_RESULTS * 2]


def digest(rows: list[dict[str, str]], *, limit_chars: int = 2_400) -> str:
    """Prompt-ready digest with source attribution."""

    lines = []
    for row in rows:
        snippet = f" — {row['snippet']}" if row.get("snippet") else ""
        lines.append(f"- [{row.get('provider', 'web')}] {row.get('title', '?')}{snippet} ({row.get('url', '')})")
    return "\n".join(lines)[:limit_chars]
