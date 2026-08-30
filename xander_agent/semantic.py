"""Dense retrieval for the skill registry.

BM25 answers "which cards contain these words". The question actually being
asked is "which cards are about this", and the two come apart constantly: a
search for ``crash loop debugging`` scores exactly zero against a card whose
text says ``CrashLoopBackOff pod``, because they share no token. With 2,000+
cards indexed, that miss is the common case rather than the corner case.

So this adds the other half: embed each card once, embed the query, and rank by
cosine. Neither signal is trusted alone — lexical and dense rankings are fused
with Reciprocal Rank Fusion, which needs no score calibration between the two
and cannot be dominated by one retriever's scale.

Vectors are keyed by ``tree_hash``, not by row id, because ``refresh()`` drops
and rebuilds the skills table; id-keyed vectors would survive as silent
mismatches pointing at the wrong cards.

Everything here degrades to nothing: if the embedder is unreachable or a card
has no vector yet, search falls back to the BM25 behaviour that was already
there. No new dependencies — stdlib only.
"""

from __future__ import annotations

import json
import math
import sqlite3
import urllib.error
import urllib.request
from array import array
from typing import Any, Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_vectors (
    tree_hash TEXT PRIMARY KEY,
    dim       INTEGER NOT NULL,
    vec       BLOB NOT NULL
);
"""

# RRF's damping constant. 60 is the value from the original Cormack et al.
# paper and is deliberately large: it flattens the head so that a single
# retriever ranking something first cannot by itself decide the result.
RRF_K = 60


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA)


def normalize(values: Sequence[float]) -> array:
    """L2-normalize so that cosine similarity is a plain dot product."""

    vector = array("f", (float(value) for value in values))
    norm = math.sqrt(sum(value * value for value in vector))
    if norm > 0:
        for index in range(len(vector)):
            vector[index] /= norm
    return vector


def embed(
    texts: Sequence[str],
    *,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    timeout: int = 120,
) -> list[array] | None:
    """Embed a batch. Returns None when the embedder cannot be reached."""

    if not texts:
        return []
    payload = json.dumps({"model": model, "input": list(texts)}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return None
    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        return None
    return [normalize(vector) for vector in vectors]


def card_text(record: Any) -> str:
    """The text an embedding should stand for: what it is, plus when to use it."""

    def field(key: str) -> str:
        try:
            return str(record[key] or "")
        except (KeyError, IndexError, TypeError):
            return ""

    parts = [field("name"), field("description"), field("triggers")]
    return "\n".join(part for part in parts if part)[:2000]


def backfill(
    registry: Any,
    *,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    batch: int = 32,
    limit: int | None = None,
) -> dict[str, Any]:
    """Embed every card that has no vector for its current content hash."""

    registry.ensure()
    with registry._connect() as connection:
        ensure_schema(connection)
        rows = connection.execute(
            """
            SELECT s.tree_hash, s.name, s.description, s.triggers
            FROM skills s
            LEFT JOIN skill_vectors v ON v.tree_hash = s.tree_hash
            WHERE v.tree_hash IS NULL
            GROUP BY s.tree_hash
            """
        ).fetchall()
        if limit is not None:
            rows = rows[:limit]

        embedded = 0
        failed = 0
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            vectors = embed(
                [card_text(row) for row in chunk],
                model=model,
                base_url=base_url,
            )
            if vectors is None:
                failed += len(chunk)
                # The embedder is down; stop rather than hammer it per batch.
                break
            for row, vector in zip(chunk, vectors):
                connection.execute(
                    "INSERT OR REPLACE INTO skill_vectors(tree_hash,dim,vec) VALUES(?,?,?)",
                    (row["tree_hash"], len(vector), vector.tobytes()),
                )
                embedded += 1
        stale = connection.execute(
            "DELETE FROM skill_vectors WHERE tree_hash NOT IN (SELECT tree_hash FROM skills)"
        ).rowcount

    return {"embedded": embedded, "pending": max(0, len(rows) - embedded), "failed": failed, "pruned": max(0, stale)}


def load_vectors(connection: sqlite3.Connection) -> dict[str, array]:
    try:
        rows = connection.execute("SELECT tree_hash, vec FROM skill_vectors").fetchall()
    except sqlite3.OperationalError:
        return {}
    vectors: dict[str, array] = {}
    for row in rows:
        vector = array("f")
        vector.frombytes(row["vec"])
        vectors[row["tree_hash"]] = vector
    return vectors


def rank(query: array, vectors: dict[str, array]) -> list[tuple[str, float]]:
    """Cosine ranking. Both sides are already normalized, so this is a dot."""

    scored: list[tuple[str, float]] = []
    for tree_hash, vector in vectors.items():
        if len(vector) != len(query):
            continue
        scored.append((tree_hash, sum(a * b for a, b in zip(query, vector))))
    scored.sort(key=lambda item: -item[1])
    return scored


def fuse(rankings: Iterable[Sequence[str]], *, k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion over any number of ranked id lists."""

    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
    return [key for key, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]
