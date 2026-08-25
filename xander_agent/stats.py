"""Fancy stats: one honest scoreboard aggregated from persisted tasks.

Everything here is deterministic reading of the TaskStore — no model calls.
Model token/speed numbers come from the ``{"kind": "model", ...}`` evidence
the engine appends after each generate, so the scoreboard reflects what was
actually spent, not what was configured.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from .models import TaskStatus
from .tasks import TaskStore

_SPARK = "▁▂▃▄▅▆▇█"
_DAYS = 14


def sparkline(counts: list[int]) -> str:
    if not counts:
        return ""
    peak = max(counts) or 1
    return "".join(_SPARK[min(len(_SPARK) - 1, round(value / peak * (len(_SPARK) - 1)))] for value in counts)


def stats_payload(store: TaskStore | None = None, *, limit: int = 500) -> dict[str, Any]:
    records = (store or TaskStore()).list(limit=limit)
    by_status = Counter(record.status.value for record in records)
    completed = by_status.get(TaskStatus.COMPLETED.value, 0)

    streak = 0
    for record in records:  # newest first
        if record.status == TaskStatus.COMPLETED:
            streak += 1
        else:
            break

    actions_ok = actions_failed = checks_passed = checks_total = 0
    model_calls = model_tokens = 0
    speed_samples: list[float] = []
    subjects: Counter[str] = Counter()
    wins_by_variant: Counter[str] = Counter()
    attempts: list[int] = []
    for record in records:
        attempts.append(record.attempt)
        if record.subject:
            subjects[record.subject] += 1
        if record.status == TaskStatus.COMPLETED:
            wins_by_variant[record.request.variant] += 1
        for result in record.results:
            if result.status.value == "ok":
                actions_ok += 1
            elif result.status.value == "failed":
                actions_failed += 1
        checks_total += len(record.check_results)
        checks_passed += sum(1 for item in record.check_results if item.status.value == "ok")
        for item in record.evidence:
            if item.get("kind") == "model":
                model_calls += 1
                model_tokens += int(item.get("tokens") or 0)
                if item.get("tokens_per_second"):
                    speed_samples.append(float(item["tokens_per_second"]))

    master_verdicts = sum(
        1 for record in records for item in record.evidence if item.get("kind") == "master_verdict"
    )
    master_happy = sum(
        1
        for record in records
        for item in record.evidence
        if item.get("kind") == "master_verdict" and item.get("happy")
    )
    lurker_briefs = sum(
        1 for record in records for item in record.evidence if item.get("kind") == "lurker_brief"
    )

    today = datetime.now(UTC).date()
    day_counts = [0] * _DAYS
    for record in records:
        try:
            created = datetime.fromisoformat(record.created_at).date()
        except ValueError:
            continue
        age = (today - created).days
        if 0 <= age < _DAYS:
            day_counts[_DAYS - 1 - age] += 1

    return {
        "event": "stats",
        "schema": "xander.stats/v1",
        "tasks": len(records),
        "by_status": dict(by_status),
        "success_rate": round(completed / len(records), 3) if records else 0.0,
        "current_streak": streak,
        "average_attempts": round(sum(attempts) / len(attempts), 2) if attempts else 0.0,
        "actions": {"ok": actions_ok, "failed": actions_failed},
        "checks": {"passed": checks_passed, "total": checks_total},
        "model": {
            "calls": model_calls,
            "tokens": model_tokens,
            "average_tokens_per_second": round(sum(speed_samples) / len(speed_samples), 2)
            if speed_samples
            else 0.0,
        },
        "squad": {
            "master_verdicts": master_verdicts,
            "master_happy": master_happy,
            "lurker_briefs": lurker_briefs,
        },
        "top_subjects": [subject for subject, _ in subjects.most_common(5)],
        "wins_by_variant": dict(wins_by_variant),
        "last_days": day_counts,
        "sparkline": sparkline(day_counts),
        "window_start": (today - timedelta(days=_DAYS - 1)).isoformat(),
    }


def render_lines(payload: dict[str, Any]) -> list[str]:
    """Rich-markup lines for the TUI Stats tab (and plain CLI, stripped)."""

    tasks = payload.get("tasks", 0)
    rate = float(payload.get("success_rate", 0.0))
    rate_style = "bold green" if rate >= 0.7 else "bold yellow" if rate >= 0.4 else "bold red"
    actions = payload.get("actions", {})
    checks = payload.get("checks", {})
    model = payload.get("model", {})
    squad = payload.get("squad", {})
    lines = [
        f"[bold #ffcb6b]scoreboard[/] · {tasks} task(s) · "
        f"[{rate_style}]{rate:.0%} verified[/] · streak [bold]{payload.get('current_streak', 0)}[/]",
        f"  activity  [cyan]{payload.get('sparkline', '')}[/] [dim](last {_DAYS} days from {payload.get('window_start', '?')})[/]",
        f"  attempts  avg {payload.get('average_attempts', 0)} per task",
        f"  actions   [green]{actions.get('ok', 0)} ok[/] · [red]{actions.get('failed', 0)} failed[/]"
        f"  ·  checks [green]{checks.get('passed', 0)}/{checks.get('total', 0)} passed[/]",
        f"  models    {model.get('calls', 0)} call(s) · {model.get('tokens', 0)} tokens · "
        f"{model.get('average_tokens_per_second', 0)} t/s avg",
        f"  squad     Master happy {squad.get('master_happy', 0)}/{squad.get('master_verdicts', 0)}"
        f" · Lurker briefs {squad.get('lurker_briefs', 0)}",
    ]
    statuses = payload.get("by_status", {})
    if statuses:
        rendered = "  ".join(f"{name}:{count}" for name, count in sorted(statuses.items()))
        lines.append(f"  status    [dim]{rendered}[/]")
    wins = payload.get("wins_by_variant", {})
    if wins:
        rendered = "  ".join(f"{name}:{count}" for name, count in sorted(wins.items(), key=lambda kv: -kv[1]))
        lines.append(f"  wins      [dim]{rendered}[/]")
    subjects = payload.get("top_subjects", [])
    if subjects:
        lines.append("  subjects  [dim]" + " · ".join(str(item) for item in subjects[:5]) + "[/]")
    return lines
