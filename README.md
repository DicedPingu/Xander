# Xander

Offline-first local coding agent — a project manager you hand orders to.
Flexible, clonable, learns from every verified task, and after learning uses
only the essential: at most 3 skills, only the relevant tools, only the model
depth the task needs.

## The mantra

Analyze › Research › Set yourself up › Work › Test › Judge/log › Learn › Repeat

Every order runs the full loop. Nothing is claimed done without command, diff,
artifact, or test evidence.

## Abilities

Xander carries a little of every ability and deploys only what's relevant:

| ability | used in phase | what it does |
|---|---|---|
| llm | analyze | language judgment, classification |
| oracle | research | local truth, current docs, existing solutions |
| quartermaster | set up | selects minimal gear: 1–3 skills, task tools |
| agent | work | the autonomous plan-and-act loop |
| algorithm | test, judge | deterministic execution, checks, judgment |
| scribe | learn | logs evidence, keeps at most one lesson per win |
| bot | repeat | tireless retries with a changed approach |

## Ready for orders (TUI)

```sh
xander            # opens the TUI in the current workspace
```

One command: **Work**. Press `w`, type the order, press Enter. The Run tab
narrates every phase live; Research, Plan, Diff, and Tests keep their own
feeds. Tasks, Skills, and Variants tabs show the persisted state.

- Orders submitted while the engine is busy are **queued** and start
  automatically when the current one finishes.
- When a plan needs a decision, Xander **asks underway**: a numbered
  `[1] … [2] …` menu appears in the Run tab — answer with numbers or ids
  and Enter resumes the task.

Keys: `[w]` work · `[enter]` run/queue · `[^p]` pause · `[^P]` resume ·
`[^n]` new · `[^t]` tests · `[^q]` quit · `[f1]` help

## Follow along outside the TUI

Every narrated line has a plain twin on disk:

```sh
tail -f ~/.local/state/xander/logs/xander.log        # central firehose
tail -f ~/.local/state/xander/logs/<variant>.log     # one clone's stream
```

## Global command

Installed once with `uv tool install --editable .` — `xander` and `xander-mcp`
land on PATH (`~/.local/bin`) and track the repo live, like the other agents.

## The army

Clones form a chain of command. Lineage decides rank: the root profile leads,
direct clones are captains, their clones sergeants, deeper recruits troopers.

```sh
xander variant clone scout --from default      # recruit a captain
xander variant clone forward --from scout      # recruit a sergeant under scout
xander army                                    # muster: leader, ranks, lessons, wins
xander variant export scout --output scout.zip # portable, hash-verified bundle
```

Each clone has its own model routing, directives, skill groups, theme, and
memory namespace — lessons learned by one clone stay with that clone, and the
muster shows every clone's lessons and verified wins. The TUI's Variants tab
renders the same hierarchy live.

## Other interfaces

```sh
xander run "goal" --accept "pytest -q"    # scriptable CLI (add --json for JSONL)
xander plan "goal"                        # proposal-only, no mutations
xander doctor                             # health: models, paths, skills, repo
xander-mcp                                # MCP stdio server (proposal-only for callers)
```
