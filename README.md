# Xander

Offline-first local coding agent — a project manager you hand orders to.
Flexible, clonable, learns from every verified task, and after learning uses
only useful context: grouped general and task-specific skills, relevant tools, and the model
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
| quartermaster | set up | assembles grouped gear by relevance and context budget, plus task tools |
| agent | work | the autonomous plan-and-act loop |
| algorithm | test, judge | deterministic execution, checks, judgment |
| scribe | learn | logs evidence, keeps at most one lesson per win |
| bot | repeat | tireless retries with a changed approach |

## Ready for orders (TUI)

```sh
xander            # opens the TUI in the current workspace
```

Xander opens at a Mission menu grounded in the folder you launched him from.
The opening has five genuinely different layouts, selectable with `--style`
or by pressing `v`: `desk`, `compass`, `chronicle`, `workshop`, and `resident`.
They change the information architecture, not just the palette: a task desk,
a navigation compass, a history-first chronicle, an outcome workshop, and a
minimal resident mode. After `v`, press `1` through `5` to jump directly to
one of them; the picker is inside Xander and leaves a readable selection line
in the Activity log.
Each keeps the same understandable destinations: Mission, History, Soul,
Desktop, and the live work log. The animated logo is decorative only; it never
blocks input or appears in `--json` output.

One command: **Mission**. Press `m` to open its dedicated form. Fill in the
outcome, mode, authority, setup policy, variant, optional time limit, proof
command, allowed paths, and constraints, then press **Start Mission**. Enter on
the compact quick-start field remains available for experienced use.

The launched folder is the workspace, not necessarily a project. Use
`--workspace /absolute/path` when the order concerns an AI configuration,
system configuration, package-manager state, or another operating-system area.
Xander keeps his task records, logs, screenshots, learned skills, and
management files in ASKAR even when the requested target workspace is outside
ASKAR. `setup=ask` is the safe default; choose `setup=allow` only when you want
package or toolchain setup actions to proceed without another prompt.

Before analysis begins, Xander writes a living Mission Guide containing the
statement, ordered TODO, current thought, progress, open questions, meaningful
changes, and result. The guide changes as evidence arrives; it is the thing
Xander uses to explain how far he has come and what he still needs from you.
The focus card above the tabs is the quick brief: `NOW WORKING ON`, `THINKING`,
`LAST MEANINGFUL CHANGE`, and `NEXT`. Activity is concise; Understand, Plan,
Changes, and Proof keep their detailed feeds.

History is the **Mission Library**. Select any Mission—including a completed
one—to read its guide, continue it for further work, or explicitly delete it.
Continuing a completed Mission reopens its guide and asks the engine for a new
plan against the current workspace; it does not discard the earlier result.
`c` cancels an active Mission before a new one is accepted; the current atomic
step is allowed to wind down safely.

Missions are the human name for workspace-scoped task records. They retain
changes, milestones, thoughts, ideas, rebirths, and results in a readable
timeline:

```sh
xander mission list
xander mission show <mission-id>
xander mission delete <mission-id>
```

Desktop observation is explicit (`d`) and local. If no supported Linux
screenshot backend is installed, Xander says so rather than pretending he can
see the screen. A real battery reporting `0%` cancels active work, records the
stop, and sends one `systemctl poweroff` request; missing battery telemetry is
treated as unknown, never as zero.

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
tail -f /home/dicedpingu/SPQR/ASKAR/Xander/logs/xander.log        # central firehose
tail -f /home/dicedpingu/SPQR/ASKAR/Xander/logs/<variant>.log     # one clone's stream
xander oversee                                                   # latest work, routes, delegations, blockers
```

Runtime records stay under `Xander/{config,state,cache,logs}`. New projects
created without an explicitly opened workspace go under `Xander/projects/`.
For an explicitly opened workspace outside that tree, requested source or
configuration changes stay in the target while the readable `PROJECT.md` and
detailed task logs are kept in `Xander/projects/external/`. Shared reviewed
material is under `ASKAR/shared/`; Xander's private learned skill hubs are
under `Xander/knowledge/skills/`.

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
xander run "install the required toolchain" --workspace /path/to/target --setup-policy allow
xander run "inspect the system configuration" --workspace /etc --setup-policy never
xander research "compare the safe package and source changes" --workspace /etc --setup-policy never
xander plan "goal"                        # proposal-only, no mutations
xander doctor                             # health: models, paths, skills, repo
xander-mcp                                # MCP stdio server (proposal-only for callers)
```
