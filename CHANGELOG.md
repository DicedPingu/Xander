# Changelog

All notable changes to Xander. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project is not
tagged in git, so entries are identified by date and commit. `pyproject.toml`
declares `xander-agent 1.0.0`. Calibers and runtime knobs are documented in
[INFO.md](INFO.md).

---

## [Unreleased] — Discussion-first routing and composer commands

The working agreement's first two rules, made mechanical: "discuss before
acting when intent is exploratory" and "anything beginning with `/` is a
command". Touches `intents.py`, `workboard.py`, `tui.py`, `conversation.py`.

### Added
- **Command registry** (`intents.COMMANDS`, `resolve_command`,
  `command_help`) — `/discuss`, `/work`, `/research`, `/goal`, `/addtodo`
  join `/help`, `/mode`, `/cd`, `/talk`. Every `/` line resolves to a known
  command or to `unknown_command` with an explanation and the nearest known
  name; `/rm -rf /` and `/bin/sh …` are explained, never run. `/help` prints
  the list.
- **Discussion-first routing** — `parse_intent` now returns `discuss` for
  scene-setting lines ("this project is going to be about …", "I'm thinking
  of …", "what if we …"), `order` with `authorized=True` only for an
  imperative opening or a clear artifact order, and `draft` for anything
  unclear. The TUI opens a discussion for the first, runs the second, and
  keeps the third as a draft with a one-line explanation. `/work` with no
  argument runs the kept draft, else the newest open goal.
- **Stored goals** (`BoardGoal`, `Workboard.goals`, `add_goal`,
  `open_goals`, `set_goal_state`) — `/goal <goal>` stores direction without
  starting work; `/goal` lists; the pinned-work rail shows open goals.
- **`/research <URL or topic>`** dispatches a read-only `research` request
  from the composer regardless of the mode wheel.

### Fixed
- `/mode …` and `/cd …` typed in the composer were reported as *Unknown
  command*: the interface-only command handler ran before intent parsing and
  swallowed them. All slash lines now go through one resolver, still ahead
  of pending approvals and plan questions so `/cancel` keeps working
  mid-decision.

### Changed
- A bare phrase in `build`/`yolo` mode ("second order", "Codewars client for
  Android") no longer becomes an implementation task silently. It is kept as
  a draft; `/work` authorizes it. In `plan` mode it is still planned.
- The conversational prompt tells the model to point at `/work` rather than
  Enter when the operator is really asking for work.

---

## [Unreleased] — Missions, the opening, and the local-first default

Missions, the opening styles, the caliber catalog and the local-first default,
with `mission.py`, `opening.py`, `desktop.py`, `power.py`, `calibers.py` and
their test modules new.

### Added
- **Missions** (`mission.py`) — the human name for a workspace-scoped task
  record, with a readable timeline of changes, milestones, thoughts, ideas,
  rebirths and results. `xander mission list|show|delete`.
- **The Mission Guide** — written *before* analysis begins and updated as
  evidence arrives: statement, ordered TODO, current thought, progress, open
  questions, last meaningful change, result (`MissionGuide`/`GuideStep` in
  `models.py`). The planner receives it as context, so its questions are
  grounded in the same record the operator reads.
- **Five opening styles** (`opening.py`) — `desk`, `compass`, `chronicle`,
  `workshop`, `resident`. They change the information architecture, not just
  the palette; `--style`, or `v` then `1`–`5` inside the TUI.
- **The focus card** — `NOW WORKING ON` · `THINKING` · `LAST MEANINGFUL
  CHANGE` · `NEXT`, kept separate from the event log.
- **The Mission Library** — History as a browsable list; a completed Mission
  can be reopened for further work, which keeps its evidence and asks for a
  fresh plan against the current workspace.
- **Mission start as a form** — outcome, mode, authority, variant, time limit,
  proof command, allowed paths and constraints, instead of an anonymous text
  box. The quick-start field remains for experienced use.
- **Ask-underway** — when a plan needs a decision, a numbered `[1] … [2] …`
  menu appears in the Run tab and Enter resumes the task.
- **Order queueing** — orders submitted while the engine is busy start
  automatically when the current one finishes.
- **Desktop observation** (`desktop.py`) — explicit (`d`) and local; with no
  supported Linux screenshot backend he says so rather than pretending to see.
- **Power awareness** (`power.py`) — a battery genuinely reporting 0 % cancels
  active work, records the stop, and sends one `systemctl poweroff`; missing
  telemetry is treated as unknown, never as zero.
- Extra `policy.py` guards and executor safety regressions, with tests.
- **The caliber catalog** (`calibers.py`) — one source of truth for which model
  fires in which role. `backend.DEFAULT_MODELS`,
  `variants.VariantProfile.model_routing` and the legacy `config.MODELS` all
  read it, so they can no longer drift. It also resolves a routed tag to the
  best build actually on disk: same family, same parameter size, strictly higher
  precision. A re-tag of identical weights (`:8b` / `:8b-v2`, digest
  `543f9deff86d`) is reported as an *alternative* and never swapped in.
- **The brain is part of the preflight** — `readiness.assess(..., brain=...)`
  now asks whether the backend is reachable and whether every locally routed
  role has a usable build, so a Mission can no longer pass preflight and then
  die mid-run on `all local models failed`. Unreachable yields `ollama serve`;
  an absent build yields `ollama pull <model>`. Cloud-routed roles are exempt,
  and a probe failure is a verdict, never an exception.
- `xander doctor` reports `resolved_routing`, `upgrades_available` and
  `alternative_builds` next to the declared routing.

### Changed
- **The classifier fires the q8_0 build** —
  `qwen2.5-vl-abliterated:3b-instruct-q8_0` (4.6 GB) instead of the q4 `:3b`
  (3.2 GB). The build had been installed and pinned in the legacy tier map since
  the previous entry, but the live package never picked it up.
- **The embedder is catalogued, not routed** — `qwen3-embedding:0.6b` was
  declared in `config.py` and referenced nowhere. It is now a first-class
  `embed` caliber, explicitly marked as not yet wired into the engine, and kept
  out of chat routing because it is not an abliterated build.
- **Default backend is now `ollama`** — `ornith` stays selectable via
  `XANDER_BACKEND=ornith`, but that service is not running and its model is
  refusal-trained, so every run under it paid a failed connect and fell through
  to Ollama anyway.
- **Vision model pinned to the q8_0 build** —
  `qwen2.5-vl-abliterated:3b-instruct-q8_0` in the legacy tier map.
- `c` cancels an active Mission before a new one is accepted, letting the
  current atomic step wind down safely.
- Activity is concise and compresses repeats; Understand, Plan, Changes and
  Proof keep their detailed feeds.

---

## 2026-08-25 — reach and self-extension

### Added
- `ec74e2e` **Operator hooks** (`hooks.py`) — declared broad (a moment plus an
  optional match pattern) and narrowed at muster: only hooks matching the goal
  arm, their constraints join the mission, their notes are spoken at their
  moment. Hooks shape work; they never execute commands. A broken regex
  degrades to substring matching; a malformed file is ignored, never fatal.
- `ec74e2e` **Abilities that make abilities** — `SkillRegistry.author()` writes
  a real `SKILL.md` of Xander's own and reindexes immediately, so what he
  learns the hard way becomes a skill the quartermaster can hand out next time
  (`xander skills create`).
- `ec74e2e` **MCP to MCP** (`mcp_client.py`) — he already served MCP; now he
  calls it. Remote stdio servers declared once in `mcp-servers.json`, each call
  an isolated, timeout-bounded session (`xander remote add|list|tools|call`).
- `ec74e2e` **A bridge out of the terminal** (`bridge.py`, `xander serve`) —
  loopback-only bind that refuses routable hosts, bearer token minted `0600` in
  the config dir, CORS granted to one named extension origin rather than `*`,
  `/ask` and `/stats` read-only, `/order` plan-only unless the body says
  `apply: true`.
- `ce0bf5e` **Hybrid Claude/local brain** — `AnthropicBackend` (claude-opus-5)
  behind the same `generate()` surface, with `anthropic` as the optional
  `xander-agent[cloud]` extra; `HybridBackend` routes per role
  (`anthropic/`-prefixed → cloud, rest → local) and **falls back to local on a
  cloud failure so a lost network never strands a mission**; `backend_for()`
  picks the shape a routing table implies.
- `ce0bf5e` **Online abilities** (`web.py`) — DuckDuckGo, Wikipedia, PyPI,
  StackOverflow and a page fetcher. Stdlib only, hard-timeouted, empty-on-
  failure rather than raising; folded into the Researcher and the Lurker's
  brief. `XANDER_OFFLINE=1` keeps him home (and the test suite sets it).

### Changed
- `ce0bf5e` `VariantProfile` keeps the **abliterated-only rule** for local
  routing and exempts `anthropic/` cloud routes.
- `926fd47` The optional cloud extra is locked.

### Removed
- `ce0bf5e` **The Smith form** — building is Xander's own trade, so ordinary
  implement missions wear no costume; he is simply Xander. Medic, Lurker and
  Master remain.

*Suite at these commits: 122 passed.*

---

## 2026-08-25 — squad, voice, and the scoreboard

### Added
- `a363945` **The squad** (`squad.py`) — **Master**, the supervisor, walks with
  every mutating mission and must end up happy: bounded to 2 plan reviews, and
  one unhappy final verdict buys exactly one reshaped attempt. **Lurker**, the
  researcher, fires on measurable budgets ("less than 30 KB") or complexity ≥ 3
  and returns one ≤ 1200-character brief of concrete techniques.
- `a363945` **Forms** — `choose_form()` gives each mission a flavour (Medic for
  repair, Lurker for read-only digging, Master for plan-only work) that colours
  the kickoff voice.
- `a363945` **Answer mode** — research, then one critic prose reply, no file
  changes.
- `14c2a96` **The Commentator** (`commentary.py`) — a first-person working
  voice with moments (kickoff, approach, action, setback, victory, progress,
  question, feedback ack), per-task line caps by voice profile, deterministic
  offline templates, bounded small-model rewording, and **fatigue that builds
  over repeated setbacks and resets on victory**. Progress reports address the
  Master.
- `14c2a96` **Preference learning** — a capped like/dislike/style list in
  `MemoryStore` (explicit vs inferred, weight bumps on repeats, explicit source
  never downgraded), injected into the planner as an `OPERATOR PREFERENCES`
  block. Constraints explicitly outrank preferences.
- `4847553` **The scoreboard** (`stats.py`, `xander stats`) — deterministic
  from the TaskStore: success rate, streaks, attempts, actions/checks, model
  calls/tokens/tok-s, Master-happiness and Lurker-brief counts, per-variant
  wins, a 14-day sparkline. Surfaced as a TUI Stats tab.
- `4847553` **Conversational intent routing** in the TUI — `/help`, a `/mode`
  wheel (ask/plan/build/yolo), chdir rebinding, feedback recorded immediately
  even mid-run instead of being queued as work, advisory questions answered in
  the engine's answer mode, and queued orders remembering their mode.
- `ae2d592` **Feedback intents** — "I don't like…", "always…", "never…", "stop
  doing…" become preference-store feedback rather than orders;
  artifact-targeting verbs still win as orders.
- `2f2ca46` `docs/DESIGN-interactive-xander.md` — the operator brief for
  interactive, learning, squad-running Xander.

### Changed
- `14c2a96` `XanderEvent` gains the `voice` type; `XanderRequest` gains the
  `answer` mode; the narrator gains a voice glyph/style with speaker prefix and
  `VariantProfile` a `voice` knob.

*Suite across these commits: 58 → 73 → 86 → 91 passed.*

---

## 2026-08-25 — baseline

- `9302736` Xander was untracked inside the ASKAR working tree; the existing
  source is committed as-is to give the interface rework a diffable starting
  point and a rollback target. Verified first: 40 tests pass, Ollama reachable,
  both `xander` and `xander-mcp` resolving to editable installs of this tree.
  Excluded from tracking (present on disk): `state/`, `logs/`, `out/` —
  superseded by the XDG paths in `xander_agent/paths.py` — plus 3.9 GB of Rust
  build artifacts under `labs/`, `.venv/` and caches.
