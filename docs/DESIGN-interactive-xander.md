# Design: Interactive, learning, squad-running Xander

Operator brief (2026-08-25): Xander should feel like a capable human teammate.
He starts the mission the moment the order lands, talks in humanlike, relevant
lines while working ("Making the background green, unless told otherwise",
"afterwards I need to learn what about WebAssembly could be offsetting"),
asks questions without stalling, learns what the operator likes/dislikes and
shapes his writing and working style from it, keeps fancy logs and stats, and
for a mission like "tic-tac-toe website under X KB" fields helper clones —
one supervising, one researching the tight constraint.

## 1. `xander_agent/commentary.py` — the voice (NEW)
- `Commentator(backend, memory, style)` produces short first-person lines at
  fixed moments; engine emits them as a new `XanderEvent` type `"voice"`.
- Moments: `kickoff` (right after the order is accepted — before analysis
  finishes), `approach` (after each plan: what he's doing + any default chosen,
  phrased "…, unless told otherwise"), `action` (first notable mutation per
  attempt), `setback` (blocking failure / failed checks: what happened + what
  he needs to learn next), `victory`, `question` (voiced default pick when a
  plan has non-required options).
- Generation: one small-model call per moment (role `classifier`, 3B, short
  timeout ~30s, `num_predict` tiny). Deterministic template fallback whenever
  the backend is down, slow, or errors. Hard cap ~10 voice lines per task.
- Style knobs come from learned preferences (see §2): verbosity, formality,
  emoji tolerance. New `VariantProfile.voice: "chatty"|"quiet"|"off"`
  (default chatty; old saved profiles load fine — missing key → default).

## 2. Preference learning — `memory.py` + `intents.py`
- MemoryStore grows a `preferences` list: `{text, kind: like|dislike|style,
  source: explicit|inferred, weight, created_at}`; capped, atomic save as now.
- Explicit capture: new intent kind `feedback` in `intents.py` — patterns
  like "i (don't) like…", "always…", "never…", "stop doing…", "prefer…",
  "less/more …". Feedback typed **during a run** is recorded immediately and
  acknowledged by voice — never queued as an order.
- Inferred capture: denied approvals and operator interrupts record a soft
  dislike tied to the plan summary; quick "nice/perfect/good" after a win
  records a like.
- Application: an `OPERATOR PREFERENCES` block joins the planner prompt in
  `engine._plan` (beside STANDING DIRECTIVES / LESSONS), and the commentator
  system prompt, so both the work and the voice adapt.

## 3. `xander_agent/stats.py` — fancy logging & stats (NEW)
- Aggregate from TaskStore + per-call backend stats. Engine appends
  `{"kind": "model", **backend.last_stats}` to `task.evidence` after each
  generate (analyze/plan/voice/critic) so tokens & t/s are persisted.
- Payload: totals by status, success rate, current streak, avg attempts,
  actions ok/failed, checks passed, model calls/tokens/avg t/s, top subjects,
  per-variant wins, lessons + preferences counts, last-14-days activity.
- Surfaces: `xander stats` CLI command (rich table + block-glyph sparkline)
  and a new TUI "Stats" tab. Narrator keeps writing the plain-file twin logs.

## 4. Instant start + conversational routing
- Orders already run on Enter (TUI) — keep; add an immediate `kickoff` voice
  line before analysis so he speaks the moment the order lands.
- Wire `intents.parse_intent` into `tui.on_input_submitted`:
  `help`/`mode`/`chdir` handled locally; `feedback` → §2; `advice` → new
  engine mode `"answer"` (research, then one critic-role prose reply emitted
  as a voice/result — no files change); `order` → run in the configured mode.
- `XanderRequest.mode` Literal gains `"answer"`; engine branch after research.

## 5. `xander_agent/squad.py` — helper clones (NEW)
Canon (operator, 2026-08-25): the supervisor is the **Master**, the
researcher is the **Lurker**, and Xander himself takes a *form* per mission.

- `Squad.muster(goal, complexity, has_checks, mode)` picks ≤2 helpers.
- **Lurker** (researcher): fires when complexity ≥3 or the goal carries a
  measurable budget (regex e.g. "less than \d+ ?[KMG]?B", ms, seconds).
  One bounded critic-role dig into the tight constraint → ≤1200-char brief
  appended to `task.research.documentation`, voiced with his name.
- **Master** (supervisor, "the one who does the judging++"): walks with
  *every* mutating mission — he has to end up happy with the result, which
  matters most exactly when the operator gave no real acceptance checks.
  Bounded to 2 plan reviews per task; his concerns join the constraints.
  After the deterministic judge passes, the Master weighs the finished
  evidence: one unhappy verdict buys exactly one reshaped attempt, then the
  work stands with his dissent on record. The worker reports progress to
  him by name each attempt: "Master — attempt 2: 3/5 action(s) landed…".
- **Forms Xander takes** (`choose_form`): **Smith** (builder — implement),
  **Medic** (repairer — test-triage / fix-the-failing goals), **Lurker**
  (read-only digs and answers), **Master** (plan-only judging missions).
  The form names the kickoff voice: "Taking Medic form for this one."
- All helper work is bounded (call caps + timeouts), best-effort, and
  narrated with a speaker tag in the voice event data (`{"speaker": name}`).

### Voice canon
Personality, but sparing: he says *how* he's building things (color,
technique, file), states chosen defaults as "…, unless told otherwise",
admits failures, and audibly **tires of repetitive failure** — fatigue
builds per consecutive setback and resets on victory, pushing him to
change the shape of the plan. Chatter is capped per task (chatty 10 /
quiet 4 / off); setbacks, victories, questions, and acks always land.

## Cross-cutting edits
- `models.py`: `"voice"` in XanderEvent.type; `"answer"` in XanderRequest.mode.
- `narrator.py`: style for `voice` (distinct glyph, run channel, speaker
  prefix); stats digest helper.
- `engine.py`: commentator + squad wiring, model-call evidence, answer mode.
- `tui.py`: intent routing, Stats tab, feedback-during-run path.
- `cli.py`: `stats` subcommand (and it powers the TUI tab).
- Tests: commentary fallback templates (offline), preference capture &
  prompt injection, feedback/advice intent patterns, stats aggregation from
  fixture TaskRecords, squad muster heuristics (no model needed).

## Order of work (each step leaves the suite green)
1. ✅ intents feedback/advice patterns + tests. (ae2d592)
2. ✅ memory preferences + planner-prompt injection + tests. (14c2a96)
3. ✅ models/narrator voice event + commentary.py with template fallback + tests. (14c2a96)
4. ✅ engine wiring (kickoff/approach/setback/victory/progress) + model-call evidence. (a363945)
5. ✅ stats.py + CLI + TUI tab + tests. (4847553)
6. ✅ squad.py (Master/Lurker + Smith/Medic forms) + engine wiring + tests. (a363945)
7. ✅ answer mode + TUI intent routing polish. (a363945, 4847553)

All seven steps landed 2026-08-25; suite at 91 passed.
