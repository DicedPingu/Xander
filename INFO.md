# Xander — what he is made of

Monica's counterpart. Where Monica *looks things up*, Xander **acts**: given an
order he researches, assembles the gear the task needs, plans, executes, tests,
judges the result, and learns one lesson per verified win. Offline-first — the
network is a bonus, never a dependency.

Source of truth for everything below: `xander_agent/backend.py`,
`xander_agent/variants.py`, `xander_agent/policy.py`, `pyproject.toml`, and the
legacy top-level `config.py`.

---

## Calibers — the models he fires

Routing is **per role**, not per session: one mission uses several models, each
where it is strongest. Operator policy is enforced in code
(`variants.VariantProfile`): a local routing entry is **rejected unless it is an
abliterated build** — `anthropic/`-prefixed entries are cloud and exempt.

### Local roles (`calibers.CALIBERS`, the one catalog)

`xander_agent/calibers.py` is the single source of truth. `backend.DEFAULT_MODELS`,
`variants.VariantProfile.model_routing` and the legacy `config.MODELS` all read
it, so they can no longer drift apart on which build a role fires.

| role | model | on disk | used for |
|---|---|---|---|
| `coder` | `huihui_ai/qwen2.5-coder-abliterate:7b` | 4.7 GB | precise edits, command synthesis — also the default when a role is unknown |
| `planner` | `huihui_ai/qwen3-abliterated:8b` | 5.0 GB | analysis, planning, replanning |
| `classifier` | `huihui_ai/qwen2.5-vl-abliterated:3b-instruct-q8_0` | 4.6 GB | cheap routing: intent detection, commentary, quick judgments |
| `critic` | `huihui_ai/qwen3-abliterated:8b` | 5.0 GB | the squad's judgment seat — Master verdicts, Lurker briefs |
| `embedder` | `qwen3-embedding:0.6b` | 639 MB | catalogued and installed; **not yet wired into the engine** |

The embedder is deliberately outside `DEFAULT_MODELS`: it emits vectors rather
than prose, so it sits outside chat routing and outside the abliterated-only
policy that guards it.

**Best-build resolution:** a routed tag is replaced at call time by the highest
precision build of the *same family and parameter size* that Ollama actually has
— which is how `classifier` reaches the q8_0 build. Resolution is conservative
on purpose: a re-tag of identical weights (`:8b` and `:8b-v2` share digest
`543f9deff86d`) is reported as an *alternative* and never swapped in, because
reloading it costs a model eviction and buys nothing. A local build never leaves
the abliterated set, and `anthropic/` routes are never rewritten.

**Role fallback:** preferred role → `coder` → `planner` → `critic`, each
resolved to its best installed build, then filtered to what Ollama actually
reports as installed (falling back to the unfiltered order if nothing matches,
so a doctor run still explains itself).

**`xander doctor` reports** `resolved_routing` (what will really fire),
`upgrades_available` (a routed tag beaten by a build already on disk) and
`alternative_builds` (siblings that are not a precision win).

**VRAM discipline:** the Ollama backend keeps **one resident model** — before
loading a new one it unloads the previous (`keep_alive: 0`), and all generation
is serialised behind a process-wide lock. That matches the host's Ollama server, which runs
with `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_FLASH_ATTENTION=1` and
`OLLAMA_KV_CACHE_TYPE=q8_0`.

**Structured output:** any call may pass a pydantic model or JSON schema as
`format`, so plans, actions and verdicts come back grammar-constrained instead
of parsed out of prose.

### Cloud roles (optional)

| backend | model | gate |
|---|---|---|
| `AnthropicBackend` | `claude-opus-5` by default; any `anthropic/…` routing value | the `anthropic` package from the `xander-agent[cloud]` extra, plus `~/.config/anthropic` |
| `HybridBackend` | per-role dispatch | `anthropic/`-prefixed roles go to the cloud, everything else stays local; **a cloud failure falls back to local**, so a lost network never strands a mission |

### Legacy top-level tiers (`config.py`)

The root modules (`config.py`, `agent.py`, `backend.py`, `memory.py`,
`research.py`, `toolbelt.py`) are the pre-package Xander, still on disk. They
keep their own tier names — `coder`, `thinker`, `fast`, `vision`, `embed`, with
`FALLBACK_MODELS = thinker → coder → fast` — but those are now aliases onto the
shared catalog rather than a second opinion (`fast` and `vision` both resolve to
the q8_0 vision build). The live agent is `xander_agent/`; treat the root set as
history.

`ornith` remains selectable via `XANDER_BACKEND=ornith` (OpenAI-compatible
shape, `XANDER_ORNITH_URL`/`_KEY`/`_MODEL`/`_CHAT_PATH`) but the service is not
running and its model is refusal-trained — every run under it paid a failed
connect and fell through to Ollama anyway, which is why the default moved.

---

## Runtime knobs

| knob | value | env |
|---|---|---|
| backend | `ollama` | `XANDER_BACKEND` |
| endpoint | `http://127.0.0.1:11434` | `OLLAMA_API_BASE` |
| generation timeout | 600 s | `XANDER_GEN_TIMEOUT` |
| connect timeout | 10 s | — |
| keep-alive | `15m` | `XANDER_KEEP_ALIVE` |
| commentary / squad call timeouts | 30 s / 60 s | — |
| config · state · cache | `Xander/config`, `Xander/state`, `Xander/cache` inside ASKAR | `XANDER_CONFIG_DIR`, `XANDER_STATE_DIR`, `XANDER_CACHE_DIR` for explicit isolation/test overrides |

Python ≥ 3.11 · `mcp`, `pydantic`, `textual`, `packaging` · installed with
`uv tool install --editable .`, which puts `xander` and `xander-mcp` on PATH.

---

## The loop

```
Analyze › Research › Set yourself up › Work › Test › Judge/log › Learn › Repeat
```

Nothing is claimed done without command, diff, artifact or test evidence.
Phases, task records, actions, verdicts and events are all strict pydantic
schemas (`models.py`, `schema: xander.task/v1` and friends) — `extra="forbid"`,
so a malformed plan fails at the boundary instead of halfway through execution.

## The squad

Xander wears a **form** per mission and may take at most two helpers; the
Master always walks along on mutating work.

| member | role model | job |
|---|---|---|
| **Master** | `critic` | the supervisor. Judges the plan, then the finished result — an empty verdict means he is happy |
| **Lurker** | `critic` | the researcher. Reads everything, touches nothing; fires when the mission carries a tight question |
| **Medic** | `critic` | the repair form — reproduces a failure first, then heals it |

## Safety

- **Risk classification** (`policy.py`) — `CRITICAL_PATTERNS`, read-only argv
  detection, git-subcommand and clone-destination analysis, secret-path checks,
  `resolve_inside()` so nothing escapes the workspace, and
  `validate_allowed_paths()` against the mission's declared allowlist.
- **Preimage hashes** on patch actions — a file that changed under Xander's
  feet aborts the patch instead of clobbering it.
- **Autonomy levels** — `proposal` · `supervised` · `full-auto`; callers
  identifying as `codex` or `claude` are forced down to **proposal-only**, which
  is also what `xander-mcp` exposes.
- **Workspace scope** — Xander may work in any explicitly opened directory,
  including project, AI, system-configuration, and package-manager workspaces;
  only Xander-owned records remain inside ASKAR. Package/toolchain setup is
  `ask` by default and can be explicitly set to `allow`; removals and
  destructive operations do not inherit that permission.
- **Legacy `DANGEROUS_PATTERNS`** in the root `config.py` (rm -rf, dd, mkfs,
  fastboot flash/erase, mount, sudo, `curl … | sh`, `git push`, fork bombs, …)
  needing explicit confirmation.
- **Bridge** (`xander serve`) — loopback `127.0.0.1` only, bearer token written
  `0600` to `config_dir()/bridge-token`, CORS granted to one named origin,
  `/ask` and `/stats` read-only, `/order` **plan-only unless `"apply": true`**.
- **Power** — a battery genuinely reporting 0 % cancels active work, records the
  stop, and sends one `systemctl poweroff`; missing telemetry is *unknown*,
  never zero.

## Reach beyond the model

- **Online search** (`web.py`) — DuckDuckGo, Wikipedia, PyPI, StackOverflow and
  a page fetcher. Stdlib only, hard-timeouted, empty result instead of raising
  when the world is unreachable.
- **Skills** (`skills.py`) — an SQLite-indexed registry over local skill roots
  (including `~/.claude/skills`), assembled by relevance and context budget;
  Xander can author skills of his own and fold one evidence-linked lesson per
  verified mission into a grouped hub.
- **MCP both ways** — `xander-mcp` serves his tools; `xander remote` calls
  *other* MCP servers' tools.
- **Variants / the army** — clones with their own routing, directives, skill
  groups, theme and memory namespace; lineage sets rank, and `xander army`
  musters leaders, lessons and verified wins.

## Interfaces

```sh
xander                                    # TUI (5 opening styles: desk, compass, chronicle, workshop, resident)
xander run "goal" --accept "pytest -q"    # scriptable CLI (--json for JSONL)
xander plan "goal"                        # proposal-only, no mutations
xander doctor                             # models, routing, paths, skills, repo
xander mission list|show|delete           # readable workspace-scoped history
xander variant / army / stats / hooks     # clones, muster, scoreboard, operator hooks
xander web / remote / serve / mcp         # search, other MCP servers, HTTP bridge, MCP stdio
```

Logs: `Xander/logs/xander.log` (firehose) and
`Xander/logs/<variant>.log` (one clone's stream). `xander oversee` shows the
latest task, chosen routes, delegations, and blockers without opening the TUI.
