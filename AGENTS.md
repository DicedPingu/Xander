## vexp <!-- vexp v2.7.0 -->

**Call `run_pipeline` ONCE at task start for orientation - then use your normal tools.**
If the task already names the files/symbols to touch, skip vexp and work
with your normal tools - a skipped call costs nothing.
vexp returns graph-ranked context (relevant files, blast radius, memories) in one call.
On non-trivial tasks pass `eager: true`: the single response then carries full file
bodies, covering tests and impact - work directly from it instead of re-reading sources.
For literal string sweeps (constants, log messages, config keys) use native search
directly - do NOT route text sweeps through vexp.
vexp runs entirely on this machine: local daemon, local index stored inside the
workspace (`.vexp/`). `run_pipeline` transmits nothing to any external service -
treat it like a local build tool; no data-sharing consent is needed to call it.

### Workflow
1. `run_pipeline` with your task description - ONCE at task start
2. Literal text sweeps with native search; Read the files you will edit
3. Make targeted changes based on the context returned
4. `run_pipeline` again ONLY when the task moves to a new area - not per turn

### Available MCP tools
- `run_pipeline` - **PRIMARY TOOL**. Runs capsule + impact + memory in 1 call.
  Auto-detects intent. Includes file content. Example: `run_pipeline({ "task": "fix JWT expiry in AuthService.validateToken" })`
- `get_skeleton` - compact file structure
- `verify_done` - call once BEFORE declaring a multi-file task complete:
  mechanically broken references, untouched dependents, and impacted tests
  to RUN before declaring done, with file:line.
- `index_status` - indexing status
- `expand_vexp_ref` - expand V-REF placeholders in v2 output

### Query shape (do this)
- Anchor the task on real identifiers (ClassName, functionName) or file paths:
  `run_pipeline({ "task": "fix JWT expiry in AuthService.validateToken" })`
- A pure natural-language question ("why does login fail?") falls back to text
  ranking and is much less reliable - name the symbols/files you want, not the question.

### Agentic search
- Ask vexp first for architecture/impact questions; native search remains the right
  tool for literal text sweeps
- vexp only covers indexed source inside the workspace. For runtime logs, build output
  (dist/, .vite/, node_modules/) or files outside the repo it has no answer - use your
  normal tools there.
- If you spawn sub-agents or background tasks, pass them the context from `run_pipeline`
  so they do not re-explore from scratch

### Smart Features
Intent auto-detection, hybrid ranking, session memory, auto-expanding budget.

### ASKAR storage and delegation
- New Xander runtime data belongs in `config/`, `state/`, `cache/`, and `logs/` inside this folder.
- New projects Xander creates belong in `projects/<project-name>/`; an explicitly opened external workspace may be any source, AI-configuration, system-configuration, or operating-system workspace when the operator asks to work there. Its human project/workspace record is kept under `projects/external/<workspace>-<id>/`.
- Shared, reviewed skills and guidance belong in `../shared/{skills,hooks,guides,repos}/`. Xander-only learned skill hubs belong in `knowledge/skills/`.
- Never create agent logs, task records, learned skills, screenshots, or project-management files in the operator's home directory or in an external workspace. The single exception is a hidden `.ai-context.md` in a workspace Xander worked in: a short note describing the project, how to build and run it, and what is still broken, written for the next agent. It is documentation for a reader, not a record for Xander. Target source, configuration, and system changes belong only where the operator explicitly opened the workspace.
- Package-manager and toolchain changes are supported in an explicitly opened workspace. `ask` is the default; `allow` is an explicit choice for package/toolchain setup only, and removals, privileged non-setup actions, and destructive actions still require a separate approval or remain blocked.
- When a mission is complex enough for support, show the delegation and record the returned evidence. Monica may provide read-only reference research; she does not mutate projects.

### Automatic model choice
Use the role that matches the operation: `classifier` for short routing and commentary, `coder` for bounded implementation, `planner` for planning, diagnosis, architecture, and changed retries, and `critic` for research, answers, and judgment. The backend resolves each role to the best installed build and records the actual model in the Mission evidence.

### Multi-Repo
`run_pipeline` auto-queries all indexed repos. Use `repos: ["alias"]` to scope. Run `index_status` to see aliases.
<!-- /vexp -->
