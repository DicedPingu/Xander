# Xander projects

New projects created without an explicitly opened workspace belong here, one
project per subfolder. Keep source files, `PROJECT.md` or `MISSION.md`, and a
`logs/` directory together. Detailed task logs use one Markdown file per task
under `logs/`; include changes, evidence, problems, resolutions, and open
questions.

If the operator explicitly opens a workspace outside ASKAR, Xander may change
its requested source or configuration there, including package/toolchain setup
when explicitly allowed. Task state and global agent logs remain in
`ASKAR/Xander/state/` and `ASKAR/Xander/logs/`; a readable `PROJECT.md` and
detailed task logs are mirrored under `Xander/projects/external/` so the target
never receives Xander management files.
