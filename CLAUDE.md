# CLAUDE.md

@AGENTS.md

AGENTS.md is the single source of truth for this repo. Edit rules there, not here, so all agents stay in sync.

## Claude-specific notes

- Before any change to `control/`, `agent/`, or `serve/`, re-read AGENTS.md sections 2, 7 and 8.
- Use plan mode for changes that touch more than one tier or the validator rules.
- Never run commands that create, start, or delete Nebius resources (`nebius ai endpoint create`,
  job submissions, etc.) or that spend Token Factory credits without the maintainer's explicit OK.
  Mocked tests and local runs are always fine.
- Don't commit or push unless asked. Commit messages: imperative, ≤ 72-char subject, reference the tier
  or module (e.g. `agent: add confidence blend to validator`).
- When reporting results, separate measured numbers from estimates, and name the hardware.