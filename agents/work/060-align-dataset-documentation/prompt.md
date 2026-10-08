# Task Prompt

- Ticket: `060-align-dataset-documentation`
- Status: Done
- Aim: Make repository and Paper 001 guides match the AAD-first, config-selectable dataset workflow.
- Scope: `AGENTS.md`, root README, Paper 001 README, implementation README, and concise workflow references; depends on 057–059.
- Changes:
  - Describe AAD as the current primary dataset and VDD as an optional dataset selected in JSON.
  - Remove stale claims that VDD-specific experiments are part of the active study, while noting that historical runs are preserved and are not new comparable evidence.
  - Document the separate custom-search and model-comparison commands and link the relevant configs/tickets.
- Acceptance criteria:
  - `AGENTS.md` and all three READMEs agree on the active dataset, experiment order, and VDD status.
  - Every documented command/config path exists and matches the implemented scripts.
  - No results, performance claims, or historical run files are changed.
- Out of scope: Manuscript edits, code changes, run deletion, and result interpretation.
- Open questions: `None`.
- Verification: Check all linked paths and commands against the implementation, search for contradictory active VDD/Kinetics instructions, and run `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
