# Task Prompt

- Ticket: `059-retire-vdd-specific-workflow`
- Status: Done
- Aim: Remove VDD-only diagnostic workflow code while retaining VDD as an optional dataset selected through JSON.
- Scope: Paper 001 VDD-specific diagnostics/helpers, their tests and notebook references; depends on 057–058.
- Changes:
  - Remove the dedicated VDD diagnostic path from active implementation/notebook workflows where the generic configured-dataset pipeline now covers the need.
  - Keep VDD supported through the same dataset configuration/resolution path as AAD; do not hard-code two VDD class names or make VDD the default.
  - Preserve all existing VDD runs and local data; do not delete generated artifacts or rewrite manuscript claims here.
- Acceptance criteria:
  - A user can still select a valid VDD directory in JSON and use the generic pipeline without VDD-specific code.
  - No active notebook or CLI invokes the retired VDD-only diagnostic workflow.
  - Historical run folders are unchanged and remain available.
- Out of scope: Removing VDD as a selectable dataset, deleting old runs/data, documentation/manuscript edits, and new training runs.
- Open questions: `None`.
- Verification: Search for VDD-only imports/entry points, run relevant tests and notebook reference checks, validate JSON, and run `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
