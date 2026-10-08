# Task Prompt

- Ticket: `057-configurable-datasets-and-classes`
- Status: Ready
- Aim: Make modular runs select a dataset in JSON and infer its complete class set from its directory layout.
- Scope: Paper 001 modular dataset, model, training, evaluation, and experiment configuration/resolution code and focused tests.
- Changes:
  - Require dataset name and optional local path in each runnable JSON config; keep AAD as the active profile default and allow VDD to be selected by changing the config. Retain AAD's tokenless public download; for datasets without an approved public downloader, require a configured local path and give a clear error.
  - Derive class names and output dimension from the selected dataset folders; remove hard-coded AAD/VDD class lists from the modular training/evaluation path.
  - Preserve the existing reproducible split, class mapping in run provenance, and clear errors for invalid dataset layouts.
  - Preserve all existing run artifacts; do not change notebooks or manuscript in this task.
- Acceptance criteria:
  - A synthetic dataset with arbitrary class-folder names can be selected and loaded without source-code label edits.
  - AAD remains the default and downloads through its existing tokenless source; selecting a local VDD copy through JSON uses its discovered classes, without changing code.
  - Model output dimensions, metrics, confusion labels, and saved class metadata match the discovered classes.
  - Focused tests cover both dataset layouts and reject empty/malformed layouts.
- Out of scope: Running training, removing VDD-specific notebook/diagnostic helpers, documentation rewrite, and deleting or editing historical runs.
- Open questions: `None`.
- Verification: Focused dataset/config/model tests, JSON validation, script help, `git diff --check`, and relevant implementation tests.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
