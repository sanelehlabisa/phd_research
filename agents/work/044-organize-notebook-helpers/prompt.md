# Task Prompt

- Ticket: `044-organize-notebook-helpers`
- Status: Ready
- Depends on: Ticket 043.
- Aim: Move notebook-only helper modules out of the CLI-oriented `src/` directory without breaking notebook or command-line workflows.
- Scope: Paper 001 implementation helpers, notebook imports, focused tests, and implementation README.
- Changes:
  - Move notebook-specific Python helpers into `implementation/notebooks/utils/` as a small importable package.
  - Update all notebook, bootstrap, and test references to the new location.
  - Keep reusable model, dataset, metrics, training, evaluation, and experiment CLI modules in `src/`.
- Acceptance criteria:
  - Notebook helpers live under `notebooks/utils/`; obsolete copies are absent from `src/`.
  - All four modular notebooks and their Colab setup/bootstrap can import the moved helpers from a fresh checkout.
  - Existing modular CLI entry points still import and run their safe `--help`/listing checks.
  - Relevant tests pass and the README documents the concise directory distinction.
- Out of scope: Behavior changes, experiment-plan changes, artifact-download behavior from ticket 043, and moving core model/training code.
- Open questions: `None`.
- Verification: Search for stale imports; validate notebook JSON/code cells; run focused pytest tests, CLI smoke/listing commands, compile checks, and `git diff --check`.

## Execution Prompt

Complete ticket `044-organize-notebook-helpers`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the approved changes, verify every acceptance criterion, set the ticket status to `Done`, and create `completion.md` from `agents/templates/completion.md`.
