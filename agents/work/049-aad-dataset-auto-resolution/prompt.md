# Task Prompt

- Ticket: `049-aad-dataset-auto-resolution`
- Status: Done
- Aim: Let each runner and notebook select a dataset by name; for AAD, reuse a local copy when available or download it from the approved Kaggle source.
- Scope: Paper 001 implementation dataset setup, JSON configs, and notebook integration.
- Changes:
  - Add one reusable `src.dataset_source` resolver and CLI; use the configured local path first and download `sanelehlabisa/abnormal-activities-dataset` only when needed.
  - Put dataset name and optional local path in the model, train, evaluate, and experiment JSON configs. Allow notebook setup to use the same AAD resolver; leave the existing Kinetics-specific path intact.
  - Validate the extracted AAD class-directory layout. Give a clear next step when Kaggle access/authentication is unavailable; never store credentials in the repository.
  - Keep downloaded data local and ignored by Git. Do not alter the Kinetics diagnostic source or its behavior.
- Acceptance criteria:
  - A valid local AAD directory is reused without a network request.
  - If the configured local copy is absent, the resolver downloads and locates the approved AAD dataset, then reports the resolved path and source.
  - Unknown dataset names, malformed downloads, and missing Kaggle access fail with concise actionable messages.
  - The modular scripts and notebooks use the shared resolver for AAD rather than separate AAD lookup/download logic; Kinetics behavior is unchanged.
  - README commands show how to prepare/use AAD, and dataset files remain untracked.
- Out of scope: Kinetics downloads, dataset relabeling/preprocessing changes, training runs, and model changes.
- Open questions: `None`.
- Verification: Run focused resolver tests with mocked downloads; exercise local-path resolution against the available AAD copy; validate config JSON and script help; run relevant pytest checks; confirm dataset paths are ignored by Git.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
