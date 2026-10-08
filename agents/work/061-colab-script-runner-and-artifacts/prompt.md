# Task Prompt

- Ticket: `061-colab-script-runner-and-artifacts`
- Status: Ready
- Aim: Make the experiment notebook easy to export as a Python script and run in one Colab cell using the modular workflows.
- Scope: Active Paper 001 Colab experiment notebook, its generated/exported Python script if repository practice requires it, minimal artifact helper, and implementation guide; depends on 057–060.
- Changes:
  - Keep the notebook as the source of truth and make it a thin orchestrator for dataset resolution, GPU/runtime check, custom search, model comparison, and result summaries; do not duplicate training logic.
  - Ensure the exported Python script follows the same stages and can run as one Colab UI cell with back execution; use the JSON configs as the only experiment settings source.
  - Remove the project-imposed wall-clock cutoff. Preserve progress and completed run artifacts at stage boundaries and provide one final ZIP download to the local computer.
  - State clearly that Colab can still terminate a runtime externally; do not promise an end-of-run download after forced termination.
- Acceptance criteria:
  - Notebook and exported script invoke the same modular commands and config files.
  - AAD is resolved/downloaded by the existing tokenless public-data path when absent; no manual dataset upload is required.
  - Successful completion packages metrics, histories, configs, checkpoints, confusion matrices, and prediction videos from the current run for download.
  - A short notebook/script smoke path runs without launching the full study.
- Out of scope: Full A100 experiment run, Drive synchronization, new model/training logic, and changes to historical artifacts.
- Open questions: `None`.
- Verification: Validate notebook JSON, execute smoke cells or exported-script smoke path where feasible, confirm archive contents with temporary artifacts, check script export consistency, and run `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
