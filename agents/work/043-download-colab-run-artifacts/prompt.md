# Task Prompt

- Ticket: `043-download-colab-run-artifacts`
- Status: Done
- Aim: Let the Colab experiment notebook download one archive of its run artifacts when the suite completes or reaches its time budget.
- Scope: Paper 001 implementation; the experiment notebook and its shared suite helper.
- Changes:
  - Create one ZIP for the current suite only, including model configurations, histories/loss curves, metrics, plots, checkpoints, summaries, and generated prediction outputs.
  - On successful completion or the suite's handled time-budget timeout, trigger a Colab browser download and retain the original run folders in `runs/`.
  - Keep incomplete runs labelled partial; never unlock later stages or test evaluation because an archive was created.
  - Do not include the downloaded source dataset or cached dataset archives. No Drive mounting or extra cloud storage.
- Acceptance criteria:
  - A complete suite produces and downloads one archive with its available model/run artifacts.
  - A budget timeout produces and downloads a partial archive of completed and in-progress saved artifacts.
  - The archive is limited to the current suite and does not change or delete source run files.
  - Existing validation/test guards and artifact provenance remain unchanged.
- Out of scope: Helper-module relocation, changes to experiment design, cloud storage backup, and recovery from abrupt VM termination before a notebook cell can handle it.
- Open questions: `None`.
- Verification: Focused tests for archive contents and timeout behavior; notebook JSON/code-cell validation; existing relevant pytest tests; `git diff --check`. Actual browser download requires a Colab run.

## Execution Prompt

Complete ticket `043-download-colab-run-artifacts`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the approved changes, verify every acceptance criterion, set the ticket status to `Done`, and create `completion.md` from `agents/templates/completion.md`.
