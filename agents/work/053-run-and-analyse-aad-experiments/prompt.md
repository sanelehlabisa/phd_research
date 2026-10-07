# Task Prompt

- Ticket: `053-run-and-analyse-aad-experiments`
- Status: Ready
- Aim: Run the current controlled AAD study and analyse validation evidence for final model selection.
- Scope: Paper 001's `aad_staged_experiments.json`, existing runner, local run artifacts and concise analysis notes.
- Changes:
  - Supersede ticket 025's old run matrix; use ticket 052's single JSON on the user's Colab A100, not the current CPU-only host.
  - Validate and record the plan, code revision, split, seeds and resource budget before launching. Do not expand compute or search dimensions without approval.
  - Run the seven-model/two-seed screen, then the declared one-factor comparisons; keep native PaperConvLSTM separate and opt-in.
  - Keep test locked. Preserve each run's configuration, history, validation-selected checkpoint, prediction examples, parameter count and runtime.
  - Analyse learning curves, validation accuracy/loss and overall micro metrics, seed mean/standard deviation, efficiency and individual factor effects.
  - Recommend a frozen architecture/input/regularisation configuration for ticket 026 using validation only; validate any combined factors before adopting them.
- Acceptance criteria:
  - Every declared job is accounted for; incomplete/resource-limited evidence is labelled and never treated as a complete screen.
  - Tables and observations link to exact run IDs/configs; historical and diagnostic runs are not mixed into the ranking.
  - A complete comparison and justified selection are available before ticket 026 proceeds.
- Out of scope: Final testing, new datasets, unplanned searches, manuscript claims, committing datasets/checkpoints or generated runs.
- Open questions: None for ticket setup; execution requires the GPU runtime and recorded validation artifacts.
- Verification: No-training plan listing; audit expected jobs, seeds, split/config hashes and checkpoints; recompute analysis from saved JSON; `git diff --check`.

## Execution Prompt

Complete ticket `053-run-and-analyse-aad-experiments` using the approved plan and repository rules. Run only on the available, authorised GPU runtime. Keep test locked, preserve historical evidence, verify the analysis, update status and create `completion.md`. If runtime or complete evidence is unavailable, record the blocker without claiming completion.
