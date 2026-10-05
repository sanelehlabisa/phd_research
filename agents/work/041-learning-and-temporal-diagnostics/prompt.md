# Task Prompt

- Ticket: `041-learning-and-temporal-diagnostics`
- Status: Done
- Aim: Diagnose weak learning with reproducible temporal, architecture and regularization comparisons within an eight-hour Colab budget.
- Approval: User answered the scope questions and explicitly requested implementation now; saved notebook results committed first (`d96533e`).
- Scope: Paper 001 modular notebooks/shared helpers, diagnostic runner, tests and guides.
- Changes:
  - Gentle plateau schedule (factor 0.5, patience 5, floor 1e-5); record learning rates, gradients, updates, runtime and clean training metrics.
  - Timestamp-based seeded training windows and fixed validation windows; audit training/validation FPS, duration, timestamps and repeated frames. Never rewrite source videos.
  - Tiny balanced training-only overfit check, then 11 existing custom candidates and three scratch 3D-CNN baselines; keep the native 50-frame/50x50 paper topology separate and resource-checked.
  - Fixed 32-frame temporal coverage comparisons at 4/8/16 FPS; dropout and weight-decay comparisons change one factor at a time. Record that FPS changes duration too.
  - Freeze validation-selected configuration; longer/finer confirmation across seeds 42/2026, with matched practical-baseline confirmation. Save incremental evidence and explicit partial/resource-limited status; no final test from incomplete runs.
  - Keep all notebook outputs, original reference, controlled AAD protocol, dataset classes and split unchanged. Preserve display recovery and checkpoint safeguards.
  - Export machine-readable results, tables and curves with provenance; existing Kinetics test is previously inspected, so results remain exploratory. No manuscript claims or accuracy promises.
- Acceptance criteria:
  - Run All uses shared Kinetics defaults and the declared suite; source FPS and sampled windows are inspectable.
  - Timing checks enforce an eight-hour cooperative compute deadline; resource failures never silently substitute smaller models or count as completed comparisons.
  - Validation alone determines configurations/checkpoints; test opens only for frozen compatible complete confirmation, with saved-report recovery.
- Out of scope: GPU experiments tonight, stateful streaming, new datasets, pretrained transfer, manuscript rewriting and fresh-holdout claims.
- Open questions: None. Implementation authorized now; no new push authorization inferred.
- Verification: Synthetic-video workflows, temporal/cache determinism, scheduler replay, model factories/counts, tiny-check diagnostics, suite/timeout/test gates, saved-output hashes, full pytest, formatting and diff checks.

## Execution Prompt

Implement the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Preserve saved outputs and existing research artifacts, verify the implementation, update this status and create `completion.md` from the template. Save runnable code for the user's later Colab execution; do not start the real GPU suite.
