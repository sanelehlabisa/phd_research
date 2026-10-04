# Task Prompt

- Ticket: `035-vdd-learnability-check`
- Status: Blocked — implementation verified locally; Colab VDD run required
- Aim: Determine whether the current ConvLSTM and training pipeline can learn a small, clearer binary video task before spending A100 time on controlled AAD experiments.
- Scope: Paper 001 implementation, Colab notebook, and the Kaggle dataset `sanelehlabisa/violence-detection-dataset`.
- Changes:
  - Inspect and record the VDD directory structure, class names, clip counts, formats, corrupt files, duration distribution, and duplicate risks before training.
  - Reuse the existing dataset, model, metric, seed, checkpoint, and run-artifact code; add only the smallest dataset adapter or configuration needed for VDD.
  - Add a guarded VDD diagnostic path separate from the controlled AAD plan.
  - First overfit one fixed, balanced tiny subset to test model and optimizer learnability.
  - Only if the tiny-subset check succeeds, run one bounded stratified train/validation diagnostic on VDD with test data locked.
  - Plot loss and accuracy and retain the resolved configuration, split, seed, history, checkpoint, and environment provenance.
  - Label every output `pipeline_learnability_diagnostic`, never comparative or manuscript evidence.
- Acceptance criteria:
  - Dataset inspection completes and the split contains no path overlap.
  - The tiny fixed subset reaches at least 95% training accuracy with a clear loss decrease, or the task records a reproducible failure with gradient, label, prediction, and batch diagnostics.
  - After a successful overfit check, the bounded VDD run shows whether training accuracy can exceed 70%; failure to reach 70% is reported, not hidden or retried with an unplanned search.
  - Validation accuracy is reported separately; no requirement assumes it begins at 0% because random binary accuracy is near 50%.
  - Test clips remain unopened and ticket 021 plus the controlled AAD plan remain unchanged.
- Out of scope: Ranking architectures, tuning until a favourable result appears, final VDD generalisation evidence, Kinetics training, opening test data, or changing manuscript claims.
- Open questions: None.
- Verification: Validate the VDD inventory and split; run the tiny-subset overfit check; if it passes, run the bounded train/validation diagnostic; inspect histories and predictions; run focused tests and `git diff --check`.

## Execution Prompt

Complete ticket `035-vdd-learnability-check`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Preserve the controlled experiment guards, keep test data locked, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
