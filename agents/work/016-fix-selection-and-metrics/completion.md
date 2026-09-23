# Completion

- Status: Done
- Summary: Corrected classification metric aggregation, made lowest validation
  loss the checkpoint and early-stopping rule, restored selected weights, and
  reserved test access for the explicit evaluation command.
- Changes:
  - Added shared sample-weighted loss and full-partition accuracy plus macro
    precision, recall, and F1 with fresh state for every call.
  - Added validation-loss selection and ten-epoch default early stopping with
    selected-epoch, actual-epoch, stopping, split, seed, and model provenance.
  - Changed training and comparisons to restore their selected checkpoints,
    write validation-only diagnostics, and never create or iterate test loaders.
  - Changed comparison ranking to validation macro-F1, validation accuracy, and
    then fewer parameters; removed final-epoch test results and the overfit gate.
  - Made evaluation require a compatible validation-selected checkpoint before
    constructing its test loader, then report explicitly labelled test metrics.
  - Updated repository status, protocol guidance, commands, and artifact notes;
    ticket 017 is next.
- Verification:
  - Black check, `py_compile`, Pylint error-only, and `git diff --check` passed.
  - Unequal synthetic batches matched direct full-partition loss and metrics;
    repeated calls proved metric state isolation and explicit macro output keys.
  - Validation selection retained epoch 3 and stopped after exactly three
    consecutive non-improvements in the focused patience test.
  - Deliberately changed later weights were replaced by the saved epoch-2
    checkpoint, whose fresh validation metrics matched its stored metrics.
  - Synthetic train and comparison runs produced valid run, configuration,
    history, result, and checkpoint JSON while test-dataset sentinels remained
    untouched.
  - Ranking tie tests confirmed macro-F1, accuracy, and parameter ordering with
    no test-result fields; evaluation rejected missing and random-weight
    checkpoints before test access.
  - Parser and model-registry checks passed. Model, dataset, split manifest,
    optimizer, scheduler, loss, augmentation, manuscript, ignored artifacts,
    active jobs, and the ticket-009 stash remained unchanged.
- Remaining issues: None.
