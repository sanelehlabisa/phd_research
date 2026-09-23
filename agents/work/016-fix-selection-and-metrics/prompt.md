# Task Prompt

- Ticket: `016-fix-selection-and-metrics`
- Status: Done
- Aim: Produce valid partition-level classification metrics, restore the
  validation-selected checkpoint, and prevent model development from accessing
  the test split.
- Scope:
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/metrics.py`
    if a small shared metric helper keeps the three commands consistent
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/train.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/evaluate.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiments.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/utils.py`
    only if shared checkpoint or metric support belongs there
  - root, paper, and implementation guidance affected by status, commands, or
    artifact descriptions
- Changes:
  - Replace averages of per-batch loss and stateful per-batch accuracy with
    sample-weighted loss and metrics computed once across the full partition.
    Report accuracy and macro precision, recall, and F1 with explicit resets
    between partitions, epochs, and models.
  - Use validation loss as the single checkpoint and early-stopping criterion.
    Add `--early_stopping_patience` with default `10`; stop after ten consecutive
    epochs without a lower validation loss and record the reason, selected
    epoch, and actual epoch count.
  - In `train.py`, save sufficient checkpoint metadata, reload the lowest-loss
    validation checkpoint after training, and report its validation metrics.
    Do not construct, iterate, score, plot, or sample the test partition.
  - In `experiments.py`, save and restore each model's lowest-validation-loss
    checkpoint. Rank the restored checkpoints by validation macro-F1, then
    validation accuracy, then fewer trainable parameters. Remove the heuristic
    overfit gate from ranking and do not access or report test results.
  - Keep validation history and clearly label validation confusion matrices or
    other diagnostic artifacts so they cannot be mistaken for test evidence.
  - Make `evaluate.py` the only command that accesses the test partition. Require
    an explicit compatible checkpoint and refuse random-weight evaluation;
    report sample-weighted test loss plus full-partition accuracy and macro
    precision, recall, and F1.
  - Mark checkpoints with their role, selection metric/value, selected epoch,
    dataset, split-manifest hash, model configuration, and run seed. Propagate
    that provenance into run metadata and result JSON.
  - Update concise commands and guidance to state that training/comparison use
    train and validation only, while evaluation is the deliberate final test
    step after a configuration is frozen.
- Acceptance criteria:
  - Uneven batch sizes produce the same loss and classification metrics as a
    direct full-partition calculation; metrics do not retain state across calls.
  - Training and comparison select the exact epoch with the lowest validation
    loss, restore those weights, and record matching checkpoint provenance.
  - Early stopping triggers only after the configured number of consecutive
    non-improving validation epochs and records the actual stopping epoch.
  - Candidate ranking uses only restored-checkpoint validation macro-F1,
    validation accuracy, and parameter count; no test metric or overfit
    heuristic affects ordering.
  - Train and experiments never iterate the test subset. Evaluate refuses a
    missing, random-weight, incompatible, or non-validation-selected checkpoint
    before test inference.
  - Existing model architectures, split membership, frame sampling,
    augmentation, optimizer, scheduler, and manuscript results remain unchanged.
  - Run JSON, configuration, history, checkpoint metadata, and result artifacts
    identify metric averaging, selected epoch, selection rule, early-stopping
    state, split manifest, and seed.
  - Active Python files compile, formatting and Pylint error-only checks pass,
    and `git diff --check` reports no errors.
- Out of scope:
  - Running full training, selecting the final architecture, or publishing new
    test results.
  - Changing the fixed split, architecture registry, augmentation policy,
    optimizer, scheduler, loss definition, input dimensions, or training budget.
  - Connecting the stashed ticket-009 configuration, aggregating multiple
    seeds, rewriting manuscript results, or adding streaming inference.
  - Moving, deleting, committing, or interrupting local datasets, checkpoints,
    ignored runs, active training jobs, or the ticket-009 stash.
- Open questions: None.
- Verification:
  - Format and compile every modified Python file; run Pylint error-only and
    `git diff --check`.
  - Test the shared metric path on synthetic logits and labels with unequal
    batch sizes, including two consecutive calls to prove state isolation.
  - Exercise checkpoint selection and early stopping on a tiny deterministic
    synthetic model/history without decoding videos or running a full model.
  - Verify a deliberately better earlier epoch is restored and its stored
    validation metrics match a fresh calculation.
  - Use test-loader sentinels or equivalent focused checks to prove train and
    experiments cannot consume test samples; verify evaluate requires and
    validates a selected checkpoint before its test loop.
  - Test ranking with synthetic tied candidates to confirm macro-F1, accuracy,
    and parameter-count ordering without any test fields.
  - Inspect temporary run/checkpoint JSON for the required provenance and clear
    validation/test labels.
  - Inspect the final diff to confirm the manifest, model topology, data policy,
    optimizer, scheduler, loss, and manuscript are untouched; ignored artifacts,
    active jobs, and the ticket-009 stash remain unchanged.

## Execution Prompt

Execute ticket `016-fix-selection-and-metrics` exactly as written in
`agents/work/016-fix-selection-and-metrics/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
