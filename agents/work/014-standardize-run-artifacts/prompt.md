# Task Prompt

- Ticket: `014-standardize-run-artifacts`
- Status: Done
- Aim: Give every model, training, evaluation, and comparison invocation a
  unique local run directory containing consistently named, reusable evidence.
- Scope:
  - `.gitignore`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/utils.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/model.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/train.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/evaluate.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiments.py`
  - root, paper, and implementation guidance affected by commands or task status
- Changes:
  - Use one local `runs/` root with purpose directories `model/`, `train/`,
    `evaluate/`, and `experiments/`. Under each purpose, create a unique leaf
    named with local date/time, dataset, and a concise model or study label.
  - Add the implementation `runs/` directory to `.gitignore`. Keep all run
    artifacts local; commit only code, documentation, and ticket records.
  - Replace the separate output-root options with the same `--runs_dir` option,
    defaulting to `runs`, across all four executable modules. Keep input
    checkpoint arguments separate from output locations.
  - Centralize safe run-directory creation, filename sanitization, JSON writing,
    Git-revision lookup, runtime environment capture, and confusion-matrix data
    export in `utils.py` instead of reimplementing them in each script.
  - Create `run.json` at invocation start and update it on normal completion.
    Record purpose, status, start/end time, duration, exact CLI arguments,
    dataset path/name, class names, model configuration, input dimensions,
    augmentation setting, split ratios, seed when available, device, Python,
    PyTorch/torchvision versions, Git revision, and artifact paths. Do not
    invent unavailable values.
  - Under `runs/model/<run>/`, keep the random-weight warning, resolved model
    configurations, parameter counts, prediction records, per-model `correct/`
    and `incorrect/` videos, and confusion matrices as both PNG and JSON.
    Do not save random-weight checkpoints.
  - Under `runs/train/<run>/`, save the resolved configuration, complete
    per-epoch loss/accuracy/learning-rate history as JSON, best checkpoint and
    checkpoint metadata, training curves, final available metrics, confusion
    matrix as PNG and JSON, and prediction videos. Use stable filenames inside
    the already unique run directory.
  - Under `runs/evaluate/<run>/`, save resolved configuration, checkpoint
    reference, metrics, confusion matrix as PNG and JSON, and prediction videos.
  - Under `runs/experiments/<run>/`, save the shared configuration and ranked
    summary, then one subdirectory per registered model containing its metrics,
    parameter count, runtime, confusion matrix PNG/JSON, and a clearly labelled
    final checkpoint. Do not call it a best checkpoint until ticket 016 fixes
    validation checkpoint selection.
  - Ensure every classification path produces a viewable confusion-matrix PNG
    and raw matrix values with the class-label order in JSON so plots can be
    recreated later.
  - Update all copy-paste AAD commands and concise guidance to use `--runs_dir`
    and explain where each command writes its outputs.
  - Do not move, rename, delete, commit, or reinterpret legacy `models/`,
    `outputs/`, `experiments/`, current JSON, curves, or checkpoints. The new
    structure applies only to invocations made after this ticket.
- Acceptance criteria:
  - The four executable modules use the same `--runs_dir` interface and never
    write new artifacts to the legacy output roots by default.
  - Concurrent or repeated invocations cannot overwrite one another; each gets
    a purpose-specific, timestamped run directory.
  - Run JSON contains the available configuration, environment, lifecycle, and
    artifact references needed to identify and reuse the run.
  - Training history is stored as machine-readable per-epoch values sufficient
    to recreate learning curves.
  - Model smoke, training, evaluation, and comparison classifications each save
    confusion matrices as PNG plus raw labelled JSON.
  - Checkpoints are stored and labelled consistently: best for training, final
    for the current comparison runner, and none for random-weight model smoke or
    evaluation-only runs.
  - Prediction videos and plots are contained inside their owning run directory.
  - `runs/` is ignored by Git, while all existing legacy artifacts remain
    byte-for-byte untouched and retain their current tracked/untracked state.
  - Active Python files compile and `git diff --check` reports no errors.
- Out of scope:
  - Running full training, evaluation, or model-comparison experiments.
  - Migrating, deleting, renaming, committing, or repairing legacy artifacts.
  - Changing models, augmentation, split membership, seeds, losses, optimizers,
    scheduler behaviour, metric definitions, or model-selection policy.
  - Implementing the fixed split manifest, stateful streaming, or manuscript
    result changes; later tickets own those items.
- Open questions: None.
- Verification:
  - Compile every modified Python file and check formatting.
  - Exercise the shared run helper under a temporary directory and verify
    purpose separation, unique leaf names, JSON serialization, environment and
    Git metadata, lifecycle updates, and declared artifact paths.
  - Run parser checks for all four modules and confirm the shared `--runs_dir`
    option replaces legacy output-root options.
  - Use small synthetic labels to verify confusion-matrix PNG/JSON pairs and
    exact class-label ordering without training.
  - Run the small AAD model smoke command and confirm all its outputs stay under
    one ignored `runs/model/<run>/` directory.
  - Inspect the training, evaluation, and comparison paths to confirm every
    declared output is below its owning run directory; do not start them.
  - Confirm models, augmentation, splits, seeds, training calculations,
    manuscript, legacy artifacts, and the ticket-009 stash are unchanged.
  - Run `git diff --check`, review the final diff for unrelated changes, and
    confirm `git status --ignored` shows new run outputs only as ignored files.

## Execution Prompt

Execute ticket `014-standardize-run-artifacts` exactly as written in
`agents/work/014-standardize-run-artifacts/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
