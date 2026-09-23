# Completion

- Status: Done
- Summary: Standardized all four classification entry points on unique, local,
  purpose-specific run directories with reusable manifests and evidence.
- Changes:
  - Added shared run creation, lifecycle JSON, environment and Git metadata,
    safe filenames, strict JSON writing, prediction collection, and labelled
    confusion-matrix PNG/JSON export in `src/utils.py`.
  - Migrated model smoke, training, evaluation, and comparison outputs to the
    shared `--runs_dir` interface and stable files below their owning run.
  - Added training history and best-checkpoint evidence, evaluation checkpoint
    references, and per-model comparison metrics and final checkpoints.
  - Ignored `runs/` and updated root, paper, and implementation guidance and
    copy-paste AAD commands without changing legacy artifact locations.
- Verification:
  - Black check, Python compilation, and Pylint error-only pass succeeded for
    all five modified Python files.
  - All four parsers expose `--runs_dir` with default `runs` and no legacy
    output-root option.
  - Temporary helper checks passed for purpose separation, unique leaves,
    strict JSON, environment/Git metadata, lifecycle completion, artifact
    references, and exact synthetic confusion-matrix class ordering.
  - The two-sample AAD model smoke completed under `runs/model/<run>/` with two
    PNG/JSON confusion-matrix pairs, four viewable videos, a completed manifest,
    and no checkpoint; Git reports the run as ignored.
  - Source inspection confirmed train, evaluate, and experiments write only
    below their owning run; full runs were not started as required.
  - `git diff --check` passed; legacy model outputs and ticket-009 stash remain
    in their pre-existing state.
- Remaining issues: Full experimental reproducibility still depends on planned
  tickets 015–017 for fixed splits, full seeding, selection, metrics, and config.
