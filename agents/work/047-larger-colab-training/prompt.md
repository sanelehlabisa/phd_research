# Task Prompt

- Ticket: `047-larger-colab-training`
- Status: Done
- Aim: Run notebook 03 with a larger, bounded Kinetics training profile on Colab.
- Scope: Paper 001 `03_train_model.ipynb`, `notebooks/utils/` helpers,
  `src/train.py` only where needed for the time limit, tests and README guides.
- Context: Master `c44b032` contains the helpers under `notebooks/utils/`.
  The reported failure came from an older export checking removed `src/` paths;
  all four current notebooks pass static helper-path and code-syntax checks.
- Changes:
  - Use the current notebook/helper layout; document reopening and re-exporting
    notebook 03 after pulling master. Updating Colab's checkout does not update
    an already-open notebook or previously exported script.
  - Add a notebook-03-only training profile: `CustomConvLSTM` layers
    `32 -> 64 -> 64`, all `3x3`; 32 RGB frames at 8 FPS, `96x96`, batch 8,
    seed 42, at most 200 epochs and eight hours of training compute.
  - Keep the current five Kinetics-600 classes, all usable videos, the existing
    more-than-2,000 unique-video gate and source-grouped split. Larger inputs,
    not extra classes or a different dataset, are the agreed change.
  - Retain the existing optimizer, gentle learning-rate schedule, dropout and
    weight decay. Keep preview, experiment-suite and controlled AAD settings
    isolated from the new single-model profile.
  - Reuse the existing deadline mechanism where practical. Start its eight-hour
    clock after preparation, check between batches/epochs and save usable
    history/checkpoints on a handled timeout. In-flight work may overrun the cap;
    partial training must be labelled partial, never reported as 200 epochs.
  - Print the resolved profile and parameter count; retain live loss/accuracy
    curves, validation-loss checkpoint selection and five validation predictions
    from a compatible selected checkpoint. Keep test locked throughout.
  - Preserve saved notebook outputs, scratch copies, user training configs,
    run artifacts, reference notebook and manuscript claims.
- Acceptance criteria:
  - Fresh notebook/export uses existing `notebooks.utils` helpers, not removed
    `src.notebook_*` imports; setup remains safe for local Colab checkout edits.
  - Notebook 03 resolves the agreed model/input/budget without changing 01/02,
    notebook 04's suite or controlled AAD configuration.
  - Completed and budget-limited paths retain valid provenance and do not access
    test; predictions require a completed validation-selected checkpoint.
  - Tests and static checks pass; GPU execution remains explicitly unverified
    unless actually run. No accuracy or eight-hour completion guarantee.
- Out of scope: New datasets/classes, pretrained models, changing the experiment
  protocol, manuscript results, long training/downloads locally, commit or push.
- Open questions: None; model/input/budget confirmed by the user.
- Verification: Run focused bootstrap/config/workflow/deadline tests with tiny
  synthetic data; validate notebook JSON, compile code and freshly concatenated
  exported cells, check helper paths, compare protected files and outputs, and
  run `git diff --check`. Run the full local suite if resources permit.

## Execution Prompt

Complete ticket `047-larger-colab-training`. Follow `AGENTS.md`,
`agents/rules.md`, `agents/config.md` and this approved prompt. Preserve saved
outputs and unrelated work, isolate the larger notebook-03 training profile,
keep test locked, verify the result, update this status and create
`completion.md` from `agents/templates/completion.md`.
