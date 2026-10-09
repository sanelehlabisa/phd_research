# Task Prompt

- Ticket: `067-expand-custom-search-boundaries`
- Status: Done
- Aim: Expand the AAD custom search around the promising `[16,16]` model and test whether the validation-selected architecture and input remain competitive with longer training and a second training seed.
- Scope: Paper 001's active AAD custom-search JSON, bounded study validation/runner, Colab search workflow and Python export, focused tests, and implementation README. Keep model-family comparison and Kinetics workflows unchanged.
- Changes:
  - Keep the current eight custom candidates and add exactly four: `[8,16]`, `[24,16]`, `[16,24]`, and `[16,16,16,16]`. Raise the bounded custom-candidate limit to 12; do not turn this into an open-ended model grid.
  - Screen all 12 candidates with seed `42`, 8 frames at `32x32`, augmentation off, weight decay `0`, and one measured batch size held constant. Raise the epoch cap from 64 to 128 and validation-loss patience from 12 to 16.
  - After validation ranks the architecture, vary only that candidate's input one factor at a time: frame sizes `[32,48,64]` and frame counts `[8,16]`. Do not run a full Cartesian product.
  - Repeat the validation-selected architecture/input once with model seed `2026`, retaining the same seed-42 split manifest, data, batch size, and other training settings. Record both seed results; keep test locked.
  - Keep caching, the measured batch-size choice, search-only default, progress logging, artifact archive, and explicit later comparison from ticket 066. Include the selected configuration, both-seed confirmation, checkpoints, metrics, logs, and prediction examples in the archive.
  - Keep the notebook and Python export aligned. State that this is a finite, validation-guided search—not proof of a global optimum—and document the 16-run maximum.
- Acceptance criteria:
  - The validated profile declares exactly 12 custom candidates and a maximum of 16 jobs: 12 architecture screens, three one-factor input checks for the validation winner, and one seed-2026 confirmation.
  - The 12 architecture screens use seed `42`, the fixed AAD split, 8×32×32 inputs, no augmentation, zero weight decay, one common measured batch size, a 128-epoch cap, and patience 16.
  - Only the validation-selected candidate receives the 48/64 frame-size and 16-frame checks; no full Cartesian expansion occurs. The additional-seed run uses the selected input and the unchanged seed-42 split manifest.
  - The selected model, parameter count, validation metrics for both seeds, input configuration, split hash, run progress, and run artifacts are saved and included in the downloadable archive. Search never reads test data or automatically starts model-family comparison.
  - Notebook/export parity, config and plan validation, candidate-limit and run-count tests, cache/split/test-lock checks, explicit-comparison behavior, focused tests, local smoke, and `git diff --check` pass. No generated run artifacts are committed.
- Out of scope: Changing comparison models or protocol, changing the dataset or split, adding augmentation/weight-decay/optimizer sweeps, editing manuscript claims, or interrupting/reconfiguring an already-running Colab study.
- Open questions: `None`.
- Verification: Validate the active profile and exact plan size; test selected-winner-only input checks and seed-2026 same-split confirmation; run focused pytest and local smoke; parse notebook JSON and verify Python-export parity; confirm search-only archiving and explicit comparison; run `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
