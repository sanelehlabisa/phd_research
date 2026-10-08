# Task Prompt

- Ticket: `056-local-aad-experiment-smoke`
- Status: Done
- Aim: Simplify the modular experiment command around flat JSON profiles for a fast local check and the AAD A100 screen.
- Scope: Paper 001 experiment runner, local/A100 JSON profiles, focused tests, and concise run instructions. Preserve `aad_staged_experiments.json` for the full two-seed/notebook workflow.
- Changes:
  - Make study JSON selectable with one `--config` argument; store `models`, `data_sizes`, `epochs`, and `weight_decays` as top-level lists.
  - Keep the local profile to two tiny custom ConvLSTMs, 8 frames at 32×32, two epochs and one weight-decay value (2 configurations).
  - Provide an A100 profile with three custom ConvLSTMs plus `r3d_18`, four `[frames, frame_size]` pairs (8/16 frames at 32/48 pixels), 16 epochs, and two weight-decay values (32 configurations).
  - Rank only by validation results; keep test locked. Label the local profile as a pipeline smoke check and the A100 profile as a single-seed screen, not final paper evidence.
  - Preserve run configs, histories, checkpoints, metrics, confusion matrices and prediction examples. Save a validation-selected config that can be used by the existing training/evaluation scripts for a later longer run.
  - Document the short commands to retrain the selected config for longer and evaluate its frozen training run with the existing scripts; test evaluation must occur only after selection is frozen.
  - Keep old AAD config files that are still referenced by notebook helpers, plus unrelated Kinetics and train/evaluate configs. Do not edit notebook files in this task.
  - Retain the related plan-runner working-directory fix and its regression test.
- Acceptance criteria:
  - A single config command can list and run each matrix without extra per-model/factor CLI arguments; the local and A100 counts are 2 and 32.
  - Tests verify the declared Cartesian run count, fixed split and validation-only selection; no configuration changes the test partition.
  - The selected config and run artifacts are sufficient to continue with existing `src.train` and `src.evaluate` commands.
  - Notebook-referenced configs and Kinetics workflow remain valid.
- Out of scope: Notebook edits, full A100 study, manuscript updates, changing the AAD split, and claiming smoke-grid results as paper evidence.
- Open questions: `None`.
- Verification: Focused experiment tests, config/list-plan validation, script help, `git diff --check`, and the implementation pytest suite. Do not launch the full matrix on a CPU-only host.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
