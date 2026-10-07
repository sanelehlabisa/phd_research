# Completion

- Status: Done
- Summary: Reduced the controlled AAD architecture screen to three evidence-led candidates and made its active stage selectable from the plan JSON.
- Changes:
  - Grouped runner configs under `configs/train/`, `configs/evaluate/`, and `configs/experiments/` and updated implementation, test, notebook, and README paths.
  - Set `[32, 16]` as the controlled reference, retained `[16, 32]` and `[32, 16, 8]`, and aligned confirmation/ablation references.
  - Matched the screen to the historical high-validation profile: seed 42, 8 frames at `64x64`, augmentation off, batch 16, LR `0.01`, WD `0`, patience 20, maximum 160 epochs.
  - Preserved Notebook 04's original 11 custom Kinetics candidates in its own manifest, keeping its 14-model suite intact.
- Verification:
  - `MPLCONFIGDIR=/tmp/phd-mpl-cache .venv/bin/python -m pytest tests -q --tb=short` — 144 passed.
  - `MPLCONFIGDIR=/tmp/phd-mpl-cache .venv/bin/python -m src.experiments --plan-config configs/experiments/aad_controlled_experiment_plan.json --list-plan` — reports the three candidates, active architecture-screen stage, approved profile, and retained baseline stages; no training started.
  - All config JSON files parse; `git diff HEAD --check` passes. A test verifies that `--plan-config` alone dispatches the JSON-selected stage.
- Remaining issues: No training or evaluation was run; the screen remains pending on the user's compute environment.
