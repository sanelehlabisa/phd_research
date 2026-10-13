# Task Prompt

- Ticket: `076-aad-shortlist-comparison`
- Status: Ready
- Aim: Train five custom finalists from a fresh AAD search and five baselines once under a frozen recipe, then evaluate the complete comparison.
- Scope: New `configs/experiments/aad_shortlist_comparison_colab.json`, minimal existing runner/validator support, the focused notebook/export, verification and concise guides. Evidence: [latest Colab report](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-13/latest_colab_results.md).
- Dependency: [077 focused rerun](../077-focused-aad-colab-rerun/prompt.md) and its verified selection format. Execute this implementation after 077; actual results require owner training.
- Context: Revised 13 October 2026: the owner chose a fresh Colab search instead of the console-derived local shortcut. Use new verified receipts/checkpoints, preserving earlier logs. Current comparison accepts only three customs; extend minimally to five.

## Declared comparison

- AAD only, all folder-derived classes, existing seed-42 split: 748/160/161. Verify the manifest against the stored source-review receipt; expected SHA-256 `aada503d63d4433b272237db6cf27d7b1c0eaeb0091d5ba8289343a6111454d8`. Never silently resplit.
- Customs: five distinct finalists from 077's declared validation rule, including `[96,64,64]` as the owner's prespecified historical 99.375% candidate. If it does not rank in the new top five, label its reserved slot and retain its actual rank; never change a measured score or force it to win.
- Baselines: input-adapted paper topology, `r3d_18`, `mc3_18`, `swin3d_t`, `swin3d_s`, all from scratch. Preserve the native published model and clearly label/count the adapted input-dependent classifier.
- Common recipe: eight frames and the shared frame-size/FPS condition frozen by 077; model seed 42, Adam, LR 0.001, weight decay 0, augmentation off, existing scheduler; **512-epoch cap, minimum 128, validation-loss patience 32**. All ten train afresh once. Disclose that the input is selected using customs; no new recipe sweep or claim of per-family optimal tuning.
- Use the existing full training-step preflight to select one safe common batch for the A100; local 6 GB execution is optional only after passing the same check. Preserve the chosen input. Native PaperConvLSTM cannot fit FP32 Adam on 6 GB; the eight-frame/64px adapted model has 134,415,195 parameters and needs measurement. Do not promise a fast full run, skip a model, or silently shrink its input.

## Changes

- Keep the command JSON-only: `python -m src.experiments --config configs/experiments/aad_shortlist_comparison_colab.json`. Use verified 077 selection to generate a resolved self-contained JSON with dataset/split, ten models, layers, frozen recipe, output directory and examples. Preserve legacy readers/profiles; avoid another standalone runner.
- Verify the new selection, complete condition coverage, candidate/config identities and exact split. Keep artifact guards intact; console-only historical results cannot stand in for new receipts. Add the comparison step to 077's focused notebook/export, keeping both aligned and reusing the existing modular runner.
- Train all ten afresh; select checkpoints and the best custom using validation. Freeze every checkpoint before ten complete final test evaluations. Preserve exposure/repeat-test guards and never select by test score.
- Save accuracy, pooled precision/recall/F1, loss, per-class TP/FP/FN/support, parameters, model bytes, timing, full configs, curves, checkpoints and confusion matrices. Print progress and convergence warnings. Retain constant-class/majority-baseline failures as optimization limitations requiring review.
- Reuse existing example-video saving: correct/incorrect folders, validation examples during selection, test confusion matrices for all models and final examples from the validation-selected best custom. Set `prediction_samples_per_category: 5` in JSON and log saved counts.
- Export one ten-row comparison CSV/JSON with traceable artifacts. Save a compact results bundle as models complete and a downloadable final results archive; keep checkpoints available separately and clearly report that browser download completion is unverified. Verify implementation synthetically; owner runs the long training.
- After the owner supplies completed artifacts, review them and fill Table 2 in `reports/2026-10-13/latest_colab_results.md`, or link a dated final report. Preserve raw logs and provisional search results. Then proceed to manuscript ticket 062.

- Acceptance criteria:
  - Profile validates/lists exactly five verified finalists plus five baselines, one fixed input/recipe/seed, and no Cartesian expansion; one JSON command and the focused notebook use the existing runner.
  - Synthetic verification reaches ten trainings, validation freeze, ten guarded tests and complete tables/confusion/example artifacts. Legacy handoff, split/config mismatch, OOM and repeat-test guards still work.
  - Full-step preflight measures memory and reports inability to fit; baseline adaptation/count and convergence limitations are explicit.
  - Focused tests, config/plan validation, links and `git diff --check` pass. No actual long training, invented score, run-artifact commit or manuscript change during implementation.
- Out of scope: Another search/calibration, new datasets/splits, extra standalone scripts, rewriting legacy notebooks, filling test dashes without outputs, commit or push.
- Open questions: `None` for implementation after 077. GPU capacity/runtime and convergence must be measured; owner execution and final report review remain pending.
- Verification: Five-custom and legacy schema/handoff checks; common-batch/full-step/OOM checks; synthetic freeze/test/metric/export workflow; parameter counts, links and `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and
`agents/config.md`. Make only the listed changes, verify the result, update this
status, and create `completion.md` from `agents/templates/completion.md`.
Do not run the long real-data comparison, commit, or push. Provide the owner's
run command; keep final-results review pending until those artifacts exist.
