# Task Prompt

- Ticket: `051-save-run-prediction-examples`
- Status: Ready
- Aim: Save a small, configurable set of playable correct and incorrect predictions with each model, training, evaluation, and experiment run.
- Scope: Paper 001 `src/model.py`, `src/train.py`, `src/evaluate.py`, `src/experiments.py`, shared output helpers, their JSON configs, and documentation.
- Changes:
  - Add one JSON setting, `prediction_samples_per_category`, to control the maximum correct and incorrect clips saved per model; use a small default of `3` and allow `0` to disable videos.
  - Save videos under each run's `predictions/correct/` and `predictions/incorrect/` folders. For experiment runs, keep them inside that model's run subdirectory.
  - Use validation samples for model smoke, training, and experiments; use test samples only in final evaluation. Do not change the split manifest or open test data earlier.
  - Log configured and actual saved counts in the console and run JSON; record true/predicted labels and source paths without copying source videos.
  - Clearly mark `model.py` random-weight predictions as smoke checks, not research results.
- Acceptance criteria:
  - All four commands use the same JSON count semantics and save each available category up to the configured limit.
  - Training saves examples from the restored validation-selected checkpoint; experiments save per-candidate validation examples; evaluation saves test examples; model smoke saves validation examples.
  - Output paths and counts are recorded in each run's summary/provenance JSON, and the console reports the same counts.
  - Zero count creates no prediction videos; missing examples in a category are reported as zero without failing.
  - Tests verify split choice, correctness categorization, count limits, filenames/metadata, and that test access remains locked outside evaluation.
- Out of scope: Changing model predictions, metrics, video preprocessing, or dataset contents.
- Open questions: `None`.
- Verification: Focused pytest checks with tiny synthetic clips and a deterministic toy classifier; run model smoke and short train/evaluate/experiment smoke commands if local data and dependencies permit; otherwise report the exact unavailable check.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
