# Task Prompt

- Ticket: `038-notebook-prediction-endings`
- Status: Done
- Aim: End each modular notebook with five clear examples appropriate to that workflow, culminating in guarded final-test predictions only after validation-based model selection.
- Scope: Paper 001 modular notebooks, shared notebook helpers, training/evaluation delegation, focused tests, and notebook documentation.
- Changes:
  - Preserve the user's saved outputs currently present in `01_dataset_setup.ipynb` and `02_model_inspection.ipynb`.
  - End `01_dataset_setup.ipynb` with five reproducibly sampled training datapoints showing path, class label, tensor shape, and a compact visual or playable preview.
  - End `02_model_inspection.ipynb` with five reproducible non-test random-weight predictions showing expected label, predicted label, confidence, and correct/incorrect; label them explicitly as a pipeline check, not trained performance.
  - End `03_model_training.ipynb` by loading the validation-selected checkpoint and showing five validation predictions with expected label, predicted label, confidence, and correct/incorrect.
  - Extend `04_controlled_experiments.ipynb` into a guarded sequence that delegates to existing runners: complete validation-only candidate selection, require an explicit frozen winner, run the longer confirmation training, perform one final test evaluation, and only then show five reproducible test predictions.
  - Require complete run/checkpoint/split provenance before final evaluation and reject incomplete screens, manually typed winners without matching validation evidence, or checkpoints from another dataset/split.
  - Keep ticket 021's candidate-expansion decision guard and tickets 025–027 research stages explicit; do not automatically open test data merely by running the notebook from the top.
  - Add or extend shared helpers so notebooks contain presentation and orchestration only, without duplicating dataset, metric, checkpoint, prediction, or training logic.
  - Keep every prediction group reproducible and clearly identify its partition and evidence role.
- Acceptance criteria:
  - Dataset notebook ends with exactly five reproducibly selected training examples.
  - Model-inspection notebook ends with exactly five non-test random-weight prediction cards and correctness indicators.
  - Training notebook uses the validation-selected checkpoint and ends with exactly five validation prediction cards and correctness indicators.
  - Controlled-experiment notebook cannot evaluate test data until a completed validation screen selects a winner and the longer confirmation run produces a compatible frozen checkpoint.
  - After that gate, final evaluation reports complete test metrics once and shows exactly five reproducible test predictions from the evaluated checkpoint.
  - Test metrics and examples are labelled final evaluation, never architecture-selection evidence.
  - Existing saved notebook outputs are preserved unless a directly edited cell must be reset for valid notebook structure.
  - Notebook JSON, cell IDs, guards, provenance validation, focused tests, safe dry runs, and `git diff --check` pass.
- Out of scope: Choosing the ticket-021 expansion outcome without timing evidence, running expensive experiments locally, tuning from test results, changing manuscript claims, or enabling Kinetics before ticket 036.
- Open questions: None.
- Verification: Parse all modular notebooks and code cells; test deterministic five-sample selection, checkpoint/split validation, winner gating, and one-time test gating; exercise list/dry-run paths; verify the reference notebook remains unchanged; run focused tests and `git diff --check`.

## Execution Prompt

Complete ticket `038-notebook-prediction-endings`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Preserve the saved outputs in the first two modular notebooks, use validation only for selection, gate final test evaluation behind a frozen validation-selected winner and compatible longer-trained checkpoint, verify all four notebook endings, update this status, and create `completion.md` from `agents/templates/completion.md`.
