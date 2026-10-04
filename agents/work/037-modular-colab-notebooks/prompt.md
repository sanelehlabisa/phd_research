# Task Prompt

- Ticket: `037-modular-colab-notebooks`
- Status: Done
- Aim: Add small, independent Colab notebooks so each workflow can run without executing unrelated cells from the reference notebook.
- Scope: Paper 001 notebook support code, notebook directory, tests, and implementation documentation.
- Changes:
  - Preserve `notebooks/aad_experiment_workflow.ipynb` unchanged as the reference workflow.
  - Add one shared dataset registry/configuration module containing Kaggle handles, exact accepted classes, paths, and one selected diagnostic dataset key.
  - Keep VDD selected by default; changing one selected-dataset value must update every modular diagnostic notebook using the shared configuration.
  - Keep controlled AAD experiment configuration explicit and frozen; a diagnostic dataset switch must not silently alter the research protocol.
  - Add reusable notebook helpers for Colab setup, dependency validation, repository paths, dataset download/resolution, display, and command execution; reuse existing `src` logic rather than copying training or experiment implementations.
  - Add four independent, thin notebooks:
    - `01_dataset_setup.ipynb`: runtime setup, selected-dataset download, inventory, split summary, and sample inspection.
    - `02_model_inspection.ipynb`: selected-dataset/model construction, parameters, tensor shapes, and labelled random-weight predictions only.
    - `03_model_training.ipynb`: selected-model training, validation-selected checkpoint, curves, validation predictions, and locked test access.
    - `04_controlled_experiments.ipynb`: controlled-plan inspection and one guarded experiment stage using the existing runner.
  - Make every notebook independently runnable from a fresh supported Colab runtime with a short setup cell and no dependency on another notebook's state.
  - Keep generated datasets, manifests, runs, checkpoints, and notebook artifacts untracked.
  - Update the implementation README with each notebook's purpose, command ownership, dataset-selection location, and safe execution order.
- Acceptance criteria:
  - The original notebook remains present and content-identical to its pre-ticket version.
  - VDD is the default shared diagnostic dataset and uses `sanelehlabisa/violence-detection-dataset` with its declared accepted classes.
  - Switching the single selected diagnostic dataset key changes all three diagnostic notebooks without editing them individually.
  - Each notebook contains only cells needed for its named workflow and delegates data, model, metrics, training, and experiment behavior to shared `src` modules.
  - The training notebook never opens the test split; the controlled experiment notebook preserves ticket 021 and all existing execution guards.
  - Notebook JSON, cell IDs, shared configuration, imports, focused tests, and dry-run/list-only paths validate successfully.
- Out of scope: Running expensive training, choosing Kinetics classes before ticket 036, changing controlled experiment results, opening test data, deleting the reference notebook, or changing manuscript claims.
- Open questions: None.
- Verification: Compare the reference notebook hash before and after; parse every notebook and code cell; test dataset selection and helper behavior; exercise safe list/dry-run commands; run focused tests and `git diff --check`.

## Execution Prompt

Complete ticket `037-modular-colab-notebooks`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Preserve `aad_experiment_workflow.ipynb` unchanged as the reference, keep VDD as the default diagnostic dataset, keep controlled AAD configuration isolated, verify the modular notebooks and shared helpers, update this status, and create `completion.md` from `agents/templates/completion.md`.
