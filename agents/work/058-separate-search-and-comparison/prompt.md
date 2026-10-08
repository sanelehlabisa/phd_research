# Task Prompt

- Ticket: `058-separate-search-and-comparison`
- Status: Done
- Aim: Use the existing experiment runner for two distinct AAD studies: custom ConvLSTM search and fixed-setting model-family comparison.
- Scope: Paper 001 modular experiment runner, its JSON profiles, focused tests, and implementation guide; depends on 057.
- Changes:
  - Keep one command entry point: `.venv/bin/python -m src.experiments --config <json>`; do not add a comparison script.
  - Maintain three active AAD experiment profiles: one tiny local smoke config, one Colab custom-search config, and one Colab model-comparison config. All select AAD and infer classes from its folders.
  - Make custom search contain only `CustomConvLSTM` candidates. Start with `[16, 16, 16]`; include predeclared one-factor depth/width variants, plus a small JSON-controlled set of frame sizes, frame counts, and weight decays. Print the total planned runs before training.
  - Make comparison use fixed input/training/regularization settings across model families and the same two seeds (`42`, `2026`) for every model. Include the validation-selected custom architecture, faithful `PaperConvLSTM`, at least two practical 3D CNNs, and up to two video transformers if a short feasibility check succeeds. Keep the selected custom layer spec/path in the comparison JSON so search is not rerun.
  - Use validation-only selection, preserve the locked test split, and retain comparable metrics, parameter counts, runtime, configs, and run provenance.
  - Update concise run instructions and distinguish the custom-search results table from the model-family comparison table.
  - Keep legacy config files that are still referenced by existing notebooks until ticket 061 migrates those notebook references; mark them as legacy rather than active profiles. Do not delete run artifacts.
- Acceptance criteria:
  - The same `src.experiments --config ...` command runs all three active profiles; no separate comparison script or multi-argument candidate/plan command is required for the new workflows.
  - The custom-search profile lists only custom architectures and visibly includes the `[16, 16, 16]` reference and the approved one-factor variants.
  - The comparison profile varies model family only: shared dataset, split, input size, two seeds, training budget, optimizer, augmentation, and regularization settings; it contains the selected custom model, paper model, two practical 3D CNNs, and two feasible transformers (or records why a transformer is unavailable).
  - `--list-plan` reports each profile's planned run count and configurations without training.
  - Tests confirm test metrics do not affect model/config selection and artifacts support the two distinct results tables.
- Out of scope: Full expensive A100 runs, notebook edits or legacy-config deletion (ticket 061), changing the split, and manuscript result claims.
- Open questions: `None`.
- Verification: Focused runner/config tests, list-plan checks for all three profiles, selection/test-lock tests, model feasibility smoke checks, script help, config-reference checks, and `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
