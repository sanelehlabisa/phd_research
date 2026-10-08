# Task Prompt

- Ticket: `058-separate-search-and-comparison`
- Status: Ready
- Aim: Separate custom ConvLSTM architecture search from comparison against established model families.
- Scope: Paper 001 modular experiment/comparison runners, focused configs, tests, and run documentation; depends on 057.
- Changes:
  - Keep the architecture-search command focused only on `CustomConvLSTM`, starting from flat `[16, 16, 16]`; compare `[16, 16]`, `[16, 16, 16, 16]`, and one-factor width variants `[32, 16, 16]`, `[16, 32, 16]`, `[16, 16, 32]`, with all other settings fixed.
  - Use one seed for the initial screen, then confirm the validation shortlist with two seeds. Select only on validation; leave test data locked.
  - Add a separate comparison command/config that consumes the selected custom architecture and compares it with `PaperConvLSTM`, at least two practical 3D CNNs, and two video transformers if both pass a small feasibility check. Keep model names and dataset/input/training settings in JSON.
  - Use the configured dataset and all discovered classes. Keep shared split, input size, training budget, and initialization policy comparable; log parameter count, runtime, loss, accuracy, precision, recall, F1, confusion matrices, and run provenance.
  - Preserve all existing outputs; update concise commands and explain which stage produces each results table.
- Acceptance criteria:
  - Search lists only custom architectures, visibly includes the `[16, 16, 16]` reference and the predeclared one-factor variants, and reports its planned run count before training.
  - Comparison is a separate one-config command and loads the selected custom architecture without rerunning architecture search.
  - The comparison includes the faithful paper topology, two practical 3D CNNs, and two feasible transformer models; if either transformer is not feasible, record the reason and keep the remaining comparison usable.
  - No test metrics influence search, ranking, or checkpoint selection; output artifacts support two distinct paper tables.
- Out of scope: Full expensive A100 runs, notebook edits, changing the split, and manuscript result claims.
- Open questions: `None`.
- Verification: Focused runner/config tests, list-plan checks for both configs, selection/test-lock tests, model feasibility smoke checks, script help, and `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
