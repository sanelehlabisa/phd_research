# Task Prompt

- Ticket: `042-merge-expanded-kinetics-suite`
- Status: In progress
- Execution approval: User explicitly approved completing ticket 042 and pushing master on 2026-10-05; this supersedes the original no-push restriction below.
- Aim: Safely integrate both laptops' work and prepare the expanded diagnostic suite for the larger Kinetics subset.
- Scope: Git integration, Paper 001 modular notebooks/shared helpers, focused tests and concise guides.
- Approved decisions: More than 2,000 unique real videos **before splitting**, not a 2,000-video maximum; retain more experiment runs within the existing eight-hour compute budget.
- Changes:
  - Preserve all current local work on a recovery branch before integrating the latest `origin/master` (there is no `main`). Resolve overlapping edits deliberately; retain both distinct ticket-041 histories.
  - Keep the remote five-class `kinetics600-subset` default for notebooks 01–04, its safe selective downloads/cache, provenance and source-grouped split. Require at least 2,001 unique real clips; print actual counts. Preserve VDD/Kinetics-400 alternatives.
  - Retain the local expanded suite: 11 custom models plus three scratch 3D CNNs, gentle LR scheduling, temporal audit, tiny training-only learning check and separate resource-checked native paper topology.
  - Add explicit 64/96/128-pixel spatial comparisons against the selected 64-pixel custom reference. Retain weight decay 0/0.0001/0.001, 4/8/16 FPS coverage and dropout comparisons. Change one factor at a time with the same split and epoch budget; print and save the full plan and run count before training.
  - Keep experiment settings separate from notebooks 01–03: preserve the remote single-model/preview defaults. Retain the expanded suite's 24-epoch screen, 32-frame/8-FPS reference and two-seed 64-epoch confirmation; final resolution must not be smaller than the selected resolution.
  - Keep the eight-hour cooperative deadline and incremental evidence. Mark unfinished runs partial, never successful; do not open test after an incomplete comparison. Do not promise every planned run will finish in eight hours.
  - Select using validation only; freeze compatible confirmation checkpoints before final test metrics and examples. Preserve curves, display recovery and saved outputs from both laptops (retain conflicting historical outputs in the recovery history).
  - Preserve the original reference notebook, controlled AAD settings, datasets, manifests, old runs and manuscript. Describe Kinetics results as exploratory; do not infer an untouched holdout from changing the dataset name.
  - Update the three guides and this ticket; complete the local integration commit only after verification. No push in this ticket.
- Acceptance criteria:
  - Local `master` contains the fetched remote history and verified integration; the original local work remains recoverable and no unresolved conflicts remain.
  - All four notebooks share Kinetics-600; the >2,000 count gate and grouped split remain enforced without counting augmentation or repeat epochs as new videos.
  - Notebook 04 executes the expanded plan, including genuine same-budget spatial and weight-decay comparisons, not merely a larger final-training resolution.
  - Source/split identities remain consistent across input views; incompatible checkpoints and incomplete studies cannot unlock test. Historical outputs remain recoverable and are not relabelled as new results.
- Out of scope: Pushing, real GPU training/downloads here, full Kinetics, additional activity classes, pretrained transfer, accuracy guarantees and manuscript claims.
- Open questions: None. Integration, verification, commit and push authorized.
- Verification: Full pytest; focused Kinetics count/grouping/provenance and suite/factor/budget/test-lock checks; synthetic-video workflows; notebook JSON/code-cell syntax and output preservation; formatting, compile checks, `git diff --check`, conflict-marker and Git ancestry checks. Report actual Colab clip counts and runtime as pending until observed.

## Execution Prompt

Complete ticket `042-merge-expanded-kinetics-suite`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Preserve local work on a recovery branch, integrate the latest `origin/master`, keep more than 2,000 unique Kinetics videos before splitting, and retain the expanded eight-hour suite with model, spatial-size and weight-decay comparisons. Preserve saved outputs, verify the integration, update this status and create `completion.md` from the template. Commit the verified integration locally; do not push or start real GPU experiments.
