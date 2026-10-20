# Task Prompt

- Ticket: `093-colab-demo-training`
- Status: Done
- Aim: Train the selected demo model on Colab GPU with one paste-and-run cell.
- Scope: New notebook/export, focused tests and README links.
- Approval: User explicitly requested immediate implementation after abandoning CPU training.
- Changes:
  - Reuse the published experiment runner with an embedded winner recipe.
  - Mount/check Drive first; save checkpoint, class/config metadata and validation results.
  - Keep dataset/cache local; finish with curves, predictions and a downloadable bundle.
- Acceptance criteria:
  - One fresh [64,64,128] training: 8 frames, 64px, 16 FPS, original split/seed.
  - Frozen LR/WD/batch; minimum 128, cap 256 epochs; validation-selected checkpoint.
  - No search, test evaluation, manual stage switches or unpublished helper dependency.
  - Notebook/export agree; GPU and Drive failures stop before training.
- Out of scope: Paper/slides, app changes, deployment, commit/push, local full training.
- Open questions: None.
- Verification: Mocked orchestration, recipe/CLI checks, notebook parity and regression tests.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Preserve existing work and evidence, verify the result, update status and create `completion.md`. Do not commit or push.
