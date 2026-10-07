# Task Prompt

- Ticket: `027-evaluate-selected-model`
- Status: Blocked
- Code preparation: Ticket 055 supplies guarded final testing; no real test evaluation has occurred.
- Execution check (2026-10-07): Ticket 026 has not produced the required frozen model/checkpoint. Keep test locked; final evaluation must follow training and selection, not run concurrently.
- Aim: Perform the one-time final AAD test evaluation and export complete
  quantitative and visual evidence for the frozen model.
- Scope:
  - Paper 001 evaluation and metric-export workflow
  - the ticket-026 validation-selected checkpoint
  - local ignored `runs/evaluate/` evidence
  - concise Paper 001 and implementation status notes
- Changes:
  - Execute only after ticket 026 freezes the configuration and checkpoint
    without consulting test results.
  - Open the committed AAD test partition once for the frozen checkpoint. Do not
    tune, retrain, switch seeds, or select another checkpoint after viewing it.
  - Export sample-weighted loss, accuracy, overall micro precision/recall/F1 (050), per-class
    precision/recall/F1/support, confusion matrix, parameter count, inference
    runtime, resolved configuration, split hash, seed, and code revision.
  - Save reusable prediction records with source clip, true class, predicted
    class, confidence, and correctness.
  - Display a small reproducible set of correct and incorrect playable examples;
    if too few examples exist in either group, report that rather than changing
    the sample rule.
  - Keep final AAD evidence separate from later VDD generalisation and any
    conditional Kinetics transfer-learning work.
- Acceptance criteria:
  - The evaluated checkpoint and configuration exactly match ticket 026.
  - One complete timestamped evaluation run contains all declared metrics,
    provenance, confusion output, predictions, and playable examples.
  - No post-test model or protocol change is made or recommended as if it were
    part of the same unbiased evaluation.
- Out of scope:
  - VDD, Kinetics, retraining, checkpoint reselection, new ablations,
    manuscript rewriting, or committing generated datasets/checkpoints/runs.
- Open questions: None. The checkpoint is chosen by ticket 026 before test
  access.
- Verification:
  - Validate checkpoint/config/split compatibility before evaluation.
  - Recalculate metrics from saved predictions and compare with exported JSON.
  - Verify deterministic example selection, artifact completeness, one test
    access path, focused tests, and `git diff --check`.

## Execution Prompt

Execute ticket `027-evaluate-selected-model` exactly as written in
`agents/work/027-evaluate-selected-model/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
