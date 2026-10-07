# Task Prompt

- Ticket: `050-overall-micro-metrics`
- Status: Ready
- Aim: Report the requested overall classification metrics without macro-averaging.
- Scope: Paper 001 metric utilities, training/evaluation/experiment logs and JSON outputs, rankings, and concise documentation.
- Changes:
  - Calculate full-partition accuracy, micro precision, micro recall, micro F1, sample-weighted loss, and the existing confusion matrix.
  - Replace macro metric names and macro-F1 ranking/logging with overall metric names; rank experiment candidates by validation accuracy, then validation loss, then parameter count. Keep checkpoint selection on validation loss.
  - Record the aggregation protocol in new run artifacts. Leave historical run files unchanged.
- Acceptance criteria:
  - Train, evaluation, and experiment outputs consistently contain loss, accuracy, precision, recall, and F1 for the full relevant partition, with no macro fields.
  - Console output and experiment summaries use the same overall metric names and ranking rule.
  - Confusion matrices remain per-class and are not replaced by a single aggregate.
  - Tests verify micro metrics against a hand-computed multiclass example and verify ranking/checkpoint selection still use validation only.
  - Documentation states that in single-label multiclass classification, micro precision/recall/F1 equal accuracy; no historical result is silently rewritten.
- Out of scope: Splits, new models, prediction-video sampling, and manuscript results.
- Open questions: `None`.
- Verification: Focused metric and ranking pytest checks; full implementation pytest suite; inspect one generated JSON/log using a small synthetic fixture; confirm historical run artifacts are unchanged.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
