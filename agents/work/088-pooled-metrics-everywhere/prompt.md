# Task Prompt

- Ticket: `088-pooled-metrics-everywhere`
- Status: Done
- Aim: Make all new active modular and campaign outputs use pooled precision/recall/F1, without macro averages.
- Scope: Existing metric, evaluation, campaign validation/reporting/export paths, targeted tests and concise guides. No parallel metric framework.
- Context: Direct JSON experiments are already pooled-only. Final `src.evaluate` still saves macro metrics; campaign leaves, matched-training reports and older validators/tables also retain them. Hiding columns alone is insufficient.
- Changes:
  - Retain accuracy, sample-weighted loss, pooled P/R/F1, full per-class counts/metrics, confusion matrices, parameters/model size and timing. Explicitly document that pooled P/R/F1 equal accuracy in single-label multiclass classification.
  - Remove automatic macro/balanced-average computation and output from new active train/evaluate/model/experiment and final-comparison results, console, JSON, CSV, Markdown, plots and ZIP summaries. Do not remove per-class evidence or substitute another unlabelled averaging scheme.
  - Adapt new campaign validators/receipts to verify pooled and per-class results against predictions/confusion counts. Preserve genuine split/class/checkpoint/selection guards. Version changed contracts where needed rather than silently relabelling old artifacts.
  - Keep historical files/scores/checkpoints immutable and maintain explicit legacy reading/verification. Do not globally disable safeguards just because old protocols required macro fields.
  - Align active README/agent context and the planned manuscript metric wording; do not rewrite the manuscript or old reports.
- Acceptance criteria:
  - Synthetic active modular and campaign train/validation/frozen-test fixtures produce no macro/balanced-average fields, columns or console scores, while retaining the requested metrics and examples/confusion artifacts.
  - Accuracy and pooled P/R/F1 match full-partition counts; uneven batches do not change aggregation. Tampered predictions/metrics and genuine provenance mismatches still fail.
  - Legacy evidence remains readable, unchanged and explicitly labelled; direct JSON smoke behavior stays compatible.
- Out of scope: Real training/testing, changing search/ranking/checkpoint selection, deleting old evidence, new datasets/models, notebook integration (089), manuscript results, commit or push.
- Open questions: None. This is preparation, not permission to launch a new study.
- Verification: Core metric/checkpoint compatibility tests; active modular/campaign output assertions; confusion/count reconciliation and mismatch rejection; legacy artifact hashes; `git diff --check`.

## Execution Prompt

Execute ticket `088-pooled-metrics-everywhere` exactly as written here.
Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the
approved changes, verify every acceptance criterion, mark `Done`, and create
`completion.md` from `agents/templates/completion.md`. Do not launch real
training/testing, commit or push.
