# Task Prompt

- Ticket: `084-diagnose-transformer-learning`
- Status: Done
- Aim: Explain the Swin3D learning collapse and fix demonstrated code defects before final comparison, without tuning on test.
- Scope: Paper 001's existing model construction, training/preprocessing path, targeted diagnostics/tests and a new dated diagnosis report. Keep helpers minimal and reuse the modular scripts.
- Evidence: [14 October review](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-14/README.md#why-the-transformers-require-caution). Swin3D-T/S trained from scratch: 93/748 training and 20/160 validation correct; both predict Begging only. The cause is not yet established; 12.5% alone does not prove a code bug.
- Changes:
  - Preserve original results/configs. Audit both transformers' input axis order, range/normalization, frame sampling, class targets/head, initialization, train/eval mode, optimizer membership, LR/scheduler, loss, finite gradients and actual parameter updates.
  - Reuse the saved histories/prediction counts and compare with the working custom/CNN path. Separate implementation errors, optimization collapse and unsupported hypotheses; do not call this ordinary overfitting.
  - Run a bounded fixed-seed, train-only small-subset learning check for each transformer, beginning with a 6-GB-safe batch. Record the subset, step budget, loss, accuracy, gradients and parameter changes. An inconclusive short check is not proof of incapacity; never read test for diagnosis.
  - Correct proven implementation defects with regression checks. Do not silently change input sizes, pretrained weights, optimizer or the comparison recipe. If those are needed, report a small validation-only proposal and request approval before a new expensive comparison. Preserve the original from-scratch failures as a separately labelled outcome.
  - Save a concise report under `implementation/reports/YYYY-MM-DD/` with reproducible diagnostic commands, findings, before/after evidence and the exact recommended handoff to [083](../083-comparison-only-recovery/prompt.md). Do not promise any accuracy threshold.
- Acceptance criteria:
  - Both transformer paths have reproducible input/gradient/update checks and recorded learning traces; established causes are distinguished from unresolved limitations.
  - Any demonstrated defect has a minimal tested fix; no test access, search rerun, fabricated improvement or unapproved protocol change occurs.
  - 083 can identify the verified code/recipe or the explicit remaining approval needed. A completed diagnosis is not a claim that final comparison has run.
- Out of scope: Full ten-model training/testing, new model families, search/ranking changes, manuscript edits, broad cleanup, commit or push.
- Open questions: None for diagnosis. Pretraining or materially different training settings, if recommended by evidence, require a later owner decision.
- Verification: Targeted model/runner regressions, real train-only smoke checks where CUDA/data are available (otherwise record the limitation), source/evidence hashes and `git diff --check`.

## Execution Prompt

Execute ticket `084-diagnose-transformer-learning` exactly as written here.
Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only approved
changes, verify each acceptance criterion, set the status to `Done`, and create
`completion.md` from `agents/templates/completion.md`. Do not launch full
comparison training, access test, commit or push.
