# Task Prompt

- Ticket: `083-comparison-only-recovery`
- Status: Ready
- Aim: Prepare a comparison-only Colab script, diagnose failed baseline learning, and safely complete held-out evaluation without repeating or changing the successful search.
- Scope: Paper 001's existing comparison/evaluation helpers, one thin comparison notebook with matching Python export, regression tests and concise guides. Reuse existing modules; no replacement experiment framework.
- Evidence: [14 October review](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-14/README.md). All 64 search jobs and ten comparison trainings finished; first test failed on `aad` versus `abnormal-activities-dataset`. Both Swin models predicted one class at 12.5% validation accuracy. This is a convergence issue to investigate, not proof that they must achieve 90%.
- Changes:
  - Treat the 64 search results, ranking, original outputs and selection receipt as immutable. Import and validate the saved selection/split; never launch search when inputs are missing. Preserve its five custom finalists and common eight-frame, 64px, 16-FPS input, seed 42 and class order.
  - Prepare a one-entry comparison-only script/notebook for those five customs plus the input-adapted paper ConvLSTM, R3D-18, MC3-18, Swin3D-T and Swin3D-S. Reuse data caches; show comparison progress 1–10. Import one verified evidence bundle if required; no manual stage toggles or hardcoded Colab run-directory edits between stages.
  - Diagnose transformer learning using saved histories, class/prediction counts, preprocessing/layout, initialization, optimizer/LR schedule, finite gradients and a bounded train-only small-subset overfit check. Correct demonstrated implementation bugs and record evidence. Do not silently switch to pretrained weights, tune on test or promise an accuracy threshold. If recipe/pretraining changes are needed, return a validation-only proposal for owner approval before a fresh expensive comparison.
  - Fix canonical dataset identity at the checkpoint/evaluation handoff, keeping genuine dataset/split/class/checkpoint mismatch guards. Support verified historical metadata without rewriting original checkpoints. Distinguish a failure before scoring from a completed or partial test; never delete receipts to force reuse.
  - Once the recipe is approved, a fresh comparison trains only the ten models, selects checkpoints on validation, freezes all models and the best custom, evaluates each test partition once, then saves labelled example predictions for that custom. Keep an explicit safe recovery path for compatible frozen checkpoints so an evaluation-only failure does not require retraining. Missing weights must be reported honestly.
  - Save new attempts separately, preserving the failed comparison. Export compact metrics/configs, histories, selection/freeze/test receipts, per-class P/R/F1/support, confusion counts, prediction records, parameters, sizes and measured timings after each stage and on failure. Keep checkpoint backups separate; label all averaging and input-adapted baselines clearly.
- Acceptance criteria:
  - Existing search data/selection hashes remain unchanged; the comparison entry point cannot invoke or modify the 64-job search.
  - All five finalists and five baselines trace to verified saved configuration; missing evidence fails clearly without a fallback search or arbitrary model selection.
  - Transformer diagnosis distinguishes training collapse from overfitting; failed models and historical scores remain visible. Any changed protocol is approved and separately labelled before final comparison/test.
  - Dataset aliases work only with compatible provenance; genuine mismatches, partial tests and already-completed tests remain protected.
  - An approved comparison run needs no manual stage switching after evidence setup; final test and examples follow validation freeze. Local verification is not claimed as A100 execution or a new score.
- Out of scope: Rerunning/reranking search, changing its values or shortlist, changing dataset/split/input sizes without new approval, manuscript edits, real full Colab training during implementation, commit or push during execution.
- Open questions: None for preparing the runner and diagnosis. A different baseline training recipe or pretraining requires a follow-up decision based on the investigation; do not assume that decision now.
- Verification: Regression tests for aliases and real mismatches, frozen-selection import, zero search calls, ten-model routing, cache reuse, synthetic train → validation freeze → test → examples → compact export, failure/resume guards, notebook/export parity, evidence hashes and `git diff --check`.

## Execution Prompt

Complete ticket `083-comparison-only-recovery`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Preserve the completed search and
all original evidence. Implement and verify only the approved preparation and
diagnosis above; obtain approval before materially changing the comparison
training protocol. Update this status and create `completion.md` from
`agents/templates/completion.md`. Do not launch full training, commit or push.
