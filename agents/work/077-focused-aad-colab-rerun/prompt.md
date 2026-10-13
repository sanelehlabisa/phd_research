# Task Prompt

- Ticket: `077-focused-aad-colab-rerun`
- Status: Ready
- Aim: Prepare a bounded A100 rerun that measures architecture and input-setting tradeoffs with recoverable evidence for the final comparison.
- Scope: One new focused AAD JSON, existing modular search/config/reporting code, a thin `notebooks/aad_focused_workflow.ipynb` and matching Python export, verification and concise guides. Preserve legacy profiles, notebook outputs, logs and runs.
- Context: Requested 13 October 2026 after reviewing [the saved search](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-13/latest_colab_results.md). The owner will rerun Colab; this ticket prepares the workflow and does not launch real training.

## Fixed manifest and budget

Keep the ten best observed distinct seed-42 architectures and six controlled anchors:

| Group | Filters per layer |
|---|---|
| Prior top ten | [96,64,64]; [32,64,64]; [64,96,64]; [48,48,48]; [64,128,64]; [64,64,128]; [128,64,64]; [64,64,32]; [64,64,64,64,64]; [96,96,96] |
| Six small/depth/large controls | [16,16,16]; [64]; [64,64]; [64,64,64]; [64,64,64,64]; [128,128,128] |

- Exactly **16 custom architectures × frame sizes [32,64] × FPS [8,16] = 64 balanced search jobs**. Fix sequence length at eight frames and kernels at 3×3. No optional expansion to 32 models in this ticket.
- AAD only, all classes, existing 748/160/161 split and source-review guards; split/model seed 42. Fixed Adam, LR 0.001, WD 0, augmentation off and existing scheduler. Use **256-epoch cap, minimum 64, validation-loss patience 24** and one measured safe common search batch. The cap preserves late-learning candidates; 64 jobs can still take hours.
- Changing FPS with eight frames changes temporal coverage; save the effective duration/sampling settings. This tests sensitivity to preprocessing, not independent-dataset or subject generalization.

## Changes

- Reuse `src.experiments --config <json>` and existing helpers. Add a focused profile rather than modifying old evidence. The new notebook/export should clone/update, prepare the environment/AAD, list the exact plan, run the modular search and save/download artifacts with concise live progress. It initially runs search only; 076 adds the final comparison.
- Cache processed train/validation clips per input condition, exclude test, and retain the same split across all 64 jobs. Preflight the largest model/input with a full optimizer step and report measured memory, throughput and estimated workload before training. Never silently drop configurations.
- Rank distinct architectures by equal-weight mean validation accuracy across all four conditions, then worst-condition accuracy, mean loss, parameters and stable name. Save all per-condition scores, mean/minimum scores and parameter counts. Never rank by the best isolated setting or any test result.
- Reserve `[96,64,64]` as a prespecified final-comparison candidate because the owner requested the former 99% model. If it is in the numerical top five, keep that five; otherwise select the top four plus this anchor and disclose its rank. Save the numerical ranking and finalist list separately.
- Freeze one common final input using the highest mean validation accuracy across the five finalists' four shared conditions; tie-break by mean loss, then smaller frame size, then higher FPS. Save its exact validated configuration for 076. No additional hyperparameter training or family comparison in this ticket.
- Evaluate clean training and validation at each selected checkpoint; preserve both losses/accuracies, gaps, full epoch histories, parameters/bytes, timing, class counts, confusion matrices and prediction records. Save width-at-fixed-depth, depth-at-fixed-width and early/middle/late-width summaries. Underfitting, overfitting, saturation and optimization collapse must remain distinct; do not guarantee a turning point or discard adverse results.
- Save progress/metrics atomically after each job. After each 16-model input block and at search completion, create/download a small results ZIP containing configs, split/selection receipts, CSV/JSON, histories, curves and confusion matrices. Keep checkpoint archives separate so a multi-GB download does not block saving metrics. Use an existing durable destination if the owner supplies one; never claim browser delivery is verified. Reuse existing retry/resume guards.
- Prepare accuracy-versus-parameter data for every size/FPS condition and matched training/validation ablation summaries; 078 handles presentation details. Preserve the old console-derived report; add a dated verified report after owner execution.

- Acceptance criteria:
  - Exactly 16 specified models and 64 condition-complete jobs, with fixed data/split/recipe/seed and test locked.
  - Width-64 depths 1–5, flat three-layer widths 16/48/64/96/128, and early/middle/late widening are represented; no comparison of multiple changed factors is labelled a one-factor ablation.
  - Stable-across-input ranking, the disclosed anchor slot, finalist selection and common-input freeze reproduce from verified artifacts; incomplete coverage or provenance fails safely.
  - Focused notebook/export parity, synthetic search/selection/archive/resume verification, cache/test-lock/OOM checks, links and `git diff --check` pass. No real training, fabricated score, manuscript edit, commit or push during implementation.
- Out of scope: 32-model expansion, LR/WD/augmentation/seed sweeps, changing datasets/splits, final test access, forcing larger models to fail, or rewriting legacy workflows.
- Open questions: `None` for this bounded plan; actual runtime/convergence and artifact delivery require measurement.
- Verification: Manifest and 64-job listing; controlled-factor/coverage/ranking/selection tests; clean matched-checkpoint metrics; full-step memory/cache/test-lock checks; synthetic compact archive/resume; notebook/export parity and `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and
`agents/config.md`. Make only the listed changes, verify acceptance criteria,
mark Done and create `completion.md` from `agents/templates/completion.md`.
Do not launch the actual Colab study, commit, or push. Return the new notebook
and copy-paste Python export and the modular command for the owner.
