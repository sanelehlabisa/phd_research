# Superseded ticket 075 specification

Historical record only. Do not execute this version; use [prompt.md](prompt.md).

# Task Prompt

- Ticket: `075-wide-depth-capacity-search`
- Status: In progress
- Execution update (10 October 2026): Paused for the user's revised planning request after reviewing `reports/anotehr.update.md`. Earlier local implementation remains uncommitted and incompletely verified; do not treat this ticket as complete or Colab-ready. See the appended planning section in the interim analysis. Final input, resolution budget, published-baseline treatment and reporting changes await agreement; the approved specification below is retained as history, not silently replaced.
- Replanning answers (10 October 2026): User prefers final 16-frame/64x64 input, validation-selected FPS, inclusion of five-layer search controls, an explicitly input-adapted paper baseline, and unaveraged per-class metrics without macro columns in new main tables. Preserve historical metrics/evidence. Clarify whether 64 means architectures or base training runs, and confirm search inputs before drafting the replacement execution specification. No new implementation, commit or push is authorized by this planning exchange; the previous specification below remains historical.
- Aim: Test capacity beyond the current width-64 boundary and identify measured width, depth and layer-position trade-offs without presuming an overfitting optimum.
- Scope: Paper 001's existing search/comparison runners, a new versioned AAD profile, active notebook/export, reports, tests and concise guides.
- Approval context: On 10 October 2026 the user approved the focused follow-up, requested larger models and depths 1/2/3/4, then supplied the execution prompt. They will run the real experiments separately.
- Inputs: [Interim analysis](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/todays.results.md), preserved raw log, existing split and tickets 073/074. These are validation-led exploratory findings, not final test evidence.
- Decisions:
  - Interpret more kernels as more filters/channels; keep spatial kernels 3x3, adaptive head and model definitions otherwise unchanged.
  - Bound the search at 128 filters per layer and four recurrent layers. No automatic expansion if the largest model wins; report a boundary optimum within the tested grid, not a global optimum.
  - Preserve current uncommitted reports/README edits, all previous results, saved notebook outputs and legacy protocols. Prune only the next candidate manifest, never historical evidence.

## Candidate manifest

Exactly **31 distinct custom architectures**, each at **32/48/64 pixels**:

| Group | Architectures | Count |
|---|---|---:|
| Width-by-depth controls | Flat widths 32/64/96/128, each at depths 1/2/3/4 | 16 |
| Small and intermediate anchors | [8,8,8], [16,16,16], [48,48,48] | 3 |
| Three-layer shape controls | [32,32,64], [32,64,32], [64,32,32], [32,64,64], [64,32,64], [64,64,32] | 6 |
| Widen one layer of [64,64,64] | [96,64,64], [64,96,64], [64,64,96] | 3 |
| Two-layer orientations | [32,64], [64,32] | 2 |
| Earlier strong shaped control | [16,32,32] | 1 |

This includes [128], [128,128], [128,128,128] and [128,128,128,128]. It is a focused, evidence-guided search, not every possible shape or a parameter-matched causal study.

## Changes

- Add a distinct protocol/profile using the existing runner; do not mutate 068/069/073 definitions or silently mix their metrics into this campaign. Reuse verified processed-input caches, not historical training results as new-protocol evidence.
- Keep AAD, the existing split/source review, split seed 42, fixed class mapping, augmentation off, 8 frames/16 FPS reference input, search cap 128/minimum 64/patience 24 and the existing gentle scheduler. Retain the existing optimizer and base WD 0.
- Calibrate one shared custom LR on [64,64,64] and [128,128,128], plus each reference family (R3D-18 and Swin3D-T), at 48px with LRs 0.001/0.003/0.01. Record the decision; never select a different LR secretly for a weak candidate.
- Run all 31 custom models and both references across three resolutions, seed 42. Rank distinct custom architectures by equal-resolution exact validation accuracy, then loss, parameters and stable name.
- Confirm the provisional top five plus the three-layer flat width-32/64/96/128 controls and both references at all resolutions with seed 2026, deduplicating overlaps. Re-rank the provisional top five using both seeds; pass the best three to comparison. Controls outside that shortlist remain stability evidence, not post-hoc entrants. Report every confirmed control and seed separately.
- Keep separate WD ablations (0 reference, 1e-4, 1e-3) for the confirmed top three at each resolution, and temporal ablations for those three plus both references at 48px: 8f/8FPS, 8f/4FPS and 16f/16FPS against 8f/16FPS. Never combine ablation scores into architecture ranking.
- Preflight the widest/deepest models at maximum search inputs, temporal settings and native comparison inputs before expensive work. Choose one safe search batch for the entire campaign; record any reduction. Do not silently omit OOM models, change their frames/resolution, mix batches in matched comparisons or declare an incomplete search complete.
- Preserve automatic Run All: search -> confirmation/ablations -> 074 native-input LR/WD checks -> three custom + five baseline trainings -> validation freeze -> guarded test -> prediction examples -> verified ZIP. Retain 50-frame/50x50/batch-1 comparison and 512/minimum-128/patience-32 training, with one validated common recipe.
- Keep 074 data/cache reuse, stable unique experiment IDs, completion verification, failure/test guards and archive/download retry. The new profile gets distinct run identities; rerunning it reuses only complete compatible work from that protocol. No manual path entry or stage switches.
- Save all metrics, per-class/confusion outputs, configs, histories, checkpoints, parameters, measured timing/memory and full appendix rows. Console labels must explicitly say micro/macro. Add matched width/depth/position summaries and accuracy/macro-F1 versus parameters/cost plots.
- For capacity interpretation, evaluate training and validation at the same selected checkpoint in evaluation mode; do not compare an epoch's online training average with a different selected checkpoint. Keep full curves and distinguish poor optimization, underfitting, validation degradation and diminishing returns. Mark unmeasured/unsupported diagnoses explicitly; never manufacture a turning point.
- Keep test locked during all new selection. Do not stop, edit or claim control of the user's live Colab run. Do not use any old/current test output to design this extension; if test results have already influenced a decision, disclose test exposure and require a reviewed evaluation protocol rather than calling it a fresh holdout. Never reset prior test-attempt guards to make a rerun proceed.
- Update concise guides with the new plan and resource warning. Preserve all earlier weak/failed configurations in historical reports; do not present pruning as an unbiased fresh search.

## Bounded run plan

| Stage | Maximum trainings before exact-config reuse |
|---|---:|
| LR calibration | 12 |
| Custom architectures: 31 x 3 resolutions | 93 |
| Two references x 3 resolutions | 6 |
| Confirmation: up to 9 customs + 2 references, seed 2026 | 33 |
| Separate weight decay | 18 |
| Separate temporal settings | 15 |
| Search total | 177 |
| Native-input recipe validation | 3 |
| Final comparison | 8 |
| Full-workflow upper bound | 188 |

Counts exclude final metric passes/memory preflight; exact compatible reuse can reduce them. This is a job/epoch budget, not an eight-hour or single-session runtime promise. Larger models can materially increase runtime despite the same job bound. Save verified stage archives; do not claim ZIP-only storage survives deletion of the Colab VM.

- Acceptance criteria:
  - Exactly 31 unique custom stacks; depths 1/2/3/4, width 128, all declared controls and 93 base custom configurations are verified.
  - Complete plan/counts, fixed-recipe comparisons, two-seed ranking, matched-checkpoint training/validation metrics and explicit macro/micro reports are reproducible.
  - A synthetic Run All reaches all eight final evaluations/examples/export without manual paths; memory, stale evidence, partial job and repeat-test guards fail safely.
  - Legacy selections remain readable; notebook outputs and prior report/raw-log hashes are preserved; no real training/test or claim of new accuracy occurs during implementation.
- Out of scope: Five-layer or >128-filter stacks, changing kernel sizes, new datasets/resplits, pretrained-baseline tuning, manuscript result claims, real Colab execution, commit or push.
- Open questions: None for this bounded implementation proposal; actual runtime/memory remain to be measured, and test exposure must be disclosed if it occurs.
- Verification: Candidate/count/ranking and OOM-path tests; tiny CPU forward/backward for 1-4 layers; matched-checkpoint metric tests; mock/synthetic full orchestration and archival tests; legacy/cache/reuse/test-guard regressions; notebook/export and saved-output checks; formatter, links and `git diff --check`.

## Execution Prompt

Complete ticket `075-wide-depth-capacity-search`. Follow `AGENTS.md`,
`agents/rules.md`, `agents/config.md` and the approved prompt. Preserve current
reports, saved notebook outputs and legacy evidence. Implement the bounded
1-4-layer wider search and automatic comparison handoff, verify it, update this
status and create `completion.md` from `agents/templates/completion.md`.
Do not run real experiments, commit or push.
