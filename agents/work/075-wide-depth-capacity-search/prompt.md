# Task Prompt

- Ticket: `075-wide-depth-capacity-search`
- Status: Done
- Completion: [Implementation and verification](completion.md).
- Delivery follow-up: The owner subsequently approved committing and pushing to `master`, with cleanup limited to disposable generated caches. Preserve evidence and legacy protocols.
- Aim: Measure ConvLSTM width/depth/shape trade-offs with a bounded search, then automatically compare validation-selected models at a practical shared input.
- Scope: Paper 001 search/comparison runners, versioned profiles, active notebook/export, tests and concise guides.
- Context: Execution approved on 10 October 2026, with two additions: cap search at **256 epochs** and show global campaign experiment progress instead of leaf `1/1`. Keep minimum 64/patience 24 and final cap 512. Final-input FPS and recipe checks also use 256/64/24 so reuse remains budget-compatible. Preserve earlier work; do not commit or push.
- History: [Superseded specification](prompt-superseded.md); [preserved evidence and planning answers](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/todays.results.md). No completed final comparison/test evidence is available in those excerpts.

## Latest validation evidence informing this revision

- The owner expanded [the console export](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/anotehr.update.md): 168 completed stage rows represent 164 unique printed configurations (four calibration reuses). Hash: `d958f356238ddd8e75fdfbb4b611dcd693e54f162bfd253be56084d445a83df6`. These are console-derived results, not independently verified predictions/checkpoints or test evidence.
- Across three sizes and two seeds, [64,64,64] averages **96.56%**, [32,32,32] **94.58%**, and [16,32,32] **90.31%**. R3D-18 averages **94.69%**; Swin3D-T stays at **12.50%**. Preserve convergence/recipe caveats rather than claim universal custom-model superiority.
- Width-32's 64px accuracy falls from **95.00% to 85.63%** across seeds; width-64 remains **94.38%/95.00%**. Refine the neighbourhood of width 64, retain both compact controls, and confirm across seeds.
- At 48px/16 FPS, moving 8 -> 16 frames gives [64,64,64] **98.13 -> 95.63%**, [32,32,32] **95.00 -> 32.50%**, and [16,32,32] **97.50 -> 65.63%**. Keep the requested final input, but measure transfer explicitly; longer input is not a proven improvement.
- WD=0 has the best three-size mean for all three customs; 1e-3 is much worse. FPS preference differs by family: 4 FPS helps R3D-18 but hurts width-64. Keep these controlled ablations and select FPS at the actual final input, not from the best mixed-stage console row.

## Decisions and candidate manifest

- **64 base custom runs = 32 architectures x two resolutions (32/64px), seed 42.** References, calibration, confirmation, ablations and final comparison are additional.
- Search input: **8 frames at 16 FPS**; final input: **16 frames at 64x64**, one validation-selected FPS shared by all eight models.
- Keep widths bounded at 128; add one five-layer width-64 control. No automatic expansion if the largest candidate wins.

| Group | Stacks | Count |
|---|---|---:|
| Flat width/depth ladder | Depth 1: widths 32/64/128; depths 2/3/4: widths 32/64/96/128 | 15 |
| Small/intermediate anchors | [8,8,8], [16,16,16], [48,48,48], [80,80,80] | 4 |
| Narrow one layer of the width-64 reference | [32,64,64], [64,32,64], [64,64,32] | 3 |
| Widen one layer to 96 | [96,64,64], [64,96,64], [64,64,96] | 3 |
| Widen one layer to 128 | [128,64,64], [64,128,64], [64,64,128] | 3 |
| Two-layer orientations | [32,64], [64,32] | 2 |
| Prior shaped control | [16,32,32] | 1 |
| Five-layer depth control | [64,64,64,64,64] | 1 |

Depth counts: **3 / 6 / 18 / 4 / 1** for depths 1-5. Fixed 3x3 kernels and the existing adaptive custom head. This is evidence-guided, not exhaustive or parameter-matched.

Four slot swaps, with no budget increase: replace [96], [32,32,64], [32,64,32] and [64,32,32] with [80,80,80] and the three single-layer 128-width placements above. This densifies the flat ladder near 64 and tests where extra capacity helps. Omitted candidates are deferred for budget allocation, not declared inferior; preserve all historical weak results. Width-64 still spans depths 1-5, widths 32/128 span depths 1-4, and width 96 spans depths 2-4.

## Changes

- Extend the existing runners. Use distinct new protocol/profile identities; preserve legacy 068/069/073/074 definitions, readers, results, raw logs, user edits and all saved notebook outputs. Replace the unfinished 075 implementation only within this revised scope; do not reuse old training results as new evidence.
- Keep AAD's existing 1,069 clips, 11-class mapping, 748/160/161 split, split seed 42, source-review/duplicate guards, Adam, gentle scheduler and augmentation off. No resplitting or new dataset. Search/reference recipe WD stays 0.
- Search training: cap **256 epochs**, minimum **64**, patience **24**, including calibration, confirmation and ablations. Calibration: [64,64,64], [128,128,128], R3D-18 and Swin3D-T at **8f/64px/16FPS**, LRs 0.001/0.003/0.01. Select one common custom LR and one LR per reference family using complete validation evidence.
- Run all 32 custom stacks and both references at 32/64px. Rank distinct customs by equal-resolution exact validation accuracy, then loss, parameters and stable name. Report every configuration, not only the five best rows.
- Confirm the provisional **top five**, flat three-layer widths 32/64/96/128, the five-layer width-64 control, and both references at both resolutions with seed **2026**; deduplicate overlaps. Re-rank only the provisional top five with equal resolution/seed weights; advance the best **three**. Other controls are stability evidence, not post-hoc entrants.
- Separate WD ablation: confirmed top three x both resolutions x {1e-4, 1e-3}; reuse WD=0 reference. Do not merge ablation scores into architecture ranking.
- Final-input FPS stage: confirmed top three plus both references x **{4,8,16 FPS}**, all at **16f/64px**, seed 42, calibrated family LR, WD=0 and the fixed comparison batch. Select one FPS by equal mean exact validation accuracy across the three custom models, then mean loss, then higher FPS. Report reference-family outcomes too; disclose that this shared setting is custom-tuned, not optimal tuning for every family.
- Reuse 074's sequential one-factor LR/WD validation on the search top-one custom at the chosen FPS and final input: incumbent LR 0.001/WD 0.0001, calibrated custom LR challenger, then search-proposed WD challenger. Keep the incumbent on a tie; reuse identical verified configurations. Freeze the final combined recipe only after validation, never by blindly combining separate ablation winners.
- Also retain that custom's already-completed selected-FPS job (calibrated LR/WD=0) as an eligible final-recipe candidate. Compare it with the sequential winner at the identical final input, batch, seed, split and **256/64/24** budget; accuracy then loss, ties retain the sequential incumbent. Verify architecture equivalence across display aliases and receipt provenance before reuse. Do not discard a better already-validated joint recipe merely because sequential LR/WD choices miss it; this adds **no new training** or untested hyperparameter combination.
- Show global unique experiment number, campaign upper bound and current stage in the runner and leaf/epoch progress; suppress misleading leaf `1/1` for campaign jobs. Reused configurations keep their original number and are explicitly marked reused; pending adaptive stages must not be counted as completed.
- Preflight full training steps before expensive work: custom bounds [128,128,128,128] and [64,64,64,64,64], both search references, and every final baseline at its actual planned input. Try batches 32/16/8/4/2/1 with documented headroom. Freeze **one common search batch** and **one common final-input batch**; the latter covers FPS checks, LR/WD checks and all final models. Record reductions; stop on failure rather than skip candidates or silently reduce their input. Differences between search/final phases are not a frame-only ablation.
- Final comparison: **three selected customs + input-adapted paper ConvLSTM + R3D-18/MC3-18 + Swin3D-T/S**, all from scratch at 16f/64px, selected FPS and frozen common recipe/batch. Cap **512 epochs**, minimum **128**, patience **32**, seed 42. Preserve the paper-topology layer design but adapt its input-dependent classifier, record the new count and use an explicit adapted name; keep the native published baseline unchanged in legacy profiles.
- Preserve automatic **Run All**: search -> confirmation/ablations -> final-input FPS/LR/WD validation -> eight fresh comparison trainings -> freeze all validation-selected checkpoints and best custom -> eight guarded test evaluations -> examples -> verified ZIP/download. No manual paths/toggles. Keep data/cache reuse, continuous unique experiment IDs, checkpoint/receipt compatibility and archive/retry guards.
- New main tables/console: accuracy, loss, parameters, model bytes, timing/memory and **unaveraged per-class precision/recall/F1/support**. Save TP/FP/FN, count confusion matrices and individual predictions. No macro-average columns/plots in this revised presentation; do not relabel micro metrics as raw. Preserve legacy macro fields/evidence and backward compatibility; retain full configs, histories and checkpoints.
- Evaluate unaugmented training and validation at the **same selected checkpoint in evaluation mode**. Save matched gaps, full learning curves, width/depth/layer-position summaries and accuracy-versus-parameters/cost plots, plus full appendix rows. Do not infer overfitting from validation rank alone or from online training metrics at a different epoch.
- Before comparison, display every final-input FPS result by architecture, with prediction-class counts and matched train/validation metrics; flag single-class predictions and scores at/below a training-majority-label baseline. Keep adverse results and label warnings as possible optimization failures, not proven capacity limits. Do not automatically remove/retrain losing models, switch inputs or promote extra candidates. In particular, the prior Swin3D-T result is not evidence against well-optimized transformers.
- Keep test locked for every selection. Never reset existing test-attempt guards or alter the live Colab run. If test outcomes have influenced this extension, disclose exposure and require a reviewed evaluation plan; do not claim a fresh holdout. Report a measured plateau/trade-off or boundary winner, not a guaranteed turning point/global optimum or unmeasured superiority/reliability.
- Update guides and resource warnings: 16x64x64 has 52.4% of the old 50x50x50 frame-pixels, but this is **not a runtime guarantee**. The adapted paper classifier is still large. Verified ZIPs must be saved outside the Colab VM before it is deleted.

## Bounded run plan

Maximum trainings **before exact-compatible reuse**, excluding synthetic preflight and metric passes:

| Stage | Maximum |
|---|---:|
| LR calibration | 12 |
| Base custom search: 32 x 2 | 64 |
| Two references x two resolutions | 4 |
| Confirmation: up to 10 customs + two references, second seed x two resolutions | 24 |
| Separate WD: three customs x two sizes x two nonzero values | 12 |
| Final-input FPS: three customs + two references x three FPS values | 15 |
| Final-input sequential LR/WD validation | 3 |
| Final comparison | 8 |
| **Full workflow** | **142** |

The **64** limit applies to the base custom sweep, not the entire workflow.
The phase/epoch caps are finite budgets, not a single-session or eight-hour promise.

- Acceptance criteria:
  - Exactly 32 distinct stacks and 64 base custom configurations; depths 1-5, bounds and all declared controls verified.
  - Two-seed top-five reranking, top-three handoff, fixed-input FPS selection and one frozen final recipe are reproducible from complete validation evidence; the recipe cannot score worse by its selection rule than a compatible selected-FPS top-one job already evaluated.
  - Synthetic Run All reaches all eight evaluations/examples/export without manual input; OOM, stale/incomplete evidence and repeat-test guards fail safely.
  - Per-class reporting and matched-checkpoint gaps reconcile with saved predictions; complete history/appendix evidence and stable experiment numbering remain available.
  - Old selections remain readable; previous raw-log hashes and notebook outputs stay unchanged. No real AAD training/test or new accuracy claim during implementation.
- Out of scope: More than 32 custom architectures, >128 filters, >5 layers, kernel changes, new datasets/splits, pretrained-baseline tuning, manuscript claims, real Colab execution, commit or push.
- Open questions: None for this replacement plan. Actual GPU safety/runtime and future results remain unmeasured.
- Verification: Manifest/count/ranking tests; five-layer CPU forward/backward; final-input FPS and recipe-selection tests including verified cross-alias reuse and joint-recipe retention; common-batch/OOM paths; matched-checkpoint/per-class metrics and constant-prediction warnings; synthetic full workflow/export; legacy/cache/reuse/test-guard regressions; notebook/export parity and saved-output/raw-log preservation; formatter, links and `git diff --check`.

## Execution Prompt

Complete the revised ticket `075-wide-depth-capacity-search` above. Follow
`AGENTS.md`, `agents/rules.md`, `agents/config.md` and this replacement prompt,
not the superseded specification. Preserve existing evidence, user changes and
saved notebook outputs. Verify the implementation, update status and create
`completion.md` from `agents/templates/completion.md`.
Do not run real experiments, commit or push.
