# AAD architecture search: interim analysis — 10 October 2026

## Bottom line

**[64,64,64] currently leads on mean validation accuracy (97.08%); [32,32,32]
is the strongest compact, resolution-consistent alternative (95.42%).**
The latter uses **74.72% fewer parameters** for **1.67 percentage points** less
mean accuracy. This is a promising accuracy–size trade-off, **not a demonstrated
interior optimum or an overfitting boundary**.

This report refines the supplied console snapshot only. In that snapshot, the
main architecture sweep is complete, but **the full search is not**: confirmation
is still running, and weight-decay/FPS/frame-count ablations are pending. The live
notebook may have progressed since this file was saved. No comparison or final
test results are included here.

[Original console output](2026-10-10-search-console.raw.txt) is preserved byte-for-byte.
Code revision: `e45756c`; NVIDIA A100-SXM4-40GB; AAD: 1,069 videos, 11 classes;
748 training / 160 validation / 161 locked test clips.

## What the snapshot establishes

| Item | Evidence available |
|---|---|
| Architecture sweep | 34 custom stacks × 3 resolutions = 102 configurations, seed 42 |
| Other completed stages | 12 LR-calibration configurations; 6 reference configurations |
| Count reconciliation | 120 stage rows represent 116 unique trainings: four calibration results are reused |
| Confirmation | Experiment 117 completed: [64,64,64], 32px, seed 2026, accuracy 96.25%; experiment 118 is still in progress |
| Pending search evidence | Remaining seed confirmation, weight decay, FPS/frame-count ablations, final frozen shortlist |
| Final comparison / test | Not present in this snapshot; do not infer their results |
| Evidence quality | Console-derived, not yet independently recomputed from downloaded predictions/checkpoints |

The shared custom sweep uses LR **0.003**, WD **0**, **8 frames at 16 FPS**,
32/48/64-pixel inputs and no augmentation. The search profile caps training at
128 epochs, minimum 64, patience 24; it requests batch 32, subject to the common
memory preflight. Verify the actual resolved batch and scheduler in the ZIP.
Individual checkpoints are selected by minimum validation loss.

The tables below average each architecture equally over its three resolutions,
using **only seed 42 at the common recipe**. They are not averages over different
LRs, repeated printouts or independent datasets. Rank by mean accuracy, then mean
loss, parameter count and name. Accuracy counts reconstructed from the printed
values are consistent with a 160-clip validation set; final artifact verification
must check those counts directly.

## Provisional top five architectures

| Rank | Architecture | Mean accuracy (%) | Mean macro-F1 (%) | Worst-resolution accuracy (%) | Parameters |
|---:|---|---:|---:|---:|---:|
| 1 | [64,64,64] | 97.08 | 97.11 | 94.38 | 745,675 |
| 2 | [32,32,32] | 95.42 | 95.67 | 95.00 | 188,523 |
| 3 | [16,32,32] | 90.63 | 90.11 | 76.25 | 140,651 |
| 4 | [32,32] | 90.00 | 90.17 | 84.38 | 114,667 |
| 5 | [32,32,16] | 89.79 | 89.15 | 79.38 | 142,203 |

These are **five distinct architectures**, not the five best individual rows.
The approved running workflow advances **three custom architectures**, with
two-seed confirmation determining their final order, alongside five baselines.
Showing a top-five search table does not change that protocol. Do not add two
comparison models or change selection after seeing test results.

## Proposed compact main-paper search table

All accuracy/F1 entries are percentages. This is a **representative, post-hoc
reporting selection**, not a new selection rule: show the full six-point
three-layer flat-width ladder, one-/two-layer controls at width 32, the two shaped
members of the provisional top five, and both search reference families. This
retains small/failed controls and the upper-boundary winner, not only favourable
results. The complete custom table and every completed training follow below.

| Role | Architecture | Parameters | Acc. 32px | Acc. 48px | Acc. 64px | Mean acc. | Mean macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Flat, very small | [4,4,4] | 3,415 | 33.75 | 37.50 | 25.62 | 32.29 | 17.59 |
| Flat, small | [8,8,8] | 12,579 | 53.75 | 49.38 | 45.63 | 49.58 | 40.16 |
| Flat, medium | [16,16,16] | 48,187 | 81.88 | 80.63 | 61.25 | 74.58 | 71.82 |
| Flat, medium | [24,24,24] | 106,835 | 91.88 | 90.00 | 77.50 | 86.46 | 86.16 |
| Flat, efficiency candidate | [32,32,32] | 188,523 | 96.25 | 95.00 | 95.00 | 95.42 | 95.67 |
| Flat, accuracy leader / upper boundary | [64,64,64] | 745,675 | 98.75 | 98.13 | 94.38 | 97.08 | 97.11 |
| One-layer depth control | [32] | 40,811 | 79.38 | 70.63 | 62.50 | 70.83 | 67.51 |
| Two-layer depth control | [32,32] | 114,667 | 91.88 | 93.75 | 84.38 | 90.00 | 90.17 |
| Expanding shape | [16,32,32] | 140,651 | 98.13 | 97.50 | 76.25 | 90.63 | 90.11 |
| Tapered shape | [32,32,16] | 142,203 | 95.63 | 94.38 | 79.38 | 89.79 | 89.15 |
| Search reference | R3D-18 | 33,171,915 | 94.38 | 95.63 | 93.13 | 94.38 | 94.50 |
| Search reference; convergence concern | Swin3D-T | 27,858,929 | 12.50 | 12.50 | 12.50 | 12.50 | 2.02 |

Custom models share LR 0.003; R3D-18 uses its calibrated LR 0.001 and Swin3D-T
uses 0.01. These are **search references**, not the final 50-frame comparison.
Do not interpret the failed Swin runs as proof that transformers are inherently
inferior. Its 12.50% accuracy and 2.02% macro-F1 are consistent with a collapsed,
single-class predictor; confirm that diagnosis from predictions/confusion matrices.

Once artifacts arrive, add macro precision/recall and measured inference/size
columns where useful. They are **not recoverable from accuracy and macro-F1
alone**, so this report does not invent them.

## What the results mean

- **Capacity helps through the tested upper boundary.** The three-layer flat
  means rise with widths 4 → 8 → 16 → 24 → 32 → 64:
  **32.29 → 49.58 → 74.58 → 86.46 → 95.42 → 97.08%**.
  There is no downturn beyond the best model: the widest tested flat stack wins.
  The smaller gain from 32 to 64 suggests diminishing accuracy returns per
  parameter, not a proved global optimum.
- **[32,32,32] is the compact stability candidate.** Its resolution scores are
  **96.25 / 95.00 / 95.00%**, a **1.25-point** range. [64,64,64] scores
  **98.75 / 98.13 / 94.38%**, a **4.38-point** range. At 64px, width 32 wins
  by only **one validation clip**; do not claim statistical superiority.
  Width 64 has **3.96×** the parameters. The provisional accuracy leader and the
  preferred efficiency compromise need not be the same architecture.
- **Depth matters under this recipe.** Holding width 32 fixed, one/two/three
  layers average **70.83 / 90.00 / 95.42%**. This is a depth-and-capacity
  comparison, not a parameter-matched proof that depth alone causes the gain.
- **Shapes interact with resolution.** [16,32,32] is excellent at 32/48px
  (**98.13 / 97.50%**) but falls to **76.25%** at 64px; [32,32,16] falls
  from **95.63%** at 32px to **79.38%** at 64px. Neither supports a claim of
  uniformly strong performance across sizes. Flat width 32 is the more stable
  of these compact choices.
- **Optimization is a real confounder.** At 48px, [24,24,24] reaches
  **56.88 / 90.00 / 95.00%** at LR 0.001 / 0.003 / 0.01. [64,64,64]
  reaches **88.75 / 98.13 / 70.63%** at the same LRs. Shared calibration
  improves comparability but does not optimize each architecture independently.
  Weak results cannot automatically be blamed on insufficient capacity or data.
- **Underfitting/overfitting is not established by these summary rows.**
  Low training and validation performance could support underfitting; sustained
  training improvement with worsening validation could support overfitting.
  Full curves at comparable budgets are needed. The lone partial progress line
  for experiment 118 is not enough to diagnose a model family.
- **Reference comparison is promising, provisional evidence.** R3D-18 averages
  **94.38% accuracy / 94.50% macro-F1** with **33.17M parameters**.
  [32,32,32] has about **176× fewer parameters**, but parameter count is not
  measured latency, and these search scores do not establish final superiority.

Do not merge yesterday's [16,24] results into this ranking: the older run used a
different learning-rate/sampling recipe. It remains historical evidence, not an
extra favourable replicate of today's search.

## Macro metrics: what should concern us?

**Micro precision = micro recall = micro-F1 = accuracy** here because each clip
has exactly one true and one predicted class. Those repeated console numbers are
expected, not four independent confirmations of quality.

**Macro-F1 averages the F1 score of each of the 11 classes equally**, so common
classes cannot dominate simply by having more videos. It is not the harmonic
mean of macro precision and macro recall. The implementation includes all declared
classes and assigns zero where a per-class denominator is zero; balanced accuracy
is mean recall over classes with support. See the
[metric implementation](../src/study_reporting.py) and
[scikit-learn averaging definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).

- [32,32,32]: mean accuracy **95.42%**, mean macro-F1 **95.67%**.
- [64,64,64]: **97.08%** and **97.11%**, respectively.
- [4]: **33.75%** and **17.86%**: the aggregate accuracy hides substantially
  weaker equal-class performance.
- Swin3D-T: **12.50%** and **2.02%**: a serious convergence/class-collapse warning.

Macro-F1 can legitimately be slightly above accuracy; the denominators and
weighting differ. Similar high means are encouraging, but do **not** establish
that every class is equally reliable. Read per-class recall/F1, support and
confusion matrices before making claims about safety-critical activities.
Each table's “mean macro-F1” is an average of three separately calculated
macro-F1 scores, **not** a pooled prediction score.

## Publication judgement and defensible conclusion

This is a useful search section for a PhD journal paper, **not sufficient evidence
by itself for a strong publication or acceptance guarantee**. A compact main table
plus a complete appendix is appropriate; selectively hiding weak runs, selecting
a favourable resolution for each model, or calling this an unbiased/global search
is not. The search was evidence-guided, and repeated validation selection can
produce optimistic estimates; final locked-test results are therefore essential.
[Cawley and Talbot, JMLR (2010)](https://www.jmlr.org/papers/v11/cawley10a.html)

Before manuscript use:

1. Verify the downloaded ZIP's configs, split hashes, predictions, checkpoints
   and complete job accounting. Do not treat this pasted log as an artifact audit.
2. Finish the planned confirmation and separate WD/temporal checks; report both
   seeds individually and their variation. Resolution spread is not seed variance.
3. Keep the existing validation-selected top-three comparison, freeze checkpoints,
   then evaluate the locked test. Do not change the search in response to test scores.
4. Retain the failed baselines and investigate their convergence before claiming
   a fair, competitive model-family comparison. The shared final recipe is not
   independently optimal for every family.
5. Disclose the small, repeatedly used validation set and clip-level split.
   Different filenames/owner-confirmed recordings do not establish subject/scene
   independence. Two seeds and a single final-comparison seed limit uncertainty claims.

Suggested search-only wording:

> Across the tested resolutions, the three-layer flat [64,64,64] ConvLSTM had
> the highest mean validation accuracy (97.08%). The smaller [32,32,32] model
> achieved 95.42% with 74.72% fewer parameters and a narrower resolution-dependent
> accuracy range. These results suggest a useful accuracy–size trade-off under
> the evaluated training protocol, while the upper-boundary winner and incomplete
> confirmation preclude a claim of globally optimal capacity or a demonstrated
> overfitting turning point.

## Appendix A — all 34 custom architectures

Seed 42, LR 0.003, WD 0, 8 frames at 16 FPS throughout.
Accuracy/F1 are percentages; loss is the mean of the three selected-checkpoint
validation losses. Printed macro-F1/loss values have limited precision.

| Rank | Architecture | Parameters | Acc. 32 / 48 / 64px (%) | Macro-F1 32 / 48 / 64px (%) | Mean acc. (%) | Mean macro-F1 (%) | Mean loss |
|---:|---|---:|---|---|---:|---:|---:|
| 1 | [64,64,64] | 745,675 | 98.75 / 98.13 / 94.38 | 98.75 / 98.10 / 94.50 | 97.08 | 97.11 | 0.3363 |
| 2 | [32,32,32] | 188,523 | 96.25 / 95.00 / 95.00 | 96.45 / 95.11 / 95.45 | 95.42 | 95.67 | 0.4180 |
| 3 | [16,32,32] | 140,651 | 98.13 / 97.50 / 76.25 | 98.10 / 97.53 / 74.71 | 90.63 | 90.11 | 0.4795 |
| 4 | [32,32] | 114,667 | 91.88 / 93.75 / 84.38 | 91.96 / 93.65 / 84.91 | 90.00 | 90.17 | 0.5083 |
| 5 | [32,32,16] | 142,203 | 95.63 / 94.38 / 79.38 | 95.43 / 94.49 / 77.55 | 89.79 | 89.15 | 0.5359 |
| 6 | [16,32,16] | 94,331 | 92.50 / 87.50 / 84.38 | 92.90 / 87.44 / 82.41 | 88.13 | 87.58 | 0.6121 |
| 7 | [32,8,16] | 66,075 | 93.13 / 90.00 / 77.50 | 93.10 / 89.67 / 76.64 | 86.88 | 86.47 | 0.7127 |
| 8 | [24,24,24] | 106,835 | 91.88 / 90.00 / 77.50 | 91.93 / 89.86 / 76.69 | 86.46 | 86.16 | 0.6585 |
| 9 | [32,16,32] | 123,947 | 93.13 / 88.13 / 75.63 | 93.66 / 88.17 / 72.93 | 85.63 | 84.92 | 0.6682 |
| 10 | [16,8,32] | 64,523 | 94.38 / 86.88 / 69.38 | 94.26 / 86.15 / 65.89 | 83.54 | 82.10 | 0.6199 |
| 11 | [32,16,16] | 86,843 | 87.50 / 90.00 / 71.88 | 87.64 / 89.87 / 68.45 | 83.13 | 81.99 | 0.7457 |
| 12 | [16,32] | 66,795 | 96.25 / 80.63 / 68.75 | 96.13 / 79.83 / 64.83 | 81.88 | 80.27 | 0.7986 |
| 13 | [16,16,32] | 85,291 | 88.75 / 82.50 / 73.75 | 88.97 / 82.35 / 72.72 | 81.67 | 81.35 | 0.7403 |
| 14 | [8,16,32] | 72,875 | 89.38 / 80.63 / 73.75 | 89.55 / 80.67 / 71.94 | 81.25 | 80.72 | 0.7756 |
| 15 | [24,24] | 65,267 | 93.75 / 74.38 / 72.50 | 93.83 / 72.53 / 70.41 | 80.21 | 78.92 | 0.8290 |
| 16 | [32,16,8] | 75,203 | 87.50 / 76.25 / 76.88 | 86.61 / 74.11 / 73.21 | 80.21 | 77.98 | 0.8429 |
| 17 | [8,32,16] | 77,307 | 85.00 / 83.13 / 68.75 | 85.63 / 81.06 / 66.57 | 78.96 | 77.75 | 0.8623 |
| 18 | [32,16] | 68,347 | 86.88 / 76.88 / 71.25 | 86.39 / 75.08 / 68.44 | 78.33 | 76.63 | 0.7915 |
| 19 | [16,24] | 45,939 | 86.88 / 71.88 / 73.75 | 86.70 / 70.48 / 72.06 | 77.50 | 76.41 | 0.7968 |
| 20 | [16,16,16] | 48,187 | 81.88 / 80.63 / 61.25 | 80.60 / 79.50 / 55.36 | 74.58 | 71.82 | 0.9151 |
| 21 | [24,16] | 46,715 | 77.50 / 76.25 / 68.75 | 76.45 / 74.66 / 63.11 | 74.17 | 71.41 | 0.9530 |
| 22 | [16,32,8] | 78,083 | 70.63 / 73.75 / 69.38 | 65.72 / 71.21 / 65.39 | 71.25 | 67.44 | 0.9355 |
| 23 | [16,16] | 29,691 | 79.38 / 68.75 / 65.00 | 78.16 / 64.53 / 59.68 | 71.04 | 67.46 | 0.9815 |
| 24 | [32] | 40,811 | 79.38 / 70.63 / 62.50 | 78.88 / 67.10 / 56.56 | 70.83 | 67.51 | 1.0343 |
| 25 | [16,8] | 18,051 | 65.00 / 55.00 / 53.13 | 56.64 / 45.81 / 43.64 | 57.71 | 48.69 | 1.2093 |
| 26 | [24] | 23,699 | 58.13 / 61.25 / 52.50 | 51.24 / 54.86 / 44.30 | 57.29 | 50.13 | 1.2397 |
| 27 | [8,16] | 17,275 | 64.38 / 58.13 / 46.25 | 58.87 / 52.02 / 36.80 | 56.25 | 49.23 | 1.1922 |
| 28 | [16] | 11,195 | 55.00 / 51.25 / 51.25 | 45.61 / 40.35 / 43.78 | 52.50 | 43.25 | 1.2835 |
| 29 | [8,8,8] | 12,579 | 53.75 / 49.38 / 45.63 | 45.44 / 39.73 / 35.32 | 49.58 | 40.16 | 1.3228 |
| 30 | [8,8] | 7,939 | 48.13 / 43.13 / 44.38 | 38.56 / 32.14 / 33.20 | 45.21 | 34.63 | 1.4453 |
| 31 | [8] | 3,299 | 40.63 / 41.88 / 36.88 | 28.69 / 30.09 / 19.47 | 39.79 | 26.08 | 1.6367 |
| 32 | [4,4] | 2,247 | 36.88 / 35.63 / 35.00 | 19.15 / 18.55 / 21.58 | 35.83 | 19.76 | 1.7184 |
| 33 | [4] | 1,079 | 33.75 / 35.63 / 31.87 | 17.40 / 19.09 / 17.08 | 33.75 | 17.86 | 1.8253 |
| 34 | [4,4,4] | 3,415 | 33.75 / 37.50 / 25.62 | 17.20 / 24.76 / 10.81 | 32.29 | 17.59 | 1.7642 |

## Appendix B — every completed unique training in the snapshot

**117 completed trainings; experiment 118 is unfinished and excluded from results.**
The table below includes all calibration settings and both reference families,
not just the custom-model ranking. There are no completed WD/temporal ablations
in the supplied snapshot. All completed jobs use WD 0 and 8 frames at 16 FPS.
Seed is 42 except experiment 117 (2026).

“Epoch” is the **selected checkpoint epoch**, not total epochs trained.
“Seconds” is the printed training/validation/checkpoint-loop time, not end-to-end
runtime or inference latency. The 117 rounded times sum to **14,433 seconds
(4 h 00 min 33 s)**; preparation, archives and final metric passes are excluded.

The four reused results are #2 (calibration → flat), #5 (calibration → flat),
#7 (calibration → references) and #12 (calibration → references).
Their repeated stage-table rows are not additional experiments.
Raw line numbers refer to [the preserved log](2026-10-10-search-console.raw.txt);
each experiment block contains its full run directory and trial ID.
“M-F1” means macro-F1. It is not printed for the final confirmation result, so
that entry remains unavailable rather than being inferred from accuracy.

| Exp. | Stage / reused role | Model | Pixels | LR | Seed | Params | Acc. (%) | M-F1 (%) | Val. loss | Epoch | Seconds | Raw line |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | calibration | [24,24,24] | 48 | 0.001 | 42 | 106,835 | 56.88 | 46.89 | 1.224780 | 128 | 126 | 127 |
| 2 | calibration + flat | [24,24,24] | 48 | 0.003 | 42 | 106,835 | 90.00 | 89.86 | 0.560937 | 128 | 123 | 165 |
| 3 | calibration | [24,24,24] | 48 | 0.01 | 42 | 106,835 | 95.00 | 95.13 | 0.451578 | 126 | 123 | 183 |
| 4 | calibration | [64,64,64] | 48 | 0.001 | 42 | 745,675 | 88.75 | 88.32 | 0.606074 | 127 | 270 | 201 |
| 5 | calibration + flat | [64,64,64] | 48 | 0.003 | 42 | 745,675 | 98.13 | 98.10 | 0.243361 | 110 | 270 | 219 |
| 6 | calibration | [64,64,64] | 48 | 0.01 | 42 | 745,675 | 70.63 | 67.60 | 0.917854 | 127 | 269 | 237 |
| 7 | calibration + references | R3D-18 | 48 | 0.001 | 42 | 33,171,915 | 95.63 | 95.89 | 0.197869 | 39 | 62 | 255 |
| 8 | calibration | R3D-18 | 48 | 0.003 | 42 | 33,171,915 | 92.50 | 92.34 | 0.276194 | 58 | 77 | 273 |
| 9 | calibration | R3D-18 | 48 | 0.01 | 42 | 33,171,915 | 91.88 | 91.86 | 0.577799 | 53 | 69 | 291 |
| 10 | calibration | Swin3D-T | 48 | 0.001 | 42 | 27,858,929 | 12.50 | 2.02 | 2.384240 | 102 | 252 | 309 |
| 11 | calibration | Swin3D-T | 48 | 0.003 | 42 | 27,858,929 | 12.50 | 2.02 | 2.384160 | 60 | 176 | 327 |
| 12 | calibration + references | Swin3D-T | 48 | 0.01 | 42 | 27,858,929 | 12.50 | 2.02 | 2.384140 | 56 | 170 | 345 |
| 13 | flat | [4] | 32 | 0.003 | 42 | 1,079 | 33.75 | 17.40 | 1.784650 | 128 | 59 | 380 |
| 14 | flat | [8] | 32 | 0.003 | 42 | 3,299 | 40.63 | 28.69 | 1.573390 | 124 | 60 | 418 |
| 15 | flat | [16] | 32 | 0.003 | 42 | 11,195 | 55.00 | 45.61 | 1.217940 | 124 | 60 | 436 |
| 16 | flat | [24] | 32 | 0.003 | 42 | 23,699 | 58.13 | 51.24 | 1.223580 | 128 | 61 | 454 |
| 17 | flat | [32] | 32 | 0.003 | 42 | 40,811 | 79.38 | 78.88 | 0.849583 | 127 | 60 | 472 |
| 18 | flat | [4,4] | 32 | 0.003 | 42 | 2,247 | 36.88 | 19.15 | 1.655840 | 125 | 80 | 490 |
| 19 | flat | [8,8] | 32 | 0.003 | 42 | 7,939 | 48.13 | 38.56 | 1.459060 | 128 | 80 | 508 |
| 20 | flat | [16,16] | 32 | 0.003 | 42 | 29,691 | 79.38 | 78.16 | 0.787877 | 127 | 82 | 526 |
| 21 | flat | [24,24] | 32 | 0.003 | 42 | 65,267 | 93.75 | 93.83 | 0.571561 | 124 | 82 | 544 |
| 22 | flat | [32,32] | 32 | 0.003 | 42 | 114,667 | 91.88 | 91.96 | 0.380879 | 123 | 84 | 562 |
| 23 | flat | [4,4,4] | 32 | 0.003 | 42 | 3,415 | 33.75 | 17.20 | 1.571940 | 128 | 101 | 580 |
| 24 | flat | [8,8,8] | 32 | 0.003 | 42 | 12,579 | 53.75 | 45.44 | 1.284970 | 128 | 99 | 598 |
| 25 | flat | [16,16,16] | 32 | 0.003 | 42 | 48,187 | 81.88 | 80.60 | 0.701769 | 127 | 103 | 616 |
| 26 | flat | [24,24,24] | 32 | 0.003 | 42 | 106,835 | 91.88 | 91.93 | 0.462292 | 128 | 102 | 634 |
| 27 | flat | [32,32,32] | 32 | 0.003 | 42 | 188,523 | 96.25 | 96.45 | 0.305760 | 120 | 106 | 652 |
| 28 | flat | [64,64,64] | 32 | 0.003 | 42 | 745,675 | 98.75 | 98.75 | 0.330738 | 95 | 135 | 670 |
| 29 | flat | [4] | 48 | 0.003 | 42 | 1,079 | 35.63 | 19.09 | 1.830150 | 128 | 63 | 688 |
| 30 | flat | [8] | 48 | 0.003 | 42 | 3,299 | 41.88 | 30.09 | 1.614920 | 125 | 64 | 706 |
| 31 | flat | [16] | 48 | 0.003 | 42 | 11,195 | 51.25 | 40.35 | 1.294830 | 128 | 65 | 724 |
| 32 | flat | [24] | 48 | 0.003 | 42 | 23,699 | 61.25 | 54.86 | 1.184520 | 124 | 66 | 742 |
| 33 | flat | [32] | 48 | 0.003 | 42 | 40,811 | 70.63 | 67.10 | 1.088630 | 125 | 71 | 760 |
| 34 | flat | [4,4] | 48 | 0.003 | 42 | 2,247 | 35.63 | 18.55 | 1.763120 | 128 | 84 | 778 |
| 35 | flat | [8,8] | 48 | 0.003 | 42 | 7,939 | 43.13 | 32.14 | 1.456330 | 123 | 85 | 796 |
| 36 | flat | [16,16] | 48 | 0.003 | 42 | 29,691 | 68.75 | 64.53 | 1.068000 | 127 | 89 | 814 |
| 37 | flat | [24,24] | 48 | 0.003 | 42 | 65,267 | 74.38 | 72.53 | 0.917187 | 124 | 96 | 832 |
| 38 | flat | [32,32] | 48 | 0.003 | 42 | 114,667 | 93.75 | 93.65 | 0.392616 | 124 | 109 | 850 |
| 39 | flat | [4,4,4] | 48 | 0.003 | 42 | 3,415 | 37.50 | 24.76 | 1.758030 | 126 | 103 | 868 |
| 40 | flat | [8,8,8] | 48 | 0.003 | 42 | 12,579 | 49.38 | 39.73 | 1.320910 | 123 | 108 | 886 |
| 41 | flat | [16,16,16] | 48 | 0.003 | 42 | 48,187 | 80.63 | 79.50 | 0.828874 | 125 | 114 | 904 |
| 42 | flat | [32,32,32] | 48 | 0.003 | 42 | 188,523 | 95.00 | 95.11 | 0.375963 | 128 | 149 | 923 |
| 43 | flat | [4] | 64 | 0.003 | 42 | 1,079 | 31.87 | 17.08 | 1.861200 | 127 | 74 | 942 |
| 44 | flat | [8] | 64 | 0.003 | 42 | 3,299 | 36.88 | 19.47 | 1.721790 | 126 | 75 | 980 |
| 45 | flat | [16] | 64 | 0.003 | 42 | 11,195 | 51.25 | 43.78 | 1.337600 | 127 | 78 | 998 |
| 46 | flat | [24] | 64 | 0.003 | 42 | 23,699 | 52.50 | 44.30 | 1.311080 | 122 | 85 | 1016 |
| 47 | flat | [32] | 64 | 0.003 | 42 | 40,811 | 62.50 | 56.56 | 1.164720 | 128 | 98 | 1034 |
| 48 | flat | [4,4] | 64 | 0.003 | 42 | 2,247 | 35.00 | 21.58 | 1.736310 | 127 | 92 | 1052 |
| 49 | flat | [8,8] | 64 | 0.003 | 42 | 7,939 | 44.38 | 33.20 | 1.420630 | 126 | 93 | 1070 |
| 50 | flat | [16,16] | 64 | 0.003 | 42 | 29,691 | 65.00 | 59.68 | 1.088580 | 126 | 128 | 1088 |
| 51 | flat | [24,24] | 64 | 0.003 | 42 | 65,267 | 72.50 | 70.41 | 0.998146 | 127 | 135 | 1106 |
| 52 | flat | [32,32] | 64 | 0.003 | 42 | 114,667 | 84.38 | 84.91 | 0.751405 | 128 | 164 | 1124 |
| 53 | flat | [4,4,4] | 64 | 0.003 | 42 | 3,415 | 25.62 | 10.81 | 1.962770 | 128 | 113 | 1142 |
| 54 | flat | [8,8,8] | 64 | 0.003 | 42 | 12,579 | 45.63 | 35.32 | 1.362520 | 126 | 114 | 1160 |
| 55 | flat | [16,16,16] | 64 | 0.003 | 42 | 48,187 | 61.25 | 55.36 | 1.214540 | 128 | 184 | 1178 |
| 56 | flat | [24,24,24] | 64 | 0.003 | 42 | 106,835 | 77.50 | 76.69 | 0.952274 | 124 | 188 | 1196 |
| 57 | flat | [32,32,32] | 64 | 0.003 | 42 | 188,523 | 95.00 | 95.45 | 0.572175 | 125 | 232 | 1214 |
| 58 | flat | [64,64,64] | 64 | 0.003 | 42 | 745,675 | 94.38 | 94.50 | 0.434665 | 128 | 449 | 1232 |
| 59 | references | R3D-18 | 32 | 0.001 | 42 | 33,171,915 | 94.38 | 94.51 | 0.219528 | 18 | 51 | 1315 |
| 60 | references | Swin3D-T | 32 | 0.01 | 42 | 27,858,929 | 12.50 | 2.02 | 2.384180 | 116 | 268 | 1333 |
| 61 | references | R3D-18 | 64 | 0.001 | 42 | 33,171,915 | 93.13 | 93.12 | 0.276306 | 31 | 80 | 1353 |
| 62 | references | Swin3D-T | 64 | 0.01 | 42 | 27,858,929 | 12.50 | 2.02 | 2.384180 | 60 | 207 | 1371 |
| 63 | refinement | [8,16] | 32 | 0.003 | 42 | 17,275 | 64.38 | 58.87 | 1.067780 | 125 | 87 | 1460 |
| 64 | refinement | [16,8] | 32 | 0.003 | 42 | 18,051 | 65.00 | 56.64 | 1.133470 | 126 | 87 | 1478 |
| 65 | refinement | [16,24] | 32 | 0.003 | 42 | 45,939 | 86.88 | 86.70 | 0.574148 | 128 | 87 | 1496 |
| 66 | refinement | [24,16] | 32 | 0.003 | 42 | 46,715 | 77.50 | 76.45 | 0.812399 | 124 | 87 | 1514 |
| 67 | refinement | [16,32] | 32 | 0.003 | 42 | 66,795 | 96.25 | 96.13 | 0.437498 | 124 | 89 | 1532 |
| 68 | refinement | [32,16] | 32 | 0.003 | 42 | 68,347 | 86.88 | 86.39 | 0.545051 | 128 | 88 | 1550 |
| 69 | refinement | [16,16,32] | 32 | 0.003 | 42 | 85,291 | 88.75 | 88.97 | 0.532032 | 122 | 108 | 1568 |
| 70 | refinement | [16,32,16] | 32 | 0.003 | 42 | 94,331 | 92.50 | 92.90 | 0.432079 | 126 | 108 | 1586 |
| 71 | refinement | [32,16,16] | 32 | 0.003 | 42 | 86,843 | 87.50 | 87.64 | 0.612459 | 128 | 108 | 1604 |
| 72 | refinement | [16,32,32] | 32 | 0.003 | 42 | 140,651 | 98.13 | 98.10 | 0.227409 | 126 | 108 | 1622 |
| 73 | refinement | [32,16,32] | 32 | 0.003 | 42 | 123,947 | 93.13 | 93.66 | 0.490734 | 122 | 107 | 1640 |
| 74 | refinement | [32,32,16] | 32 | 0.003 | 42 | 142,203 | 95.63 | 95.43 | 0.241536 | 126 | 110 | 1658 |
| 75 | refinement | [8,16,32] | 32 | 0.003 | 42 | 72,875 | 89.38 | 89.55 | 0.605333 | 126 | 107 | 1676 |
| 76 | refinement | [8,32,16] | 32 | 0.003 | 42 | 77,307 | 85.00 | 85.63 | 0.730901 | 125 | 107 | 1694 |
| 77 | refinement | [16,8,32] | 32 | 0.003 | 42 | 64,523 | 94.38 | 94.26 | 0.307597 | 120 | 107 | 1712 |
| 78 | refinement | [16,32,8] | 32 | 0.003 | 42 | 78,083 | 70.63 | 65.72 | 0.888513 | 128 | 107 | 1730 |
| 79 | refinement | [32,8,16] | 32 | 0.003 | 42 | 66,075 | 93.13 | 93.10 | 0.536653 | 128 | 107 | 1748 |
| 80 | refinement | [32,16,8] | 32 | 0.003 | 42 | 75,203 | 87.50 | 86.61 | 0.749192 | 128 | 108 | 1766 |
| 81 | refinement | [8,16] | 48 | 0.003 | 42 | 17,275 | 58.13 | 52.02 | 1.183360 | 128 | 89 | 1784 |
| 82 | refinement | [16,8] | 48 | 0.003 | 42 | 18,051 | 55.00 | 45.81 | 1.217710 | 125 | 89 | 1802 |
| 83 | refinement | [16,24] | 48 | 0.003 | 42 | 45,939 | 71.88 | 70.48 | 0.900385 | 128 | 95 | 1820 |
| 84 | refinement | [24,16] | 48 | 0.003 | 42 | 46,715 | 76.25 | 74.66 | 0.952259 | 127 | 96 | 1838 |
| 85 | refinement | [16,32] | 48 | 0.003 | 42 | 66,795 | 80.63 | 79.83 | 0.886603 | 127 | 100 | 1856 |
| 86 | refinement | [32,16] | 48 | 0.003 | 42 | 68,347 | 76.88 | 75.08 | 0.816307 | 125 | 104 | 1874 |
| 87 | refinement | [16,16,32] | 48 | 0.003 | 42 | 85,291 | 82.50 | 82.35 | 0.728885 | 127 | 125 | 1892 |
| 88 | refinement | [16,32,16] | 48 | 0.003 | 42 | 94,331 | 87.50 | 87.44 | 0.673791 | 127 | 130 | 1910 |
| 89 | refinement | [32,16,16] | 48 | 0.003 | 42 | 86,843 | 90.00 | 89.87 | 0.607894 | 128 | 129 | 1928 |
| 90 | refinement | [16,32,32] | 48 | 0.003 | 42 | 140,651 | 97.50 | 97.53 | 0.407027 | 127 | 137 | 1946 |
| 91 | refinement | [32,16,32] | 48 | 0.003 | 42 | 123,947 | 88.13 | 88.17 | 0.688793 | 124 | 139 | 1964 |
| 92 | refinement | [32,32,16] | 48 | 0.003 | 42 | 142,203 | 94.38 | 94.49 | 0.503276 | 128 | 142 | 1982 |
| 93 | refinement | [8,16,32] | 48 | 0.003 | 42 | 72,875 | 80.63 | 80.67 | 0.781516 | 126 | 117 | 2000 |
| 94 | refinement | [8,32,16] | 48 | 0.003 | 42 | 77,307 | 83.13 | 81.06 | 0.788631 | 127 | 122 | 2018 |
| 95 | refinement | [16,8,32] | 48 | 0.003 | 42 | 64,523 | 86.88 | 86.15 | 0.629104 | 125 | 118 | 2036 |
| 96 | refinement | [16,32,8] | 48 | 0.003 | 42 | 78,083 | 73.75 | 71.21 | 0.916110 | 126 | 125 | 2054 |
| 97 | refinement | [32,8,16] | 48 | 0.003 | 42 | 66,075 | 90.00 | 89.67 | 0.681688 | 128 | 123 | 2072 |
| 98 | refinement | [32,16,8] | 48 | 0.003 | 42 | 75,203 | 76.25 | 74.11 | 0.856949 | 128 | 126 | 2090 |
| 99 | refinement | [8,16] | 64 | 0.003 | 42 | 17,275 | 46.25 | 36.80 | 1.325550 | 126 | 96 | 2108 |
| 100 | refinement | [16,8] | 64 | 0.003 | 42 | 18,051 | 53.13 | 43.64 | 1.276580 | 126 | 100 | 2126 |
| 101 | refinement | [16,24] | 64 | 0.003 | 42 | 45,939 | 73.75 | 72.06 | 0.915761 | 127 | 121 | 2144 |
| 102 | refinement | [24,16] | 64 | 0.003 | 42 | 46,715 | 68.75 | 63.11 | 1.094490 | 128 | 127 | 2162 |
| 103 | refinement | [16,32] | 64 | 0.003 | 42 | 66,795 | 68.75 | 64.83 | 1.071700 | 123 | 132 | 2180 |
| 104 | refinement | [32,16] | 64 | 0.003 | 42 | 68,347 | 71.25 | 68.44 | 1.013020 | 125 | 142 | 2198 |
| 105 | refinement | [16,16,32] | 64 | 0.003 | 42 | 85,291 | 73.75 | 72.72 | 0.959972 | 127 | 185 | 2216 |
| 106 | refinement | [16,32,16] | 64 | 0.003 | 42 | 94,331 | 84.38 | 82.41 | 0.730575 | 127 | 176 | 2234 |
| 107 | refinement | [32,16,16] | 64 | 0.003 | 42 | 86,843 | 71.88 | 68.45 | 1.016870 | 128 | 197 | 2252 |
| 108 | refinement | [16,32,32] | 64 | 0.003 | 42 | 140,651 | 76.25 | 74.71 | 0.804154 | 127 | 197 | 2270 |
| 109 | refinement | [32,16,32] | 64 | 0.003 | 42 | 123,947 | 75.63 | 72.93 | 0.824968 | 128 | 201 | 2288 |
| 110 | refinement | [32,32,16] | 64 | 0.003 | 42 | 142,203 | 79.38 | 77.55 | 0.862848 | 126 | 210 | 2306 |
| 111 | refinement | [8,16,32] | 64 | 0.003 | 42 | 72,875 | 73.75 | 71.94 | 0.939933 | 125 | 153 | 2324 |
| 112 | refinement | [8,32,16] | 64 | 0.003 | 42 | 77,307 | 68.75 | 66.57 | 1.067310 | 125 | 163 | 2342 |
| 113 | refinement | [16,8,32] | 64 | 0.003 | 42 | 64,523 | 69.38 | 65.89 | 0.923120 | 124 | 151 | 2360 |
| 114 | refinement | [16,32,8] | 64 | 0.003 | 42 | 78,083 | 69.38 | 65.39 | 1.001820 | 125 | 165 | 2378 |
| 115 | refinement | [32,8,16] | 64 | 0.003 | 42 | 66,075 | 77.50 | 76.64 | 0.919641 | 127 | 163 | 2396 |
| 116 | refinement | [32,16,8] | 64 | 0.003 | 42 | 75,203 | 76.88 | 73.21 | 0.922628 | 125 | 167 | 2414 |
| 117 | confirmation | [64,64,64] | 32 | 0.003 | 2026 | 745,675 | 96.25 | not printed | 0.2378 | 115 | 149 | 2557 |

Experiment 118: [32,32,32], 32px, seed 2026; shown at epoch 116/128.
Its displayed training accuracy 98.1% and validation accuracy 93.8% describe
that in-progress epoch, **not a completed selected-checkpoint result**. Leave
it out of rankings until its final metrics are saved.

## Evidence and verification record

- Original snapshot: `2026-10-10-search-console.raw.txt`, 293,101 bytes.
- SHA-256: `cde28bc1fc132205d25c2afb6fed3d5de7a1bd21a61733d318a00ff157f5f7cf`.
- Last complete stage table: original lines 2434–2553 (120 rows).
- Independent parsing checked 34 custom triplets, 6 reference rows, 12 calibration
  rows, 116 unique base trainings, the completed confirmation #117, and incomplete
  #118. All 120 printed stage rows map to exactly one completed training each.
- Cross-resolution means use reconstructed integer correct counts (denominator
  160); loss and macro-F1 use the printed values. Repeated calibration rows and
  the extra seed are not double-counted in the seed-42 ranking.
- The raw snapshot is unchanged; no notebook, configuration, experiment or
  selection code was modified. No commit or push was made.

## Later update: completed search, slow comparison preparation — 10 October 2026

Source: [new console excerpt](anotehr.update.md), saved at 15:12 local time.
This supplements the earlier snapshot; it does not replace its evidence.
Completion is reported by the console, not independently verified from a ZIP here.

### Latest available best-model results

The new excerpt does **not** contain the completed two-seed ranking, ablation
tables or selected-config contents. These remain the **earlier seed-42,
common-recipe** results, not newly verified final winners. Correct counts are
reconstructed from that earlier output. Each resolution evaluates the same
160 validation clips; the three columns are **not 480 independent videos**.
No macro averages are added to this update.

| Provisional rank | Architecture | Correct at 32px | Correct at 48px | Correct at 64px | Mean accuracy | Parameters |
|---:|---|---:|---:|---:|---:|---:|
| 1 | [64,64,64] | 158/160 | 157/160 | 151/160 | 97.08% | 745,675 |
| 2 | [32,32,32] | 154/160 | 152/160 | 152/160 | 95.42% | 188,523 |
| 3 | [16,32,32] | 157/160 | 156/160 | 122/160 | 90.63% | 140,651 |
| 4 | [32,32] | 147/160 | 150/160 | 135/160 | 90.00% | 114,667 |
| 5 | [32,32,16] | 153/160 | 151/160 | 127/160 | 89.79% | 142,203 |

The new job reports `custom_top1` with 745,675 parameters, **consistent with
[64,64,64]**. This inference does not replace checking `selected_config.json`.
Final top-two/top-three identities, confirmation means and test performance are
not recoverable from this excerpt.

### What progressed, and what has not

- Search reports completion and a saved validation-only top-three selection.
  The console reports verified ZIPs of approximately 6.60 GB; their contents
  and successful download have not been independently checked in this review.
- Run All automatically enters LR/WD recipe validation before eight final
  comparison trainings. No manual handoff is requested in the excerpt.
- Experiment **165** is a **recipe check**, not a completed final comparison:
  `custom_top1`, LR 0.001, WD 0.0001, batch 1, **50 frames at 50×50**,
  cap 128 epochs. Caching finishes at 908/908 train/validation clips.
- At **29/128 epochs**, elapsed **43:42**, it shows training accuracy **10.7%**,
  validation accuracy **8.1%**, training loss **2.407**, validation loss **2.420**,
  LR **0.000729**, selected epoch **11**. These are current-epoch displays,
  not final selected-checkpoint metrics. Uniform predictions over 11 classes
  have cross-entropy ln(11) ≈ **2.398** and expected accuracy **9.1%**.
- This job is not showing useful learning at that point. It does not establish
  a broken dataset or an incapable architecture. Frames, spatial input, batch,
  LR and WD differ from successful search: this is not a one-factor comparison.
- At the displayed **90.72 seconds/epoch**, 128 epochs would take about
  **3.23 hours**, and 512 about **12.90 hours**, if that rate persisted.
  These are this job's extrapolations, excluding extra work—not measured
  full runtimes or estimates for every architecture.
- No completed recipe decision, final comparison score or test result appears.
  Keep test locked; do not infer final accuracy from the search.

### Faster follow-up proposal — pending approval, not implemented

1. **Final input:** propose **8 frames at 16 FPS, 32×32 pixels**, with one
   memory-safe common batch and one validation-checked recipe for all models.
   The earlier leading three all achieved at least 96.25% at 32px. This supports
   a practical validation-led choice, not universal optimality.
2. **Search:** retain the proposed **31 architectures**, width at most 128 and
   depths **1/2/3/4**, but consider **32/48px** instead of three resolutions.
   This reduces base custom configurations from **93 to 62**, not necessarily
   total runtime by one third. Keep small anchors, flat width/depth ladders and
   targeted layer-position changes. Do not remove all weak controls. Preserve
   older 64px evidence separately; do not count it as new-protocol runs.
3. **Turning point:** a plateau or useful accuracy/cost trade-off is enough.
   If the widest model still wins, report the boundary honestly rather than
   expanding indefinitely or manufacturing a downturn. Overfitting requires
   matched-checkpoint training/validation evidence and full curves, not rank
   alone. Disclose that this follow-up is guided by earlier validation results.
4. **Published baseline:** the repository's paper-topology classifier flattens
   the full temporal/spatial tensor. Reducing input changes its parameter count.
   A fast version must be labelled **input-adapted paper topology**, not an
   exact published reproduction. Alternatively keep a separate slow native-input
   experiment. This decision needs approval.
5. **No macro columns in future requested displays:** propose overall
   accuracy/loss, **per-class precision/recall/F1 and support**, TP/FP/FN,
   count confusion matrices and individual prediction records, plus parameters,
   model size and timing. Per-class metrics are unaveraged (`average=None`).
   A single multiclass precision/recall/F1 needs an averaging rule; micro values
   equal accuracy here, so calling them “raw” does not make them independent.
   See [per-class metric definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
   and [micro versus accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).
   Preserve all previously recorded metrics. This excerpt provides no per-class
   results to populate yet.
6. Preserve automatic search → selection → recipe checks → comparison →
   validation freeze → guarded test → examples → ZIP. The longer comparison
   epoch budget remains unless separately revised. Faster inputs alone do not
   guarantee that eight models finish in one session.

Ticket 075 was already being implemented when this planning request arrived.
Its local code is **paused for revised decisions, not complete or fully verified**.
Earlier code edits remain uncommitted. No new experiment code was changed for
this update, no real experiments were run, and nothing was committed or pushed.
Do not treat the unverified working tree as ready for Colab.

Preserved new-excerpt SHA-256:
`b87bcebddfe9eef6de89e6036ff6b2c1afe024141285693616299944de859478`.

Preserved earlier raw-log SHA-256:
`cde28bc1fc132205d25c2afb6fed3d5de7a1bd21a61733d318a00ff157f5f7cf`.

### Planning answers received — 10 October 2026

These supersede the proposed 8-frame/32px final input above; they are not new
results or an implemented replacement protocol.

- User prefers **16 frames at 64x64** for final comparison, and FPS chosen using
  validation evidence. Recommend checking 4/8/16 FPS at this exact final input
  before freezing one shared FPS; an 8-frame ablation winner is not evidence
  for the untested 16-frame combination. Test remains locked.
- User accepts an **input-adapted paper-topology baseline** and wants five-layer
  controls alongside 1-4-layer models. All final models should share the same
  input and sampling recipe. Adapted results do not establish an exact published
  reproduction or general superiority across untested inputs/datasets.
- Reporting choice confirmed: accuracy/loss, unaveraged per-class
  precision/recall/F1 and support, confusion counts and individual predictions;
  no macro columns in new main tables. Preserve historical metrics unchanged.
- **64 configurations remains ambiguous:** 32 architectures at two resolutions
  give 64 base runs; 64 architectures give 128 base runs. Confirmation, FPS/WD
  checks and final comparison are extra. Search input proposal awaiting agreement:
  32/64px, with either an 8-frame sweep or 16 frames throughout.
- A 16x64x64 clip contains **52.4%** as many frame-pixels as 50x50x50, but
  **eight times** as many as the previously proposed 8x32x32. This is an input
  volume calculation, not a runtime or memory guarantee; deeper models add cost.
- The objective remains a measured accuracy/size/stability trade-off within a
  declared budget. A plateau or boundary winner is a valid outcome; a downturn
  cannot be required in advance. Expanding a validation-guided search also risks
  overfitting model selection itself; keep the independent test locked and retain
  confirmation evidence. See [Cawley and Talbot's model-selection study](https://www.jmlr.org/papers/v11/cawley10a.html).

Only planning notes were updated for these answers. Ticket 075 remains paused
and incompletely verified; no new code changes, experiments, commit or push.

### Search budget resolved — replacement prompt ready

- User confirmed **64 base custom runs**, not 64 distinct architectures:
  **32 stacks x 32/64px**, with **8 frames** in the architecture sweep.
- The replacement proposal adds **[64,64,64,64,64]** to the earlier 31 stacks:
  4 one-layer, 6 two-layer, 17 three-layer, 4 four-layer and 1 five-layer model.
  Width remains capped at 128; the five-layer model is a matched width-64 depth
  control, not an automatic expansion to every five-layer shape.
- Final input is **16 frames at 64x64**. Proposed FPS validation compares
  4/8/16 FPS at that exact input, selects one common value from validation, then
  checks LR/WD before freezing one recipe for all eight final models.
- Revised upper bound: **142 trainings** before reuse, including calibration,
  confirmation, WD/FPS checks and final comparison. This is not a runtime promise.
- See the [replacement execution prompt](../../../../agents/work/075-wide-depth-capacity-search/prompt.md)
  for the exact manifest, deterministic selection rules, resource checks and
  verification requirements. The earlier specification is archived, not erased.
  No new experiment code was implemented for this planning revision; the partial
  working tree remains unverified until the user supplies the revised prompt.

## Completed search ranking reviewed — 10 October 2026, 15:33 export

The owner expanded [anotehr.update.md](anotehr.update.md) after the earlier review.
Its current SHA-256 is
`d958f356238ddd8e75fdfbb4b611dcd693e54f162bfd253be56084d445a83df6`;
the earlier `b87b...` hash above identifies the previous shorter version,
not the current file. The original long morning log remains unchanged.

### Reconstructed confirmation results

Parsed **168 complete stage rows**, ranks 1-168 with no gaps. Four identical
calibration configurations reappear in the base matrix: **164 unique printed
training configurations**, consistent with the subsequent recipe job being #165.
Counts: calibration 12, flat 48, refinement 54, references 6, confirmation 15,
WD 18, temporal 15. All accuracy values reconcile to integer counts out of 160.

The table below uses only common-recipe base/confirmation results, equally
weighted over 32/48/64px and seeds 42/2026. It excludes calibration duplicates,
WD/FPS/frame-count ablations and all test results. These remain **console-derived
validation scores**, not independent recomputations from prediction/checkpoint
artifacts. Repeated seeds/resolutions evaluate the same validation clips.

| Model | Seed 42 mean accuracy (%) | Seed 2026 mean accuracy (%) | Two-seed mean accuracy (%) | Parameters |
|---|---:|---:|---:|---:|
| [64,64,64] | 97.08 | 96.04 | 96.56 | 745,675 |
| [32,32,32] | 95.42 | 93.75 | 94.58 | 188,523 |
| [16,32,32] | 90.63 | 90.00 | 90.31 | 140,651 |
| R3D-18 | 94.38 | 95.00 | 94.69 | 33,171,915 |
| Swin3D-T | 12.50 | 12.50 | 12.50 | 27,858,929 |

- **[64,64,64] remains the strongest confirmed custom**, not just the best
  isolated row. Its two-seed mean exceeds R3D-18 by **1.875 percentage points**,
  with approximately **44.5x fewer parameters**. This is search-protocol evidence,
  not a significance test, measured speedup or final comparison claim.
- **[32,32,32] is compact but less stable than the first snapshot suggested**:
  at 64px it changes from **95.00% to 85.63%** across seeds. Width-64 gives
  **94.38%/95.00%** there. Preserve the earlier favourable seed, but do not
  continue calling width-32 uniformly stable.
- **[16,32,32]** remains a useful compact shaped control; its 64px scores are
  **76.25%/81.25%**. Strong low-resolution scores do not transfer uniformly.
- Only three custom architectures were confirmed in the old protocol. The new
  two-seed custom order is reconstructed for those three; do not relabel it a
  confirmed top-five ranking over all 34 architectures.

### Weight decay and temporal findings

WD table: mean validation accuracy across three resolutions, seed 42, 8f/16FPS,
custom LR 0.003. Zero is the reused common-reference recipe.

| Architecture | WD 0 | WD 1e-4 | WD 1e-3 |
|---|---:|---:|---:|
| [64,64,64] | 97.08 | 96.25 | 59.79 |
| [32,32,32] | 95.42 | 87.29 | 61.67 |
| [16,32,32] | 90.63 | 88.96 | 62.08 |

WD=0 leads on the three-size mean for each custom. However, WD=1e-4 improves
width-64 at 64px from **94.38% to 97.50%**; do not generalize the mean result
to every input. Keep all three levels, including the adverse control.

Temporal table: accuracy (%) at **48px, seed 42, WD=0**, each family's calibrated LR.
Each FPS column at eight frames can be compared with 8f/16FPS; the last column
changes frame count at fixed 16 FPS. The future 16f/64px/4-or-8FPS combinations
are **not** measured in these data.

| Model | 8f/16FPS | 8f/8FPS | 8f/4FPS | 16f/16FPS |
|---|---:|---:|---:|---:|
| [64,64,64] | 98.13 | 98.13 | 43.13 | 95.63 |
| [32,32,32] | 95.00 | 87.50 | 92.50 | 32.50 |
| [16,32,32] | 97.50 | 93.75 | 93.75 | 65.63 |
| R3D-18 | 95.63 | 95.00 | 98.13 | 96.25 |
| Swin3D-T | 12.50 | 12.50 | 12.50 | 12.50 |

- Longer clips are not automatically better: width-32 falls **62.50 points**
  and [16,32,32] **31.875 points** at 16 frames; width-64 falls only **2.50**.
  Full curves and matched training metrics are needed to explain why.
- R3D-18 benefits from 4 FPS here, whereas width-64 falls to **43.13%**.
  Choose one shared FPS from validation at the actual final input, not the
  highest-scoring row across different models/settings.
- Swin3D-T remains at **12.50%** throughout these settings. That warrants a
  convergence/prediction-distribution check; it is not evidence that an optimized
  transformer is generally inferior. Individual predictions are unavailable in
  this excerpt, so single-class collapse is not confirmed.
- The updated comparison-preparation progress is **46/128 epochs**, **1:09:40**
  elapsed, training accuracy **10.6%**, validation **12.5%**. It is still a
  50-frame recipe check, not the new 16-frame protocol or a final test result.

### Prompt refinements within the same budget

The [updated 075 prompt](../../../../agents/work/075-wide-depth-capacity-search/prompt.md)
still declares **32 architectures / 64 base runs / up to 142 total trainings**.

- Add **[80,80,80]** between 64 and 96 to sample the flat-width curve more closely.
- Add **[128,64,64], [64,128,64], [64,64,128]** alongside the existing 96-width
  placements. This tests where extra capacity helps around the confirmed leader.
- Replace **[96], [32,32,64], [32,64,32], [64,32,32]** in the *unexecuted
  proposal*, not in historical evidence. They are deferred, not proven losers.
- Retain all declared small controls, compact winners, width-64 narrowing
  placements, flat three-layer widths through 128, and the five-layer width-64
  control. Revised depth counts are **3/6/18/4/1** for depths 1-5.
- Keep actual final-input validation and prediction-distribution warnings.
  Retain the already-validated selected-FPS top-one recipe as an eligible
  comparison recipe: sequential LR/WD checks must not discard a better tested
  combination. Reuse requires equivalent architecture and matching input,
  batch, seed, split, budget and verified provenance; no extra trainings.
- Preserve raw/per-class reporting, all adverse results, automatic Run All and
  the test lock. No new macro averages have been calculated for these tables.

Verification for this review: 168 parsed rows, 164 unique keys, four consistent
duplicate pairs, integer accuracy counts, 32 unique proposed stacks and unchanged
budget arithmetic. This is planning/report verification, not experiment-code
verification. No new implementation, real training, commit or push.
