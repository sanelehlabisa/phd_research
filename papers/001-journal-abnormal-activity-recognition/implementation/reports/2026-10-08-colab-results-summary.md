# AAD Colab experiment results - 8 October 2026

## Main conclusion

- **Best custom architecture in the matched architecture search:** three ConvLSTM
  layers **[16, 16, 32]**, 3x3 kernels, **85,291 trainable parameters**.
  Its reference-setting validation accuracy was **24.38% +/- 1.77 percentage points**
  across seeds 42 and 2026.
- **Best tested custom configuration by mean validation accuracy:** that architecture
  with **8 frames at 48x48**, weight decay **0.0001**:
  **26.25% +/- 0.88 pp**. Seed scores were 25.625% and 26.875%.
- **Best model actually run in the model-family comparison:** the published
  PaperConvLSTM, **84.38% +/- 2.65 pp**. R3D-18 achieved **68.44%** and MC3-18 **64.69%**.
- **Important comparison error:** the comparison's `custom_selected` was actually
  **[16, 16, 16]**, not the search winner **[16, 16, 32]**. Therefore, the downloaded
  comparison does **not** establish the best custom model's performance against the
  other families under matched settings.
- All numbers below are **validation results**, not final test accuracy. The 161-clip
  test partition is recorded as locked in every one of the 32 runs.
- This report summarises recorded artifacts; it does not change code, checkpoints,
  configs, splits, tickets or manuscript claims. Generated artifacts remain local;
  links into `runs/` and `configs/experiments/temp/` require the downloaded files.

## Evidence and audit

The recently downloaded batch is identified by
[its progress manifest](../runs/notebook_studies/20261008_122455_053379/progress.json):
**20 custom-search jobs + 12 comparison jobs = 32 completed runs**.

- Recorded execution: 8 October 2026, **14:25:05 to 21:21:23 SAST**
  (12:25:05 to 19:21:23 UTC).
- Source revision in every run: `59bc974e6027e109ba723567f9715f51abbb7e00`.
- All 32 leaf `run.json` records say complete. Their summaries agree with their
  per-model metrics files and checkpoint metadata.
- All 32 checkpoint files are present locally. Their binary contents were not
  reloaded or rerun for this report; saved metadata/history/confusion were audited.
- Recomputed accuracy from every saved confusion matrix agrees with the reported
  accuracy. Each selected epoch is the minimum-validation-loss epoch in its history.
- All runs share split hash
  `aada503d63d4433b272237db6cf27d7b1c0eaeb0091d5ba8289343a6111454d8`.
- The local [split manifest](../splits/abnormal-activities-dataset_seed42.json)
  matches that hash after normalising Windows CRLF line endings to LF. Its raw
  Windows-file hash differs because of line endings, not different assignments.
  A strict byte-hash loader may still need the original LF artifact.
- The downloaded [search config](../configs/experiments/temp/aad_custom_search_colab.json)
  and [comparison config](../configs/experiments/temp/aad_model_comparison_colab.json)
  match the current profiles as parsed JSON.
- The downloaded copy contains leaf results and the notebook progress file, but
  **no `runs/studies/` group summaries, selections or selected configuration file**.
  Rankings here were reconstructed from leaf results. The archive helper collects
  `runs/experiments/`, configs and notebook progress, but not `runs/studies/`;
  see [the archive code](../notebooks/utils/aad_study.py).
- Historical September runs and separate Kinetics/VDD diagnostics are excluded.
- Summed recorded training time: **6.76 hours**. Summed leaf-run wall time:
  **6.93 hours**. These are not inference-latency measurements.

## Protocol and how to read the metrics

| Setting | Custom search / ablations | Model-family comparison |
|---|---|---|
| Dataset | AAD, same 11 classes | Same AAD inventory and split |
| Split | 748 train / 160 validation / 161 locked test | Same |
| Seeds | 42 and 2026 | 42 and 2026 |
| Reference input | 8 RGB frames at 32x32 | 50 RGB frames at 50x50 |
| Batch size | 16 | 1 |
| Optimizer / initial LR | Adam / 0.001 | Adam / 0.001 |
| Maximum epochs | 16 | 16 |
| Early-stopping patience | 4 non-improving validation-loss epochs | 4 |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| Reference weight decay | 0.0001 | 0.0001 |
| Augmentation | Off | Off |
| Checkpoint selection | Lowest validation loss | Lowest validation loss |
| Initialization | Scratch / no pretrained weights | Scratch / no pretrained weights |

- **Accuracy, precision, recall and F1 are percentages.** Precision/recall/F1 use
  the saved **micro** convention. In this single-label multiclass task they equal
  accuracy; identical columns are expected, not four independent confirmations.
- Aggregate rows are the arithmetic means of the two independently trained seeds.
  Accuracy SD is their **sample standard deviation**, in percentage points, not a
  confidence interval. With only two seeds, uncertainty remains poorly estimated.
- Loss is sample-weighted cross-entropy, not a percentage; lower is better.
- Each run's metrics use its restored minimum-loss checkpoint, not its highest
  accuracy epoch or last epoch. Appendix A separately reports the peak accuracy.
- Ranking: mean validation accuracy, then lower mean loss, then fewer parameters.
  Never use the locked test set to choose a model.
- Timing includes each run's actual number of training epochs. Early-stopped runs
  may look faster because they ran fewer epochs; do not infer deployment speed.
- The split is clip-level. Source-video/person independence and absence of near
  duplicates have not been established by this results audit.

## 1. Custom architecture search - all six candidates

All rows use **8 frames, 32x32, batch 16, weight decay 0.0001**.
All custom kernels are 3x3, classifier dropout is 0.5, and there is no hidden
dense layer between global average pooling and the classifier.

| Model / setting | Accuracy % | Precision % | Recall % | F1 % | Accuracy SD (pp) | Mean loss | Parameters | Mean training min/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom [16,16,32] | 24.37 | 24.37 | 24.37 | 24.37 | 1.77 | 1.9842 | 85,291 | 11.20 |
| Custom [16,16] | 24.06 | 24.06 | 24.06 | 24.06 | 1.33 | 2.1104 | 29,691 | 10.56 |
| Custom [16,32,16] | 22.19 | 22.19 | 22.19 | 22.19 | 0.44 | 2.0928 | 94,331 | 11.09 |
| Custom [16,16,16] | 17.81 | 17.81 | 17.81 | 17.81 | 7.51 | 2.2845 | 48,187 | 6.96 |
| Custom [32,16,16] | 17.19 | 17.19 | 17.19 | 17.19 | 6.63 | 2.2465 | 86,843 | 8.22 |
| Custom [16,16,16,16] | 17.19 | 17.19 | 17.19 | 17.19 | 6.63 | 2.2877 | 66,683 | 10.96 |

**Interpretation:** [16,16,32] wins numerically, but [16,16] is only
**0.3125 pp behind** while using **65.2% fewer parameters** (29,691 versus 85,291).
Across two 160-video validation evaluations, that mean gap corresponds to just
**one additional correct prediction in total**. This is not strong evidence that
the larger custom stack is generally better.

Widening the middle layer did not beat either of those candidates. Adding a fourth
16-filter layer, or widening the first layer, was less successful in this budget.
More layers/parameters are therefore not automatically the answer.

## 2. One-factor checks on [16,16,32] - every tested setting

Only the named factor changes from the reference. These are separate alternatives;
there is **no evidence here for a combined larger-resolution + longer-clip + different-WD setting**.

| Model / setting | Accuracy % | Precision % | Recall % | F1 % | Accuracy SD (pp) | Mean loss | Parameters | Mean training min/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reference: 8 frames, 32x32, WD 0.0001 | 24.37 | 24.37 | 24.37 | 24.37 | 1.77 | 1.9842 | 85,291 | 11.20 |
| Only resolution -> 48x48 | 26.25 | 26.25 | 26.25 | 26.25 | 0.88 | 2.0986 | 85,291 | 11.12 |
| Only frame count -> 16 | 25.94 | 25.94 | 25.94 | 25.94 | 3.09 | 1.9549 | 85,291 | 10.90 |
| Only weight decay -> 0 | 25.31 | 25.31 | 25.31 | 25.31 | 0.44 | 1.9500 | 85,291 | 10.26 |
| Only weight decay -> 0.001 | 21.25 | 21.25 | 21.25 | 21.25 | 2.65 | 2.1413 | 85,291 | 10.23 |

- **48x48:** +1.875 pp versus the reference; highest mean custom accuracy.
  Its mean loss is worse, however (2.0986 versus 1.9842), so the accuracy gain does
  not mean every aspect of prediction quality improved.
- **16 frames:** +1.5625 pp; lower loss, but greater seed variation. The **28.125%**
  seed-2026 result is the best individual selected custom checkpoint, not the
  best two-seed average.
- **No weight decay:** +0.9375 pp, with lower loss; a modest improvement, not a cure.
- **Weight decay 0.001:** -3.125 pp; the strongest tested regularisation was worse.
- 48x48 beats the 16-frame alternative by only **0.3125 pp** in mean accuracy.
  Two seeds on 160 validation clips do not establish a robust ordering.
- None of these changes lifts the custom model out of the low-20s accuracy range.

## 3. Model-family comparison - all six models

All rows here use **50 frames, 50x50, batch 1, weight decay 0.0001**.
This input/batch protocol differs from the architecture-search protocol.

| Model / setting | Accuracy % | Precision % | Recall % | F1 % | Accuracy SD (pp) | Mean loss | Parameters | Mean training min/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Published PaperConvLSTM | 84.38 | 84.38 | 84.38 | 84.38 | 2.65 | 40.2311 | 512,197,467 | 20.21 |
| R3D-18 | 68.44 | 68.44 | 68.44 | 68.44 | 0.44 | 1.0042 | 33,171,915 | 12.91 |
| MC3-18 | 64.69 | 64.69 | 64.69 | 64.69 | 1.33 | 1.0163 | 11,495,883 | 11.29 |
| Custom [16,16,16] (`custom_selected`) | 12.50 | 12.50 | 12.50 | 12.50 | 0.00 | 2.3839 | 48,187 | 15.56 |
| Swin3D-S | 12.50 | 12.50 | 12.50 | 12.50 | 0.00 | 2.3858 | 49,517,537 | 27.05 |
| Swin3D-T | 11.56 | 11.56 | 11.56 | 11.56 | 1.33 | 2.3867 | 27,858,929 | 14.38 |

**Do not interpret the 12.50% custom row as the search winner's result.**
Its checkpoint metadata explicitly records three 16-filter layers and 48,187
parameters; the winning search model has a 32-filter final layer and 85,291
parameters. Evidence:
[comparison checkpoint metadata](../runs/experiments/20261008_155316_510990_abnormal-activities-dataset_screen/models/custom_selected/checkpoints/best_model.json).

The notebook runner invokes the comparison profile after the search, but does
not transfer the selected layer list into that profile. Its static
`custom_selected` entry remained [16,16,16]. The [runner](../notebooks/utils/aad_study.py)
and downloaded comparison JSON agree with the checkpoint evidence.

The best custom search configuration's **26.25%** and the published model's
**84.38%** are useful separate observations, **not a controlled head-to-head**:
architecture, temporal/spatial inputs and batch size differ.

### Parameter trade-offs

- Search winner: **85,291** parameters; smaller [16,16] alternative: **29,691**.
- R3D-18: **33,171,915**, about **389x** the search winner.
- MC3-18: **11,495,883**, about **135x** the search winner.
- Published PaperConvLSTM: **512,197,467**, about **6,005x** the search winner.
- Swin3D-T / Swin3D-S: **27,858,929 / 49,517,537**.
- The custom networks are genuinely lightweight, but the present accuracy is not
  competitive. The published topology is not a lightweight deployment candidate.
  Parameter count alone does not determine memory use or latency.

## 4. What the curves and confusion matrices mean

### The small custom models are mostly under-learning, not showing classic overfitting

At the end of the 48x48 custom runs, training accuracy was only **21.26% / 22.59%**;
selected validation accuracy was **25.63% / 26.88%**. The reference winner ended
at **23.26% / 23.93%** training accuracy. This is not a high-training/low-validation
pattern. Training metrics are collected during training with dropout enabled,
so their exact gap to evaluation-mode validation is not a clean generalisation-gap estimate.

The 48x48 model correctly recognises mainly Begging, Normal Videos and Pollution.
It has zero recall for seven of the eleven classes. Therefore a result near 26%
does not mean it has learned all activities moderately well.

### Several runs collapsed to predicting one class

- The comparison's custom [16,16,16] predicts **Begging for all 160 validation clips**
  in both seeds: 20 correct, hence **12.5%**.
- Swin3D-S does the same in both seeds.
- Swin3D-T predicts Begging throughout for seed 42 and Normal Videos throughout for
  seed 2026: **12.5% / 10.625%**.
- The largest validation class has 20/160 clips, so an always-Begging classifier
  already achieves **12.5%**. Uniform random guessing over 11 classes has expected
  accuracy of approximately **9.09%**.
- These are failed learnability outcomes for those particular configurations,
  not evidence that transformers or ConvLSTM generally cannot work.

### Rapid learning-rate decay did not cause these results

**Every saved epoch in every run used learning rate 0.001.** The scheduler never
reduced it during this batch. The code's scheduler patience is 5, whereas early
stopping waits only 4 non-improving epochs, so stopping can occur before a
plateau-triggered LR reduction has a chance to help.

**18 of the 20 custom-search jobs reached the 16-epoch cap; 11 selected epoch 16.**
The short run budget often ended while the best validation loss was still at the
end. This supports testing a better-aligned training budget/patience, but does not
promise that longer training alone will solve the problem.

### The dataset contains learnable signal, but the paper model is unstable

The published ConvLSTM and both CNNs learn substantially on the same dataset/split.
That argues against the dataset being completely unlearnable or the whole pipeline
having random labels. It does **not** prove data quality, split independence or
absence of model-specific issues.

The published topology has the highest accuracy but concerning losses:

- Seed 42: selected epoch 5, accuracy **82.5%**, validation loss **76.2879**.
- At epoch 6 its validation loss jumped to **37,582.69**, accuracy **11.25%**.
  The selected checkpoint was restored; the final row is not that collapsed epoch.
- Seed 2026: selected epoch 14, accuracy **86.25%**, loss **4.1742**; this run also
  had a large earlier loss spike.
- Consequently mean published-model loss is **40.2311**, despite high accuracy.
  Cross-entropy penalises very confident mistakes heavily. This is consistent with
  unstable or severely miscalibrated predictions; the precise cause is not
  established by these artifacts. Do not describe it as a stable solution yet.

R3D-18 and MC3-18 have much more moderate validation losses (~1.00), while their
training accuracy ends around **85-88%** and **81-90%**, respectively. Their
validation accuracies are lower (about 68% and 65%), which suggests a meaningful
train/validation gap rather than the custom models' near-failure to fit.

## 5. Class-level diagnostics

The next table is **mean validation recall per class (%) across the two seeds**.
The custom column is the best-mean-accuracy 8-frame/48x48 setting; comparison models
use 50-frame/50x50 inputs. Use this table to identify failures, not as a matched
cross-protocol ranking.

| Class | Custom [16,16,32], 48x48 | Published ConvLSTM | R3D-18 | MC3-18 |
|---|---:|---:|---:|---:|
| Begging | 95.00 | 100.00 | 95.00 | 80.00 |
| Drunkenness | 0.00 | 92.31 | 84.62 | 69.23 |
| Fight | 0.00 | 60.71 | 32.14 | 85.71 |
| Harassment | 0.00 | 85.71 | 64.29 | 71.43 |
| Hijack | 4.17 | 91.67 | 62.50 | 62.50 |
| Knife Hazard | 0.00 | 66.67 | 56.67 | 23.33 |
| Normal Videos | 76.47 | 85.29 | 44.12 | 52.94 |
| Pollution | 55.88 | 85.29 | 73.53 | 44.12 |
| Property Damage | 0.00 | 96.43 | 89.29 | 71.43 |
| Robbery | 0.00 | 61.54 | 76.92 | 100.00 |
| Terrorism | 0.00 | 100.00 | 68.18 | 54.55 |

As an additional diagnostic, macro-F1 recomputed from the saved confusion matrices
(equal weight per class, not the official micro metric) averages **11.81%** for the
48x48 custom model, **84.09%** for the published model, **67.05%** for R3D-18 and
**62.27%** for MC3-18. This makes the custom model's poor coverage of most classes
more visible. These diagnostic macro values do not replace the saved metric protocol.

## 6. Findings and next decisions

1. **Keep [16,16,32] and [16,16] as provisional custom candidates.** The latter is
   much smaller for essentially tied reference accuracy. Do not declare the
   architecture search scientifically settled from two short seeds.
2. **Repair the search-to-comparison handoff before a final comparison.** The
   actual search winner has not yet been evaluated under the matched 50-frame,
   50x50 comparison protocol. Correct the selected architecture explicitly and
   preserve the current runs as evidence of what was actually executed.
3. **Investigate custom-model learnability before scaling everything.** A small,
   balanced training-only overfit check can separate failure to optimise from
   failure to generalise. The pooled custom head has only 16 or 32 features,
   dropout 0.5 and no hidden dense layer; these are plausible factors to isolate,
   not proven causes. No state-carrying across unrelated videos is justified.
4. **Align the training budget, scheduler and stopping patience.** LR did not decay;
   many custom runs hit their epoch cap. Test changes with validation only, one
   factor at a time. Do not silently combine all favourable settings.
5. **Investigate the published model's extreme loss spikes.** Its high validation
   accuracy is real in the saved matrices, but stable optimisation/calibration and
   computational practicality remain unresolved.
6. **Preserve/retrieve group-level study artifacts before finalisation.** Archive
   the study summaries/selections and exact split manifest alongside the leaf runs.
   The current local group-level evidence is incomplete even though all leaf jobs
   finished and can be summarised.
7. **Keep test locked until selection is frozen.** These are preliminary validation
   findings, not final generalisation evidence or a completed manuscript result.

These are suggested next decisions, not implemented changes. This report does not
claim a final trained model, final test result, statistical significance, or a
dataset fault.

## Appendix A. Every individual run

Each row links to the exact saved summary. Accuracy/precision/recall/F1 are the
**restored checkpoint's validation metrics (%)**. Selected/run epochs distinguish
checkpoint selection from training duration. Peak accuracy is descriptive only;
it is not substituted for the loss-selected checkpoint. Train-last accuracy is
the last epoch's training-mode metric.

### A1. Custom architecture search and one-factor trials

| Run | Model / variant | Seed | Acc % | Prec % | Rec % | F1 % | Loss | Params | Selected / run epochs | Peak val acc % | Train-last acc % | Train min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [20261008_122505_956936](../runs/experiments/20261008_122505_956936_abnormal-activities-dataset_screen/summary.json) | Custom [16,16,16] | 42 | 12.500 | 12.500 | 12.500 | 12.500 | 2.3884 | 48,187 | 1 / 5 | 12.500 | 12.30 | 3.28 |
| [20261008_122838_332677](../runs/experiments/20261008_122838_332677_abnormal-activities-dataset_screen/summary.json) | Custom [16,16,16] | 2026 | 23.125 | 23.125 | 23.125 | 23.125 | 2.1806 | 48,187 | 16 / 16 | 23.125 | 19.79 | 10.65 |
| [20261008_123931_821983](../runs/experiments/20261008_123931_821983_abnormal-activities-dataset_screen/summary.json) | Custom [16,16] | 42 | 23.125 | 23.125 | 23.125 | 23.125 | 2.1051 | 29,691 | 16 / 16 | 23.125 | 20.86 | 10.63 |
| [20261008_125024_414063](../runs/experiments/20261008_125024_414063_abnormal-activities-dataset_screen/summary.json) | Custom [16,16] | 2026 | 25.000 | 25.000 | 25.000 | 25.000 | 2.1158 | 29,691 | 16 / 16 | 25.000 | 20.45 | 10.50 |
| [20261008_130108_521245](../runs/experiments/20261008_130108_521245_abnormal-activities-dataset_screen/summary.json) | Custom [16,16,16,16] | 42 | 12.500 | 12.500 | 12.500 | 12.500 | 2.3850 | 66,683 | 14 / 16 | 12.500 | 12.43 | 10.99 |
| [20261008_131223_230115](../runs/experiments/20261008_131223_230115_abnormal-activities-dataset_screen/summary.json) | Custom [16,16,16,16] | 2026 | 21.875 | 21.875 | 21.875 | 21.875 | 2.1904 | 66,683 | 16 / 16 | 21.875 | 19.25 | 10.92 |
| [20261008_132333_360164](../runs/experiments/20261008_132333_360164_abnormal-activities-dataset_screen/summary.json) | Custom [32,16,16] | 42 | 21.875 | 21.875 | 21.875 | 21.875 | 2.1035 | 86,843 | 16 / 16 | 21.875 | 20.45 | 10.94 |
| [20261008_133445_476189](../runs/experiments/20261008_133445_476189_abnormal-activities-dataset_screen/summary.json) | Custom [32,16,16] | 2026 | 12.500 | 12.500 | 12.500 | 12.500 | 2.3895 | 86,843 | 4 / 8 | 12.500 | 10.29 | 5.51 |
| [20261008_134032_043456](../runs/experiments/20261008_134032_043456_abnormal-activities-dataset_screen/summary.json) | Custom [16,32,16] | 42 | 22.500 | 22.500 | 22.500 | 22.500 | 2.0215 | 94,331 | 16 / 16 | 22.500 | 18.85 | 10.97 |
| [20261008_135145_203056](../runs/experiments/20261008_135145_203056_abnormal-activities-dataset_screen/summary.json) | Custom [16,32,16] | 2026 | 21.875 | 21.875 | 21.875 | 21.875 | 2.1642 | 94,331 | 16 / 16 | 21.875 | 18.85 | 11.21 |
| [20261008_140312_748157](../runs/experiments/20261008_140312_748157_abnormal-activities-dataset_screen/summary.json) | Custom [16,16,32] | 42 | 23.125 | 23.125 | 23.125 | 23.125 | 1.9849 | 85,291 | 15 / 16 | 26.250 | 23.26 | 11.16 |
| [20261008_141437_282446](../runs/experiments/20261008_141437_282446_abnormal-activities-dataset_screen/summary.json) | Custom [16,16,32] | 2026 | 25.625 | 25.625 | 25.625 | 25.625 | 1.9835 | 85,291 | 16 / 16 | 25.625 | 23.93 | 11.23 |
| [20261008_142606_337025](../runs/experiments/20261008_142606_337025_abnormal-activities-dataset_ablation/summary.json) | Custom [16,16,32]; 8f/48px/WD0.0001 | 42 | 25.625 | 25.625 | 25.625 | 25.625 | 2.1334 | 85,291 | 15 / 16 | 25.625 | 21.26 | 11.20 |
| [20261008_143733_332002](../runs/experiments/20261008_143733_332002_abnormal-activities-dataset_ablation/summary.json) | Custom [16,16,32]; 8f/48px/WD0.0001 | 2026 | 26.875 | 26.875 | 26.875 | 26.875 | 2.0638 | 85,291 | 16 / 16 | 26.875 | 22.59 | 11.04 |
| [20261008_144850_783664](../runs/experiments/20261008_144850_783664_abnormal-activities-dataset_ablation/summary.json) | Custom [16,16,32]; 16f/32px/WD0.0001 | 42 | 23.750 | 23.750 | 23.750 | 23.750 | 1.9693 | 85,291 | 15 / 16 | 25.000 | 20.86 | 10.85 |
| [20261008_145956_451719](../runs/experiments/20261008_145956_451719_abnormal-activities-dataset_ablation/summary.json) | Custom [16,16,32]; 16f/32px/WD0.0001 | 2026 | 28.125 | 28.125 | 28.125 | 28.125 | 1.9405 | 85,291 | 14 / 16 | 28.125 | 27.41 | 10.96 |
| [20261008_151109_613152](../runs/experiments/20261008_151109_613152_abnormal-activities-dataset_ablation/summary.json) | Custom [16,16,32]; 8f/32px/WD0 | 42 | 25.000 | 25.000 | 25.000 | 25.000 | 1.9665 | 85,291 | 15 / 16 | 26.875 | 23.80 | 10.32 |
| [20261008_152143_239151](../runs/experiments/20261008_152143_239151_abnormal-activities-dataset_ablation/summary.json) | Custom [16,16,32]; 8f/32px/WD0 | 2026 | 25.625 | 25.625 | 25.625 | 25.625 | 1.9334 | 85,291 | 16 / 16 | 26.875 | 23.93 | 10.19 |
| [20261008_153208_429592](../runs/experiments/20261008_153208_429592_abnormal-activities-dataset_ablation/summary.json) | Custom [16,16,32]; 8f/32px/WD0.001 | 42 | 19.375 | 19.375 | 19.375 | 19.375 | 2.1868 | 85,291 | 15 / 16 | 22.500 | 20.19 | 10.18 |
| [20261008_154232_725058](../runs/experiments/20261008_154232_725058_abnormal-activities-dataset_ablation/summary.json) | Custom [16,16,32]; 8f/32px/WD0.001 | 2026 | 23.125 | 23.125 | 23.125 | 23.125 | 2.0958 | 85,291 | 16 / 16 | 23.125 | 22.19 | 10.28 |

### A2. Model-family comparison

| Run | Model / variant | Seed | Acc % | Prec % | Rec % | F1 % | Loss | Params | Selected / run epochs | Peak val acc % | Train-last acc % | Train min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [20261008_155316_510990](../runs/experiments/20261008_155316_510990_abnormal-activities-dataset_screen/summary.json) | Custom [16,16,16] (`custom_selected`) | 42 | 12.500 | 12.500 | 12.500 | 12.500 | 2.3831 | 48,187 | 7 / 11 | 12.500 | 12.30 | 18.02 |
| [20261008_161139_720065](../runs/experiments/20261008_161139_720065_abnormal-activities-dataset_screen/summary.json) | Custom [16,16,16] (`custom_selected`) | 2026 | 12.500 | 12.500 | 12.500 | 12.500 | 2.3847 | 48,187 | 4 / 8 | 12.500 | 11.10 | 13.11 |
| [20261008_162507_679637](../runs/experiments/20261008_162507_679637_abnormal-activities-dataset_screen/summary.json) | Published PaperConvLSTM | 42 | 82.500 | 82.500 | 82.500 | 82.500 | 76.2879 | 512,197,467 | 5 / 9 | 82.500 | 54.41 | 14.34 |
| [20261008_164018_299409](../runs/experiments/20261008_164018_299409_abnormal-activities-dataset_screen/summary.json) | Published PaperConvLSTM | 2026 | 86.250 | 86.250 | 86.250 | 86.250 | 4.1742 | 512,197,467 | 14 / 16 | 86.250 | 83.42 | 26.08 |
| [20261008_170711_912018](../runs/experiments/20261008_170711_912018_abnormal-activities-dataset_screen/summary.json) | R3D-18 | 42 | 68.125 | 68.125 | 68.125 | 68.125 | 1.0347 | 33,171,915 | 9 / 13 | 68.750 | 88.24 | 12.37 |
| [20261008_171954_635535](../runs/experiments/20261008_171954_635535_abnormal-activities-dataset_screen/summary.json) | R3D-18 | 2026 | 68.750 | 68.750 | 68.750 | 68.750 | 0.9737 | 33,171,915 | 11 / 15 | 68.750 | 85.03 | 13.45 |
| [20261008_173342_668715](../runs/experiments/20261008_173342_668715_abnormal-activities-dataset_screen/summary.json) | MC3-18 | 42 | 63.750 | 63.750 | 63.750 | 63.750 | 0.9557 | 11,495,883 | 7 / 11 | 63.750 | 81.42 | 9.53 |
| [20261008_174332_697729](../runs/experiments/20261008_174332_697729_abnormal-activities-dataset_screen/summary.json) | MC3-18 | 2026 | 65.625 | 65.625 | 65.625 | 65.625 | 1.0769 | 11,495,883 | 11 / 15 | 66.875 | 89.71 | 13.04 |
| [20261008_175654_014815](../runs/experiments/20261008_175654_014815_abnormal-activities-dataset_screen/summary.json) | Swin3D-T | 42 | 12.500 | 12.500 | 12.500 | 12.500 | 2.3854 | 27,858,929 | 10 / 14 | 12.500 | 12.70 | 16.76 |
| [20261008_181402_048831](../runs/experiments/20261008_181402_048831_abnormal-activities-dataset_screen/summary.json) | Swin3D-T | 2026 | 10.625 | 10.625 | 10.625 | 10.625 | 2.3881 | 27,858,929 | 6 / 10 | 12.500 | 8.96 | 12.01 |
| [20261008_182626_788221](../runs/experiments/20261008_182626_788221_abnormal-activities-dataset_screen/summary.json) | Swin3D-S | 42 | 12.500 | 12.500 | 12.500 | 12.500 | 2.3858 | 49,517,537 | 9 / 13 | 12.500 | 10.70 | 26.00 |
| [20261008_185252_002929](../runs/experiments/20261008_185252_002929_abnormal-activities-dataset_screen/summary.json) | Swin3D-S | 2026 | 12.500 | 12.500 | 12.500 | 12.500 | 2.3858 | 49,517,537 | 10 / 14 | 12.500 | 10.70 | 28.11 |

## Appendix B. Reproducibility and limitations

- Calculation: for each group, mean = (seed-42 value + seed-2026 value) / 2;
  sample SD uses denominator n-1 = 1. Metrics shown rounded; all calculations used
  the saved values before display rounding.
- Validation accuracy was independently checked as confusion-matrix diagonal sum
  divided by total observations (160). Micro precision/recall/F1 were also checked
  against that result.
- Per-class recall = true positives / actual class count. Diagnostic macro-F1 is
  computed per run as the mean over classes of 2TP/(actual count + predicted count),
  then averaged across seeds.
- The same fixed validation clips appear in both seeds: that is not 320 independent
  validation examples. Many models/settings reuse this validation split, so winner
  selection is optimistically biased relative to a genuinely untouched final test.
- No new training, model inference, dataset decoding or test evaluation was run.
  Checkpoint presence/metadata was checked, not binary weight integrity or prediction
  reproducibility from the weights.
- The configurations and histories support comparisons within each stage, but the
  short common training recipe may suit different architectures unequally. These
  results cannot establish each model family's best achievable performance.
- Use the run links above to inspect configurations, histories, confusion matrices
  and checkpoint provenance. Local artifacts still contain original Colab absolute
  paths; this report links to the corresponding downloaded relative locations.

**Copyable summary for discussion:** The two-seed AAD custom search provisionally
favours [16,16,32] (85,291 parameters), with 24.38% reference validation accuracy
and 26.25% when only resolution increases to 48x48. [16,16] is almost tied at
24.06% using 29,691 parameters. The separate model-family comparison achieved
84.38% with the published ConvLSTM, 68.44% with R3D-18 and 64.69% with MC3-18.
However, its custom entry incorrectly remained [16,16,16] and collapsed to 12.5%;
the search winner has not received a matched comparison. Swin models also collapsed.
The custom models mostly fail to fit well rather than simply overfit; all runs kept
LR at 0.001, and most custom jobs reached the 16-epoch cap. The published model has
serious loss spikes despite high accuracy. All results are validation-only, on
1,069 clips with a 748/160/161 split; test remains locked.
