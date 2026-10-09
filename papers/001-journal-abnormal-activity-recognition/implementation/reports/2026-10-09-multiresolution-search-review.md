# AAD search results and next experiment decisions

Follow-up on 9 October 2026: the repository owner checked the similarly named
videos and confirmed they are different recordings, not segments of one recording.
The filename-only concern below is therefore resolved by user attestation, not
independent source/subject verification. Exact duplicate and contrary source-ID
checks remain active. [Ticket 073](../../../../agents/work/073-focused-shape-search-run-all/prompt.md)
consolidates and supersedes the initial 070–072 recommendations below. Original
results are unchanged; the imported bundle was moved intact to ignored
`runs/imports/20261009-aad-search-2905fca/`.

The 9 October 2026 archive contains a completed **36-run validation search**:
12 custom ConvLSTM architectures at three resolutions. The strongest architecture
across these settings is **[16,24]**, averaging **91.46% validation accuracy**
with **45,939 parameters**. This is encouraging evidence that compact models can
learn this clip-level task, not a final test result or proof of an optimal architecture.

A source-independence audit is the priority before publication claims: filenames
suggest that segments of the same original recordings may cross train, validation
and test splits. This is a warning from names, not confirmed leakage. The current
split and all saved results remain unchanged.

## What completed

| Stage | Evidence |
|---|---|
| Custom search | All 36 jobs complete |
| Validation | Performed during every training run; selected checkpoint evaluated on all 160 validation clips |
| Architecture selection | Complete; one numerical tie needs correction before a new shortlist is frozen |
| Eight-model comparison | Not run; an archived configuration is only a template |
| Final test and test examples | Not run; saved test state is locked |
| Artifact export | 738 inventory entries verified by size and SHA256; 58,335,113 bytes |

The notebook ran the search-only branch. Its helper explicitly ends with
“Custom search complete; model-family comparison is a separate run.”
It did not fail between search and comparison: automatic handoff was not implemented.
Validation is already part of search, not a missing separate training phase.

The approved follow-up will run search, validation selection, eight-model comparison,
checkpoint freeze, one-time test evaluation, examples and ZIP export in sequence.
It must use returned stage paths, not ask the user to paste directories or toggle stages.
Invalid or incomplete evidence must still stop dependent stages.

Sources: [saved study summary](<../runs/imports/20261009-aad-search-2905fca/runs/studies/20261009_161429_214290_abnormal-activities-dataset_multiresolution/summary.json>),
[stage progress](<../runs/imports/20261009-aad-search-2905fca/runs/studies/20261009_161429_214290_abnormal-activities-dataset_multiresolution/progress.json>),
and [notebook helper](../notebooks/utils/aad_study.py).

## Protocol and actual runtime

| Setting | Saved value |
|---|---|
| Code and study | Revision `2905fca`; ticket 068, not the new ticket 069 |
| Data | AAD, 11 classes, 1,069 clips |
| Fixed split | 748 train / 160 validation / 161 test; clip-level stratification |
| Seeds | Split 42; model 42 only |
| Inputs | 8 frames; 32x32, 48x48, 64x64 |
| Sampling | Legacy rounded-stride sampler, nominal target 16 FPS |
| Optimizer | Adam, initial LR 0.01, weight decay 0 |
| Training | Batch 32, no augmentation, maximum 128 epochs, early-stop patience 16 |
| Checkpoint selection | Lowest validation loss, not highest epoch accuracy |
| Architecture ranking | Equal-weight mean validation accuracy over resolutions, then mean loss, parameter count and name |
| Full search wall time | **72 minutes 42 seconds**, 16:14:29–17:27:11 UTC |
| Training time summed over jobs | 68 minutes 14 seconds |
| Typical job | Mean wall time 2.02 minutes |
| Cache | 33 reuses; three initial builds, one per resolution |

The 32-pixel block took 18.18 minutes, the 48-pixel block 24.13 minutes and the
64-pixel block 30.38 minutes. The approximately 20-minute observation may refer
to the first block; it does not describe the complete archived search.

27 of 36 jobs reached the 128-epoch cap; completed lengths range from 23 to 128
epochs. Do not extrapolate these two-minute jobs directly to wider models, reference
families, 200 epochs or the 50-frame comparison.

The old sampler uses a rounded source-FPS stride. The nominal 16 FPS is not
evidence of exact 16-FPS sampling for every source. This archive does not establish
source FPS or full-video temporal coverage.

Sources: [resolved study](<../runs/imports/20261009-aad-search-2905fca/runs/studies/20261009_161429_214290_abnormal-activities-dataset_multiresolution/study.json>),
[split manifest](<../runs/imports/20261009-aad-search-2905fca/runs/studies/20261009_161429_214290_abnormal-activities-dataset_multiresolution/split_manifest.json>),
and [job receipts](<../runs/imports/20261009-aad-search-2905fca/runs/studies/20261009_161429_214290_abnormal-activities-dataset_multiresolution/jobs/>).

## Top five and complete architecture ranking

The first five rows are the recommended interpretation of the saved search under
its declared tie-break. Brackets list recurrent-layer widths; all kernels are 3x3.
Accuracy is a percentage. Each resolution has its own trained checkpoint.

| Rank by exact accuracy and tie-break | Layers | Saved rank | 32 px | 48 px | 64 px | Mean accuracy | Parameters |
|---|---|---|---|---|---|---|---|
| 1 | `[16,24]` | 1 | 92.50 | 92.50 | 89.38 | 91.46 | 45,939 |
| 2 | `[16,32,16]` | 2 | 95.63 | 84.38 | 87.50 | 89.17 | 94,331 |
| 3 | `[32,16]` | 4 | 95.00 | 92.50 | 77.50 | 88.33 | 68,347 |
| 4 | `[16,16,16]` | 3 | 91.25 | 88.75 | 85.00 | 88.33 | 48,187 |
| 5 | `[24,24,24]` | 5 | 95.00 | 92.50 | 75.63 | 87.71 | 106,835 |
| 6 | `[16,16]` | 6 | 90.00 | 77.50 | 75.00 | 80.83 | 29,691 |
| 7 | `[32,16,16]` | 7 | 93.75 | 90.00 | 51.25 | 78.33 | 86,843 |
| 8 | `[16,16,32]` | 8 | 93.13 | 96.88 | 31.25 | 73.75 | 85,291 |
| 9 | `[24,16]` | 9 | 45.00 | 84.38 | 85.00 | 71.46 | 46,715 |
| 10 | `[8,16]` | 10 | 78.75 | 75.00 | 59.38 | 71.04 | 17,275 |
| 11 | `[32,32,32]` | 11 | 63.75 | 73.13 | 61.88 | 66.25 | 188,523 |
| 12 | `[16,16,16,16]` | 12 | 12.50 | 59.38 | 15.00 | 28.96 | 66,683 |

The ranking has one consequential numerical defect. **[16,16,16] and [32,16] both
produce 424 correct predictions across the three 160-clip validation evaluations**,
so both have exactly 88.3333% mean accuracy. Stored floating-point means differ by
about 0.00000199 percentage points. That tiny difference places [16,16,16] third
before the loss tie-break is reached.

Using exact correct/total counts, [32,16] should be third because its mean validation
loss is lower: **0.6218 versus 0.7607**. Thus:

- Saved top three: [16,24], [16,32,16], [16,16,16].
- Intended tie-break top three: **[16,24], [16,32,16], [32,16]**.

This report does not rewrite the archived selection. Correct the future ranking
implementation with a versioned selection record and regression tests. Do not
solve it by arbitrarily rounding all scores to two decimal places.

The highest individual checkpoint is **[16,16,32] at 48 px: 96.875% (155/160)**.
It falls to 31.25% at 64 px, so it is not the strongest architecture across the
tested resolutions. [16,24] stays between **89.375% and 92.5%**, a spread of only
3.125 percentage points.

Source: [original selection and rankings](<../runs/imports/20261009-aad-search-2905fca/runs/studies/20261009_161429_214290_abnormal-activities-dataset_multiresolution/selected_config.json>).
Accuracy above was recomputed from saved full-validation predictions.

## Precision recall F1 and model cost

Macro metrics give equal weight to each of the 11 classes, then equal weight to
the three resolutions. They were recomputed from the saved confusion matrices.
The older fields named simply precision, recall and F1 are **micro** averages:
for this single-label task they equal accuracy, and must not be labelled macro.
Macro recall here is also balanced accuracy.

| Layers | Macro precision % | Macro recall % | Macro F1 % | Mean loss | Weight only KiB | Mean training seconds |
|---|---|---|---|---|---|---|
| `[16,24]` | 91.61 | 91.54 | 91.08 | 0.4589 | 179.45 | 97.8 |
| `[16,32,16]` | 89.99 | 89.23 | 89.20 | 0.5743 | 368.48 | 134.9 |
| `[32,16]` | 89.36 | 88.25 | 88.11 | 0.6218 | 266.98 | 105.9 |
| `[16,16,16]` | 89.09 | 88.21 | 88.00 | 0.7607 | 188.23 | 129.4 |
| `[24,24,24]` | 87.88 | 87.64 | 86.98 | 0.5301 | 417.32 | 124.0 |
| `[16,16]` | 81.57 | 80.47 | 80.44 | 0.7843 | 115.98 | 100.6 |
| `[32,16,16]` | 75.17 | 77.37 | 75.54 | 0.8546 | 339.23 | 141.8 |
| `[16,16,32]` | 69.02 | 72.21 | 69.35 | 0.9606 | 333.17 | 135.8 |
| `[24,16]` | 74.25 | 70.39 | 68.82 | 0.8973 | 182.48 | 88.4 |
| `[8,16]` | 71.74 | 70.41 | 68.86 | 1.0992 | 67.48 | 87.6 |
| `[32,32,32]` | 66.86 | 65.30 | 63.81 | 1.0777 | 736.42 | 145.8 |
| `[16,16,16,16]` | 22.74 | 26.00 | 20.74 | 1.9492 | 260.48 | 72.9 |

Weight-only size excludes optimizer state and other checkpoint content.
For [16,24], each saved checkpoint is **565,045 bytes**, versus **183,756 bytes**
of weight tensors. Training times include each run's actual stopping behaviour;
they are not inference speed measurements or equal-epoch throughput comparisons.
No per-model inference-latency comparison is established by this archive.

No new CNN, transformer or published-topology comparison results are present.
Do not mix older baseline results with this search and call it a controlled comparison.

## What the learning curves imply

- **Compact models are learning:** [16,24] is strong at all three tested resolutions.
  More layers or parameters are not automatically better under this recipe.
- **Poor large-model scores are not a proven capacity turning point:** [32,32,32]
  has selected-epoch training accuracy of only about 62%, 75% and 54% across
  resolutions. It has not simply fitted training perfectly and overfitted validation.
- **There are optimization failures:** [16,16,16,16] at 32 px selects epoch 21,
  predicts Begging for every validation clip and scores 12.5%. Its 64-pixel run
  stops after only 23 epochs. Do not present these as evidence that four layers
  are inherently unsuitable.
- **Longer training and LR checks are justified, not guaranteed fixes:** 27 jobs
  reached the epoch cap. The final recorded LRs range from approximately 0.00478
  to 0.009, so these logs do not show an extreme LR collapse.
- **Resolution matters:** the mean accuracy over all 12 architectures is 78.85%
  at 32 px, 83.91% at 48 px and 66.15% at 64 px. These averages include optimization
  failures; they do not prove that 48 px is universally best.

For the winning architecture, class recall identifies remaining weaknesses.
At 64 px, Drunkenness, Hijack and Knife Hazard fall behind the other classes.

| Class | Validation clips | Recall at 32 px % | Recall at 48 px % | Recall at 64 px % |
|---|---|---|---|---|
| Begging | 20 | 80.00 | 85.00 | 90.00 |
| Drunkenness | 13 | 84.62 | 84.62 | 61.54 |
| Fight | 14 | 100.00 | 100.00 | 100.00 |
| Harassment | 14 | 92.86 | 92.86 | 92.86 |
| Hijack | 12 | 83.33 | 83.33 | 66.67 |
| Knife Hazard | 15 | 86.67 | 93.33 | 66.67 |
| Normal Videos | 17 | 100.00 | 100.00 | 100.00 |
| Pollution | 17 | 94.12 | 82.35 | 100.00 |
| Property Damage | 14 | 100.00 | 100.00 | 100.00 |
| Robbery | 13 | 100.00 | 100.00 | 100.00 |
| Terrorism | 11 | 100.00 | 100.00 | 100.00 |

There are only 11–20 validation clips per class. The same 160 clips were reused
throughout search; 480 predictions across three models are **not 480 independent
videos**. This run provides no between-seed uncertainty estimate.

## Source independence needs checking

A filename-only screen removes a trailing underscore and number from each stem,
for example `GP012506_10` becomes `GP012506`. The 1,069 filenames collapse to
28 apparent groups; 18 groups span partitions and contain 1,058 clips.

| Apparent source stem | Classes | Train clips | Validation clips | Test clips |
|---|---|---|---|---|
| `GP012506` | Begging | 92 | 19 | 20 |
| `GOPR2529` | Hijack, Property Damage | 86 | 16 | 17 |
| `GOPR2466` | Pollution | 77 | 17 | 17 |
| `GOPR2513` | Knife Hazard | 70 | 15 | 15 |
| `GOPR2517` | Drunkenness | 63 | 13 | 14 |

For example, `Begging/GP012506.mp4` is in test, while
`Begging/GP012506_1.mp4` is in train. Similar names could indicate segments from
one recording, but could also reflect an unrelated naming convention. Verify the
preprocessing rules and actual source identifiers before deciding.

Under this heuristic, Drunkenness, Knife Hazard, Robbery and Terrorism each have
only one apparent source group. If confirmed, a source-disjoint three-way split
covering every class may not be possible with the current data. Do not silently
force a new split or claim independence that the dataset cannot provide.

The archive contains neither raw source videos nor authoritative source IDs.
The proper next step is an audit and a documented protocol decision. Known
cross-split source overlap, or this unresolved suspicion, should block automatic
final testing for publication evidence. Existing results can remain useful
clip-level exploratory evidence.

The manifest matches the earlier October 8 report. This is not an independent
dataset replication, and changes in training settings prevent attributing the
improvement to architecture alone.

## Changes worth making before the next run

Keep ticket 069's staged design rather than expanding the grid again now.
It already addresses shallow models, larger flat models, three resolutions,
LR calibration, two-seed confirmation, weight decay and temporal ablations.

Three bounded follow-ups are proposed, without changing implementation in this review:

1. **070 source audit:** establish source identity and cross-split overlap, report
   per-class group feasibility, and resolve the protocol before final testing.
2. **071 search corrections:** use precision-safe ranking and add the observed
   [16,24] winner as an explicit control at all three resolutions. Ticket 069
   currently omits it. Train the control under the new recipe; do not import old
   scores. This adds at most three jobs, raising the pre-deduplication upper
   budget from 156 to 159.
3. **072 automatic workflow:** remove manual stage/path handoff; automatically
   run the approved sequence, preserve incremental evidence and provide ZIP
   export/download retry without training or testing again.

The approved comparison remains **eight models, 50 frames at 50x50, batch 1,
up to 256 epochs, seed 42**. Its inputs differ from search; the comparison is a
fresh controlled test of the shortlisted architectures, not a guarantee that
their search order persists. Freeze checkpoints and choose the custom model for
examples using validation before reading any test scores.

An optional later temporal-order or single-frame control could test whether the
model is using motion rather than mostly static appearance. It is not included
in these implementation tickets and does not replace the source audit.

To identify a useful capacity turning point, compare converged models at each
depth and resolution under the declared recipe, confirm the best width and its
neighbours with both seeds, and inspect accuracy versus parameter/runtime cost.
If the best width is on a boundary or larger models fail to optimize, report that
limitation rather than manufacturing a peak. Even an interior peak supports
**best tested within this protocol**, not global optimality.

A defensible current conclusion is: “A two-layer [16,24] ConvLSTM achieved the
highest mean validation accuracy across three tested resolutions with 45,939
parameters under the current clip-level split.” Broader generalization and
competitiveness claims need the source audit and completed comparison.

## All 36 selected checkpoints

Values below refer to the minimum-validation-loss checkpoint, not a favourable
epoch chosen afterward. Epochs are selected/completed; time is recorded training
time. Precision and recall for each class remain in the linked raw run artifacts.

| Layers | Pixels | Accuracy % | Loss | Macro F1 % | Epochs | Training seconds |
|---|---|---|---|---|---|---|
| `[16,24]` | 32 | 92.50 | 0.5681 | 92.48 | 124/128 | 81.8 |
| `[16,24]` | 48 | 92.50 | 0.3776 | 92.37 | 112/128 | 92.2 |
| `[16,24]` | 64 | 89.38 | 0.4311 | 88.39 | 128/128 | 119.3 |
| `[16,32,16]` | 32 | 95.63 | 0.2014 | 95.32 | 128/128 | 103.3 |
| `[16,32,16]` | 48 | 84.38 | 0.7627 | 84.40 | 128/128 | 126.4 |
| `[16,32,16]` | 64 | 87.50 | 0.7588 | 87.90 | 127/128 | 174.9 |
| `[32,16]` | 32 | 95.00 | 0.5155 | 95.38 | 125/128 | 81.4 |
| `[32,16]` | 48 | 92.50 | 0.6441 | 92.83 | 128/128 | 99.7 |
| `[32,16]` | 64 | 77.50 | 0.7058 | 76.13 | 108/124 | 136.5 |
| `[16,16,16]` | 32 | 91.25 | 0.7288 | 91.12 | 127/128 | 101.7 |
| `[16,16,16]` | 48 | 88.75 | 0.6640 | 88.87 | 98/114 | 101.6 |
| `[16,16,16]` | 64 | 85.00 | 0.8893 | 84.00 | 128/128 | 185.0 |
| `[24,24,24]` | 32 | 95.00 | 0.3644 | 95.21 | 128/128 | 102.6 |
| `[24,24,24]` | 48 | 92.50 | 0.3894 | 92.42 | 128/128 | 128.5 |
| `[24,24,24]` | 64 | 75.63 | 0.8364 | 73.32 | 81/97 | 140.8 |
| `[16,16]` | 32 | 90.00 | 0.6453 | 90.12 | 123/128 | 81.2 |
| `[16,16]` | 48 | 77.50 | 0.7515 | 76.51 | 121/128 | 90.5 |
| `[16,16]` | 64 | 75.00 | 0.9561 | 74.68 | 126/128 | 130.0 |
| `[32,16,16]` | 32 | 93.75 | 0.4577 | 93.60 | 118/128 | 103.5 |
| `[32,16,16]` | 48 | 90.00 | 0.6204 | 89.49 | 121/128 | 126.4 |
| `[32,16,16]` | 64 | 51.25 | 1.4856 | 43.52 | 124/128 | 195.5 |
| `[16,16,32]` | 32 | 93.13 | 0.6108 | 93.44 | 127/128 | 104.2 |
| `[16,16,32]` | 48 | 96.88 | 0.4241 | 96.68 | 117/128 | 121.3 |
| `[16,16,32]` | 64 | 31.25 | 1.8468 | 17.94 | 128/128 | 182.0 |
| `[24,16]` | 32 | 45.00 | 1.3089 | 37.78 | 54/70 | 46.2 |
| `[24,16]` | 48 | 84.38 | 0.5494 | 83.81 | 125/128 | 94.4 |
| `[24,16]` | 64 | 85.00 | 0.8336 | 84.85 | 127/128 | 124.5 |
| `[8,16]` | 32 | 78.75 | 0.9702 | 77.98 | 108/124 | 78.7 |
| `[8,16]` | 48 | 75.00 | 1.1208 | 74.36 | 124/128 | 85.7 |
| `[8,16]` | 64 | 59.38 | 1.2067 | 54.23 | 126/128 | 98.3 |
| `[32,32,32]` | 32 | 63.75 | 0.9910 | 60.02 | 88/104 | 85.0 |
| `[32,32,32]` | 48 | 73.13 | 0.9696 | 71.38 | 128/128 | 147.3 |
| `[32,32,32]` | 64 | 61.88 | 1.2726 | 60.02 | 98/114 | 205.0 |
| `[16,16,16,16]` | 32 | 12.50 | 2.3842 | 2.02 | 21/37 | 36.2 |
| `[16,16,16,16]` | 48 | 59.38 | 1.1057 | 54.27 | 119/128 | 139.8 |
| `[16,16,16,16]` | 64 | 15.00 | 2.3578 | 5.94 | 7/23 | 42.6 |

## Evidence verification

- All **738** inventory entries match recorded sizes and SHA256 values;
  all **612** file references in leaf receipts also match.
- Every leaf is complete and uses the same split, seed and locked-test state.
- Every validation prediction set contains exactly the manifest's 160 validation
  clips, with matching labels and no test clips.
- Recomputed accuracy, micro metrics, macro metrics, per-class metrics and
  confusion matrices reconcile with saved evidence.
- Each selected epoch is a minimum-validation-loss epoch and agrees with the
  saved checkpoint metadata. Binary checkpoints were hashed, not loaded or rerun.
- Original videos, inference performance and source independence were not verified
  by this artifact-only review.

[Archive inventory](<../runs/imports/20261009-aad-search-2905fca/archive_inventory.json>) SHA256:
`c40075751931f90718daa4d0cb2223c4f85c8be2dab8d9f1265bce924b110e4a`.

Split manifest SHA256:
`aada503d63d4433b272237db6cf27d7b1c0eaeb0091d5ba8289343a6111454d8`.

The archive is local research evidence and remains untracked. The report records
results and planned changes only; ticket 069, notebook outputs, source code and
the manuscript are untouched. Nothing was committed or pushed.
