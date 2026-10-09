# Paper 001 Implementation

Code for the [journal paper](../README.md). It now provides a small, explicit
family of ConvLSTM architectures for controlled comparison—not an unrestricted
hyperparameter grid.

## Current roadmap

[Ticket 068](../../../agents/work/068-multisize-search-top3-comparison/prompt.md)
implements 12 architectures at three resolutions (36 jobs, 128-epoch cap),
then the top three plus five baselines (eight jobs, 256-epoch cap).
Both stages use seed 42 only; test follows a validation-based freeze.
The [October 9 review](reports/2026-10-09-multiresolution-search-review.md) verifies
all 36 search jobs; comparison/test execution is still pending. `[16,24]` leads
at 91.46% mean validation accuracy, subject to a source-independence audit.
Historical outputs are preserved.

[073 focused shape search and Run All](../../../agents/work/073-focused-shape-search-run-all/prompt.md)
consolidates 070–072. The owner checked the similarly named videos and confirmed
independent recordings; `aad_source_review.json` records that attestation against
the exact split content and dataset inventory. It is not independent evidence of
subject/scene separation. Exact duplicates or conflicting source IDs still block.
Imported 068 evidence is intact under ignored `runs/imports/20261009-aad-search-2905fca/`.

The active notebook defaults to automatic search, comparison, frozen test,
examples and ZIP export. Legacy 068 and 069 profiles remain unchanged.
Local verification is not a measurement of new AAD accuracy or A100 runtime.

AAD is the primary dataset. Run All completes the cached validation-only search,
passes its verified top-three selection internally to comparison, freezes every
checkpoint and the custom example model using validation, then evaluates test.
No manual stage toggle or directory entry is needed. The local profile is only
a pipeline check; incomplete or conflicting evidence still stops execution.

VDD remains optional through the same JSON dataset fields as AAD, with a local
class-folder path and classes discovered from its folders. Preserve historical
VDD runs, but do not treat runs lacking the current split, seeds, metrics and
provenance as comparable new evidence. Kinetics notebooks are exploratory
diagnostics, not part of the active AAD paper study. The active notebook is
[`aad_experiment_workflow.ipynb`](notebooks/aad_experiment_workflow.ipynb), with
a matching [Python export](notebooks/aad_experiment_workflow.py).

## Model context

`ConvLSTM` returns either its full sequence or final hidden state.
`CustomConvLSTM` stacks explicit recurrent layers and uses a
resolution-independent adaptive-pooling head. `PaperConvLSTM` remains separate
because it preserves the source paper's full-sequence, flattened classification
topology. Legacy architecture names and compatibility wrappers are removed.

The [2022 source paper](https://www.mdpi.com/1424-8220/22/8/2946) instead reports
50 RGB frames at `50×50`, keeps the full sequence through `ConvLSTM2D(64)` and
the second `Conv2D(16)`, then flattens `(50, 50, 50, 16)` into `Dense(256)`.
The default 11-class implementation has `512,197,467` trainable parameters and
`128` Batch Normalization state values, matching the paper's reported
`512,197,595` total. This model must remain separate rather than being silently
made lightweight.

### Published-model audit

- The paper reports the layer order, `50×50×3` input, 50 frames, filters,
  kernels, dropout `0.5`, flattened shape, `Dense(256)`, 11-class output, and
  parameter counts. Its reported shapes require same padding and full-sequence
  ConvLSTM output.
- Where the pseudo-code is silent, the implementation follows the documented
  Keras defaults used by the paper: linear Conv/Dense layers, ConvLSTM `tanh`
  state activation, hard-sigmoid gates, Glorot input kernels, orthogonal
  recurrent kernels, unit forget bias, and Batch Normalization defaults.
- PyTorch Batch Normalization uses `eps=0.001` and `momentum=0.01`, equivalent
  to Keras moving-statistic momentum `0.99`. The final layer returns logits
  instead of applying the paper's softmax because cross-entropy applies it.
- The paper does not disclose random seeds or every training/runtime detail.
  Those omissions remain limitations; matching topology and framework defaults
  does not claim bit-for-bit reproduction.

The custom family uses explicit recurrent layer specifications:

```python
layers=[
    (8, (3, 3)),
    (16, (3, 3)),
    (32, (5, 5)),
]
```

Each tuple is `(filters, kernel_size)`. Intermediate layers retain
`(B, T, C, H, W)`; the final layer returns `(B, C, H, W)` to an adaptive-pooling
classification head. Small tensors and short sequences are appropriate for
smoke tests. A smaller screening input is acceptable only when every compared
model uses it and the results are labelled preliminary.

## Baseline warning

Runner configs are grouped under `configs/model/`, `configs/train/`,
`configs/evaluate/`, and `configs/experiments/`.

Training configs include a dataset name and optional local path. To reuse the
local AAD copy or download the public Kaggle copy when missing, run:

```bash
.venv/bin/python -m src.dataset_source --config configs/train/aad_screening_reference.json
```

Public AAD downloads use KaggleHub without a token. VDD and other local datasets
must supply their class-folder path in JSON; the same generic pipeline discovers
their classes. Dataset files stay local and are ignored by Git. Notebook
diagnostics still use their separate Kinetics-oriented configuration and do not
produce the active paper's AAD results.

For an experiment profile, dataset selection is under `dataset`; for example,
replace the AAD fields in a copied profile with:

```json
"dataset": {
  "name": "vdd",
  "path": "datasets/violence-detection-dataset",
  "split_manifest": null
}
```

The directory should contain one subdirectory per class. Do not add fixed VDD
class names; labels are inferred from the folders.

The comparison runner uses `r3d_18`, `mc3_18`, and `r2plus1d_18` with
`weights=None` as this study's practical 18-layer 3D-CNN baselines. The source
paper reports 3D ResNet-50, 3D ResNet-101, and 3D ResNet-152. These groups are
not architecture-equivalent reproductions.

### Focused automatic study (073)

Open `aad_experiment_workflow.ipynb` on Colab and choose Run All, or use its
matching Python export. `WORKFLOW_STAGE = "all"` needs no other control edits.
After dependency setup, the same workflow can be invoked from the implementation
directory with:

```bash
python -c "from notebooks.utils.aad_study import run_full_aad_study; run_full_aad_study('.')"
```

The **34** predeclared custom architectures contain:

- Flat widths 4/8/16/24/32 at depths 1/2/3, plus the [64,64,64] wide control.
- Both orientations of two-layer pairs (8,16), (16,24), (16,32).
- Six non-flat three-layer arrangements using 16/32, plus all six permutations
  of 8/16/32. This covers all 1/3/13 relative-width patterns for depths 1/2/3.

[32,32,32] and [64,64,64] remain wide controls; no four-layer or 48-wide grid.
This is an evidence-guided follow-up, not an unbiased or exhaustive search.
Poor convergence is not proof of a capacity limit.

| Stage | Upper jobs before exact reuse |
|---|---:|
| LR calibration | 12 |
| Flat models at 32/48/64 pixels | 48 |
| R3D-18 and Swin3D-T references | 6 |
| Shaped models at all three resolutions | 54 |
| Two-seed shortlist and flat-neighbour confirmation | 24 |
| Separate weight-decay ablation | 18 |
| Separate FPS/frame-count ablation | 15 |

Search: **177 maximum**, cap 128 epochs, minimum 64, patience 24; otherwise
the 069 common recipe and at-most-16-frame timestamp sampling remain in place.
Exact correct/total counts break accuracy ties fairly before loss/parameters/name;
resolutions and model seeds receive equal weight. Old scores are not pooled in.

Final comparison: **eight fresh trainings**, cap 512 epochs, minimum 128,
patience 32, fixed 50 frames at 50x50, batch 1, seed 42. Caps are not promises
to finish that many epochs or within eight hours. Test starts only after all
eight validation-selected checkpoints and the custom example model are frozen.
All models receive full metrics, including explicitly macro/micro metrics;
the validation-selected custom model supplies correct/incorrect video examples.

Rerunning Run All verifies and reuses complete matching work. A failed/partial
test attempt never automatically runs again. Stage ZIPs and the final combined
ZIP retain configs, receipts, histories, checkpoints and reports, not datasets
or caches. Download-only retry never trains or tests. Save the ZIP before the VM
is deleted; a browser download request does not prove delivery.

The existing tracked split resolves automatically; changed datasets/assignments
require a new source review, never a silent resplit. The attestation compares
canonical JSON content so Windows line endings cannot change its meaning.

Use one runner and one JSON path for each profile:

| Profile | Fixed protocol |
|---|---|
| [Local smoke](configs/experiments/aad_local_smoke.json) | One tiny 1-epoch pass; not paper evidence |
| [Focused search (073)](configs/experiments/aad_shape_search_colab.json) | 34 custom stacks; up to 177 search jobs, 128-epoch cap; automatic notebook default |
| [Final comparison (073)](configs/experiments/aad_final_comparison_colab.json) | Verified top three plus five baselines; 512-epoch cap, minimum 128, patience 32 |
| [Capacity search (069)](configs/experiments/aad_capacity_search_colab.json) | Adaptive validation-only stages; at most 156 jobs before dedup/reuse, 200-epoch cap; see below |
| [Custom search](configs/experiments/aad_custom_search_colab.json) | 12 architectures x 32/48/64 pixels; 8 frames; 36 jobs, up to 128 epochs, patience 16 |
| [Comparison template](configs/experiments/aad_model_comparison_colab.json) | Top 3 custom + published ConvLSTM + R3D-18/MC3-18 + Swin3D-T/S; eight fresh trainings, 50 frames at 50x50, batch 1; up to 256 epochs, minimum 32, patience 12 |

The legacy 068 search and comparison use model/split seed 42. Caps are budgets, not guaranteed completed
epochs. Checkpoints minimise validation loss. Search holds the optimiser,
LR, weight decay, augmentation and stopping rule fixed; there is no temporal
sweep, winner-only input check or second-seed confirmation.

The top-three ranking requires the complete 36-job matrix: equal-weight mean
validation accuracy over the three resolutions, then mean loss, parameters and
name. It retains each resolution's score/rank and worst-case accuracy. Exact
layers/heads, source runs and config/split hashes accompany the handoff.
Incomplete or altered evidence is rejected; there is no placeholder model.

Comparison keeps Adam, LR 0.001, weight decay 0.0001, no augmentation and the
gentle shared plateau scheduler. It never warm-starts search checkpoints.
All eight checkpoints/configs and the best *custom* identity are frozen from
validation before any test decoding. Each gets one full test evaluation.
Only the validation-selected best custom saves up to three correct and three
incorrect playable test examples; missing categories are reported honestly.

Single-seed findings do not estimate seed uncertainty. Resolution coverage
does not prove a global optimum or unseen-data generalisation; clip-level
stratification does not establish source-group independence.

### Run and resume

The [active notebook](notebooks/aad_experiment_workflow.ipynb) and matching
[Python export](notebooks/aad_experiment_workflow.py) support:

- `WORKFLOW_STAGE = "all"` (default): focused 073 search, verified handoff, comparison,
  validation freeze, one-time test/examples and ZIP. No manually supplied paths.
- `"capacity_search"`: run the staged legacy 069 protocol
  below, export validation-only top three and ZIP evidence; never run comparison.
- `"search"`: run the legacy 068 profile; benchmark one common memory-safe batch,
  run the matrix, print the saved `selected_config.json`, then package/download.
- `"comparison"`: set `SELECTED_CONFIG_PATH` to that saved selection. Train
  eight models once, freeze, test and export the combined results.
- `"smoke"`: validate/list profiles only; no dataset download or training.
- `"resume"`: set `RESOLVED_PROFILE_PATH` to a saved resolved search/comparison/capacity
  JSON under `runs/notebook_studies/`. Verify saved work and execute pending jobs.
- `"download"`: set `ARTIFACT_ZIP_PATH`; verify/request download only.
- `"repackage"`: set `SAVED_STUDY_DIRECTORY`; rebuild the ZIP from saved evidence
  only, then use `"download"`. Capacity groups under `runs/studies/` are supported.

Rerunning search/comparison with the same request reuses its saved resolved
profile. Completed jobs are checksum-verified, not silently retrained/retested.
Interrupted training/test attempts, changed code/data, or incompatible evidence
stop for inspection; do not delete guards to tune against test. A forced VM
shutdown may leave a lock: first confirm no worker is running and inspect its
receipts. A new comparison is not an automatic recovery action.

Before expensive runs, list the plan or execute the local smoke:

`python -m src.experiments --config configs/experiments/aad_custom_search_colab.json --list-plan`

`python -m src.experiments --config configs/experiments/aad_local_smoke.json`

The comparison template deliberately cannot execute without verified top-three
evidence. Its resolved JSON supports the same `--list-plan` and execution command.
Legacy grids and notebook 05 remain for historical workflows, not the active study.

### Staged capacity search (069)

Run from this implementation directory. Set `dataset.path` and, if necessary,
`dataset.split_manifest` in the new profile to the existing AAD data/split.
The default split lookup must find the existing seed-42 manifest; execution
stops instead of silently creating a new split.

```bash
python -m src.experiments --config configs/experiments/aad_capacity_search_colab.json --list-plan
python -m src.experiments --config configs/experiments/aad_capacity_search_colab.json
```

| Stage | Planned jobs before exact-config reuse |
|---|---:|
| LR calibration: two custom representatives + R3D-18/Swin3D-T, three LRs | 12 |
| Flat widths 4/8/16/24/32/48/64, depths 1/2/3, three resolutions | 63 |
| Two reference models at 32/48/64 pixels | 6 |
| Best three-layer width: four-layer/half-width variants and two controls | Up to 18 |
| Shortlist + references + peak/neighbours, seed 2026, three resolutions | Up to 24 |
| Confirmed custom top three: WD 1e-4/1e-3, three resolutions; WD 0 reused | 18 |
| Custom top three + references: (8 frames, 8 FPS), (8, 4), (16, 16) | 15 |

Upper budget: **156**, reduced by deduplication and verified exact-config reuse.
Reference input is 8 frames/16 FPS. Temporal ablations fix 48x48 and WD 0;
WD ablations fix reference timing. All runs use Adam, no augmentation, up to
200 epochs, minimum 64 before early stopping, patience 24 and the gentle shared
plateau schedule (factor 0.9, patience 5). The lowest-validation-loss checkpoint
is retained. Calibration freezes one custom LR and a separate LR per reference;
the families receive unequal search effort, not exhaustive baseline tuning.

Before training, the largest four-layer custom stack and both references are
tested at 16 frames/64x64 to choose one common safe batch. Resolved settings and
conservative throughput estimates are saved; CPU verification cannot certify
GPU safety. Runtime varies by architecture/epochs/input; 156 jobs at up to 200
epochs may take substantially longer than earlier runs. Checkpoint storage and
temporary ZIP replacement need disk headroom; one selected checkpoint per job
is kept, not every epoch.

Custom ranking uses equally weighted validation accuracy across all three
resolutions, then mean loss, parameters and name. Shallow models remain eligible.
Only the declared top-three shortlist is reordered using both seeds; peak and
neighbour confirmation is sensitivity evidence. Reports include each seed's
resolution mean and sample SD across the two seed means (not across resolutions).
A boundary winner is flagged, never claimed to establish a turning point or
global optimum. A non-converged run also cannot establish a capacity limit.

`timestamps_v1` starts at the first source timestamp and uses the latest frame
at or before each target timestamp. Short clips repeat their last frame; low-FPS
sources repeat without speeding up motion. VFR uses actual timestamps, not a
fabricated constant source FPS. Per-video audits record coverage, selected
timestamps, unique frames, padding and duplicate fractions. Repeats are not new
temporal information. Missing/broken timestamps stop the run. Legacy sampling
is unchanged; sampler version, FPS, frame count, size and split isolate caches.
Only train/validation frames are probed/decoded/cached. File hashes and available
source IDs are audited across the manifest without decoding test videos; known
cross-split duplicates block training, and missing source grouping stays an
explicit limitation. Full-video/stateful processing remains out of scope.

Each stage prints **all** completed configurations and failures/pending jobs;
dependent unresolved stages remain visible. The capacity group saves
`results.json`, `results.csv`, `results.md`, width/depth and efficiency plots,
`decisions.json`, two-seed rankings and `ablation_recommendations.json`.
Leaves retain complete validation predictions, confusion/per-class metrics,
explicit micro/macro metrics, balanced accuracy, LR histories and checkpoints.
Model-only inference uses batch 1, three warmups and ten synchronized measured
forwards at each declared input; hardware/dtype, latency, throughput and CUDA
peak memory are recorded separately from decode-inclusive training time.
CPU GPU-memory measurements are unavailable, not zero.

Receipts validate configs, split/code/data identity, metrics and file checksums.
Complete matching jobs resume without retraining; failed/interrupted jobs or
changed evidence stop for inspection. Do not delete guards to force a restart.
Partial/complete stage ZIPs live under `runs/capacity_archives/`; the notebook
also packages its logs and resolved profile. Retry packaging/download is
independent of training and testing. Download and verify before deleting a VM.

`selected_config.json` exports exact layers/heads and six source receipts per
custom model (two seeds x three resolutions), separately from single-factor
ablation options/recommendations. Recommendations use validation accuracy then
loss; ties prefer lower WD or the reference temporal input. Both remain
single-seed evidence, never an inferred combined recipe.
The selection is accepted by the existing comparison handoff, but does
not launch comparison or combine ablation winners. **The 16-frame cap applies
only to 069**: the eight-model, 50-frame/50x50, batch-1, 256-epoch final comparison
and its frozen-test gate are unchanged; execution/protocol review is later work.

### Evidence and download

Train/validation RAM caching is keyed by dataset file metadata, preprocessing,
split, frame count, resolution and FPS. One configuration is retained at a time,
bounded to 2 GiB per runner process; larger inputs fall back to lazy decoding.
Resolution-grouped jobs reuse the cache. Test clips are never cached during
selection. Every leaf records cache footprint/reuse, configuration, provenance,
per-epoch history/progress and its single validation-selected checkpoint.

Search selections and comparison tables live in `runs/studies/`; leaf runs in
`runs/experiments/`; full test reports in `runs/evaluate/`. `comparison.json` and
`comparison.csv` include validation/test loss, accuracy and explicitly micro
precision/recall/F1, parameter counts, state-dict tensor bytes versus full
checkpoint bytes, actual/selected epochs and timing. Linked reports retain
per-class metrics, confusion matrices, all predictions and training curves.
Training time includes train/validation epochs and checkpoint/history writes;
metric-pass time includes decoding/loading, not pure inference latency.
Micro precision/recall/F1 equal accuracy for this single-label multiclass task.

The ZIP includes completed/partial leaves, study selections/freezes/receipts,
the exact split, configs, logs, checkpoints, tables and example videos. Raw data
and caches are excluded. Packaging checks free disk space and verifies the
complete SHA-256 inventory and ZIP reads before replacing an older archive.
Failures remain visible; retry packaging/download never trains or tests.
A browser download request is not proof of a completed download. Check your
local file before deleting the Colab runtime: ZIP-only storage cannot survive
VM deletion.

Execution/analysis remain tracked by [053](../../../agents/work/053-run-and-analyse-aad-experiments/prompt.md);
manuscript updates wait for reviewed, versioned evidence
([062](../../../agents/work/062-update-manuscript-after-aad-results/prompt.md)).
The unarchived reported >90% validation score is context, not paper evidence.

## Modular Colab notebooks

### AAD final training and test (Colab)

The [05 AAD final training/evaluation](notebooks/05_aad_train_and_evaluate.ipynb)
workflow is retained from the earlier study scaffolding. It is separate from
the current search/comparison notebook and is not used by the active profiles.
Keep its linked runs, dataset and split manifest available at their recorded
paths when using it for later final training/evaluation.

The matching [plain-Python export](notebooks/05_aad_train_and_evaluate.py) uses
the same setup, pinned requirements and GPU check. From the Colab implementation
directory, run:

```bash
AAD_STUDY_RUN_DIR="runs/studies/<completed-run>" \
  python notebooks/05_aad_train_and_evaluate.py
```

The notebook contains no magics, so a normal Python export or one-cell copy
also works. Push the complete code update before using its checkout setup.
If dependency installation requests a restart, restart once and rerun.

It validates every study job, chooses the best tested single-factor configuration
of the architecture winner by seed-mean validation accuracy/loss/parameter count,
then confirms seeds 42/2026 for at most 64 epochs. No factors are combined.
This is ticket 026's confirmation budget, not longer training than the current
160-epoch screen. Each seed restores its minimum-validation-loss checkpoint;
lowest confirmation validation loss selects the final checkpoint, with accuracy
then seed as tie-breakers. Both seeds' mean/sample SD and curves remain available.

Only then does final testing export overall micro and per-class metrics, full
prediction records, confusion output and up to three correct/incorrect videos.
Inference timing includes data loading/decoding; it is not pure model latency.
Custom, practical 3D-CNN and Swin winners use their unchanged registered topology.

Selection/progress/checkpoint hashes live in `<study>/final/`; training leaves
reuse `runs/experiments/`, and evaluation uses `runs/evaluate/`. Completed
reruns verify and reuse evidence. Interrupted seed/test attempts stop for
inspection instead of silently repeating; do not delete guards to tune on test.
The final display cell can be rerun independently after a renderer failure.
Save the study, linked runs/checkpoints and evaluation folder before Colab
disconnects: `/content` is temporary. Real 026/027 runs are still pending.

### Exploratory diagnostic notebooks (not paper results)

Notebook-only helpers and the Colab bootstrap live in `notebooks/utils/`;
reusable model, dataset, training, evaluation and CLI modules remain in `src/`.
Helper filenames are concise (`config.py`, `data.py`, `workflows.py`); their
directory already identifies them as notebook support.

If an old export reports missing `src/notebook_data.py` or similar files, reopen
and re-export the current notebook from master: the helpers moved to
`notebooks/utils/`. Pulling inside an old setup cell updates repository files,
not that cell's source. All current notebooks use `notebooks.utils` imports.

[Ticket 047](../../../agents/work/047-larger-colab-training/prompt.md) implements
the larger notebook-03-only profile below. After pulling master, close and reopen
notebook 03 and create a fresh export; pulling inside an old export cannot replace
that export's removed `src.notebook_*` imports. Setup uses a safe fast-forward
pull and stops on conflicts without discarding local edits.

Integration [ticket 042](../../../agents/work/042-merge-expanded-kinetics-suite/prompt.md)
combines Kinetics-600 (>2,000 unique videos before splitting) with the expanded
eight-hour suite. Real Colab inventory/training remains pending.

These independent notebook flows are not the active AAD experiment runner and
their Kinetics/VDD outputs are not new comparable paper evidence. Open a
diagnostic notebook and choose **Run All** on a Colab GPU. The four notebooks use
`SELECTED_DIAGNOSTIC_DATASET = "kinetics600-subset"` in
[shared configuration](notebooks/utils/config.py). Preparation downloads only the
five Kinetics-600 training archives (about 3 GB compressed), extracts them
safely, prints actual per-class/total counts and stops unless it finds over
2,000 unique clips. The exact extracted total remains to be observed on Colab.
This is an exploratory five-activity diagnostic, not surveillance-paper evidence.
The notebook-only dataset selector can use VDD or the previous Kinetics-400 copy,
but the active modular paper profiles select datasets in JSON. Existing saved
notebook outputs describe earlier runs. Kinetics diagnostic notebook 04 keeps
its separate model manifest at
`configs/experiments/kinetics_diagnostic_candidates.json`.

| Notebook | End-to-end workflow |
|---|---|
| [AAD experiments](notebooks/aad_experiment_workflow.ipynb) | Default Run All: focused search, comparison, frozen test, examples and ZIP without manual stage/path handoff |
| [01 Dataset](notebooks/01_dataset_setup.ipynb) | Split summary; the same training video at native FPS, sampled FPS, then augmented |
| [02 Model](notebooks/02_model_inspection.ipynb) | Architecture, parameter count, one random-weight prediction, playable labelled video and probabilities |
| [03 Training](notebooks/03_train_model.ipynb) | Train/validate one model, live epoch curves, restore the selected checkpoint, show five validation predictions |
| [04 Kinetics diagnostics](notebooks/04_run_experiments.ipynb) | Exploratory temporal audit, small learnability check, diagnostic comparisons and test examples; not current AAD paper evidence |

The legacy manual AAD search uses the local smoke profile for a no-training
configuration check and runs the custom-search or comparison JSON through
`src.experiments`. Search first caches sampled clips, checks batch-16 versus
batch-32 throughput, then applies the measured batch consistently. Set
`WORKFLOW_STAGE = "smoke"` for profile checks; use `"comparison"` later and
point `SELECTED_CONFIG_PATH` at the saved validation selection. Successful runs
write directly to `runs/`; each stage downloads one ZIP with profiles, progress,
and new
run folders (including histories, metrics, checkpoints, confusion matrices, and
prediction videos). It imposes no wall-clock cutoff. Colab may still terminate
the runtime externally before the final ZIP can be created or downloaded; the
notebook does not promise recovery after forced termination. At the current
AAD reference input, only 748 train and 160 validation clips are cached (about
0.09 GB); test clips are not decoded or cached. Dataset files and unrelated older
runs are excluded from the ZIP.

The notebook is the source of truth; the committed `.py` file is its matching
one-cell export. Both clone/update the repository, prepare pinned dependencies,
check the attached GPU, validate all three profiles, and call the same modular
commands. If the pinned runtime needs repair, restart Colab and rerun the cell.
AAD uses the existing public tokenless KaggleHub resolver when not already
present; no manual dataset upload is needed.

Notebooks 03/04 have fresh filenames to work around VS Code's logged
`notebook controller is DISPOSED` error: copies ran while the old paths did not.
Their cells, metadata and saved outputs are preserved. Close the old tabs, open
the files linked above and select the working Colab GPU kernel for each.
This is a filename-based workaround; local tests cannot verify the editor's Run button.

Notebooks 01/02 retain 16 frames at 16 FPS, batch 8, seed 42 and 32×32 input.
Notebook 03 uses ConvLSTM `32 → 64 → 64` (3×3 kernels), 32 RGB frames at 8 FPS,
96×96, batch 8 and seed 42, for at most 200 epochs or eight diagnostic hours after preparation.
The five Kinetics-600 classes, all usable clips, >2,000-video gate and source-grouped
split are unchanged. Edit only `TRAIN_*` for this profile. Its resolved configuration,
parameter count and live curves are printed; the lowest-validation-loss checkpoint
supplies five validation predictions. Test stays locked.

The clock is checked between batches/epochs; in-flight work and output saving can
overrun it. Time-limited runs retain completed validation epochs and are labelled
partial, never 200-epoch results. If no validation epoch completed, no checkpoint
or predictions are claimed. Decoder errors remain failures, not handled budget stops.
The larger profile has local synthetic tests, not an A100 accuracy/memory guarantee.

Notebook 04 has separate `SUITE_*` settings:
32 frames at 8 FPS, 64×64 screening input, 24 screening epochs and 64 confirmation
epochs. The retained legacy `SCREEN_*` quick-grid lists do not control notebook 04.
The notebook-only Adam schedule starts at 0.001, halves after six
non-improving validation epochs and floors at 0.00001. Custom dropout is 0.1,
weight decay 0.0001 and spatial augmentation is off. Controlled AAD defaults are unchanged.
Notebook 04 declares the following suite before training:

1. Audit train/validation video timestamps, source FPS, duration and repeated
   thumbnails; save checksums. Do not inspect test or rewrite source files.
2. Memorize two fixed training clips per class with a `16–32` custom stack,
   dropout/weight decay/scheduler off, up to 512 updates. Require at least 95%
   clean training accuracy and reduced loss; otherwise stop with diagnostics.
3. Compare the existing eleven custom candidates and three scratch 3D CNNs for
   24 epochs each, identical inputs/split/budget. No pretrained weights.
4. On the validation-selected custom reference, compare 64/96/128 pixels,
   4/8/16 FPS, dropout 0/0.1/0.5 and weight decay 0/0.0001/0.001 one factor at a
   time with the same split and 24-epoch budget. Fixed 32 frames
   make this a temporal **coverage** comparison (about 8/4/2 seconds), not isolated FPS.
5. Attempt the native 50-frame/50×50 published topology separately, with a
   dataset-specific output head. Memory preflight and a 30-minute stage cap
   preserve time for confirmation. Resource-limited status is not a result;
   the topology is never downsized or ranked alongside matched-input models.
6. Select the custom configuration by validation; retrain it and all three
   CNNs **from scratch** for 64 epochs at max(96, selected size), both seeds
   42/2026, with matched
   sampling/optimizer/budgets. Rank mean validation metrics; export seed variation.
7. Freeze all eight compatible checkpoints before exploratory test scoring.
   Show five examples from the validation winner's predeclared seed 42, never
   select a seed/model from test results.

The plan prints and saves **31 training runs** (14 screen, eight one-factor,
one native attempt, eight confirmation), plus the tiny check. This is a declared
plan, not a promise all runs finish on A100 within eight hours.
The shared eight-hour **compute** deadline starts at `run_screen` (after setup/
downloads), spans subsequent cells and is checked between batches/audit frames.
An in-flight CUDA operation, checkpoint write or frontend render may finish
after the deadline. Interrupted/budget-limited evidence remains partial; it
cannot trigger final testing. Completed trials and checkpoint/history files
remain saved, but automatic mid-trial resume is not implemented. Run All starts
a new suite, so do not rerun it merely to recover display.
Checkpoints use minimum validation loss; candidate ranking uses validation
accuracy, loss, parameter count, then name. Batch progress is printed,
loss/accuracy plots update each epoch, and artifacts go under ignored `runs/`.

These are **exploratory diagnostics**, not the active AAD paper workflow or
evidence of cross-dataset generalisation. Notebook 04 follows the selected
diagnostic dataset. The active AAD profiles and CLI commands are listed above;
the diagnostic notebook's separate budget does not impose a limit on them.

Preparation inventories and hashes files without decoding test frames. It reuses
the approximately stratified 70:15:15 manifest; Kinetics-600 clips sharing known
source-video IDs stay together. Exact byte-identical copies are recorded and
excluded; if identical content appears under different activity labels, all
copies are excluded as ambiguous. The 2,001-video gate counts only usable unique
clips. The diagnostic loader decodes only the
chosen input window, samples by timestamps and keeps a bounded 64 MiB cache.
Training windows use seeded random offsets per epoch; validation, clean training
checks and test use fixed central windows. Cache keys include temporal state.
Notebook 01 displays native/sampled/augmented versions of the same central window.
Short videos repeat the last sampled frame; the duration audit makes these visible.
Corrupt clips fail explicitly instead of substituting another split's sample.
This preprocessing is recorded separately from the legacy AAD loader.
The Kinetics-600 split is grouped by known YouTube ID, but subject independence
and absence of near-duplicates are **not** established. Other diagnostic splits
remain clip-level.

Notebook 04 opens test automatically only after all matched candidates/ablations
and both seeds of every confirmation model finish and are frozen. Metrics precede example videos;
metrics and the final report are saved before rendering. Rerunning its final
cell reuses completed metrics/reports after an export/display failure. Do not
tune or rerank from test feedback. Notebooks 01–03 never evaluate test. No
accuracy threshold is promised.

The expanded suite exports `source_audit.json`, `tiny.json`, predeclared plans,
`screen.csv`, `ablations.csv`, `confirmation.csv`, `confirmation_mean_std.csv`
and `test_exploratory.csv` under the printed suite directory. Each training run
retains config, split identity, code revision, model specification, parameters,
selected checkpoint, clean training metrics and curves; epoch histories include
learning rate, gradient norm, sampled parameter-update norm and runtime.
Copy the suite **and its referenced training directories** out of `/content`
before Colab disconnects; notebook displays alone are not the full evidence.

Kinetics-400/VDD test scores have already informed exploratory discussion. They
must not be presented as a new untouched final holdout. Good performance still
requires clip/source-quality checks, suitable independent confirmation and the
separate AAD/VDD paper protocol before manuscript claims. Poor performance across
models does not establish bad FPS/data: optimization and shared sampling remain
possible causes. No recurrent state is carried across videos or splits.

Every setup cell safely fast-forwards its checkout, verifies the required source
files, prints the code revision, checks NumPy and notebook imports in a fresh
Python process, and installs `requirements.txt` only when pinned versions do not
already match. A broken NumPy wheel is repaired and triggers one kernel restart
before imports. Git preserves unrelated edits and refuses updates
that would overwrite conflicting edits; no reset, clean or automatic stash is used.
If dependency versions change, restart this notebook's Python kernel once and
rerun from the top; keep the same A100 runtime if it remains attached. Updated
local code must be pushed/synced before Colab can use it.
Setup prints the GPU attached to each notebook's kernel, accepts CUDA wheel
version suffixes, and installs the pinned CUDA pair when a visible GPU has a
CPU-only or incompatible PyTorch installation. Select the Colab kernel separately
for each notebook; a GPU connection in another notebook does not confirm this one.
`notebooks.utils.data` is a repository module, not a pip package: a kernel restart
cannot download source files that have not been committed and pushed.
Active notebooks retain the remote laptop's historical saved outputs unchanged;
they do not demonstrate the revised code has run. The local versions remain in
recovery commit `42f2130` (branch `recovery/pre-042-20261005`). Both ticket-041
records are retained under their distinct full names. Open the updated notebook
from Git before Run All; updating a checkout does not rewrite already-open cells.

### Recover an existing Colab video-display failure

Older IPython versions can raise `TypeError: stat ... NoneType` for
`Video(filename=...)`; the helper now passes the path positionally. If an
existing run failed in `show_predictions` after saving its examples, run this
cell in the same session to display them without retraining or rescoring test:

```python
import json
from pathlib import Path
from IPython.display import Video as _Video
import notebooks.utils.display as nd

nd.Video = lambda filename, **kwargs: _Video(filename, **kwargs)
records = json.loads(
    (Path(screen_dir) / "test_predictions" / "predictions.json").read_text()
)
nd.show_predictions(records)
```

This recovers saved examples only; runs before the persistence fix may have
displayed test metrics without saving a final report. Do not rerun training.

### Kinetics-600 diagnostic source

Notebook 01 downloads only the training archives for `headbutting`, `slapping`,
`punching person (boxing)`, `hugging (not baby)`, and `shaking hands` from the
[CVDF Kinetics-600 release](https://github.com/cvdfoundation/kinetics-dataset).
The five archives total about 3 GB compressed. Extraction verifies the archive,
rejects unsafe paths/links and records content hashes. Preparation prints unique
counts per activity and requires at least 2,001 clips before creating the
source-ID grouped 70:15:15 split. The upstream annotation CSV is incomplete, so
the actual extracted count is pending the first Colab Run All. The split uses
only the upstream training archives and is not an official Kinetics benchmark
partition. Subject independence and absence of near-duplicates are unknown.

The local cache is `runs/datasets/kinetics600-five-activities/`. To reuse a
copied extracted root, keep its `.kinetics600-source.json` provenance file with
the video folders. The existing Kinetics-400 subset below remains available by
changing the shared dataset setting to `kinetics-subset`.

### Previous Kinetics-400 interest-filtered subset

`CLASSES_OF_INTEREST` contains the existing AAD labels and these exact
Kinetics folders. Only Kinetics uses the intersection with available classes;
absent names are printed and skipped. Labels stay separate activities, not a
binary violence mapping. Fewer than two matches or an impossible stratified
split fails clearly instead of loading everything.

| Version-1 folder | Clips |
|---|---:|
| `headbutting` | 22 |
| `slapping` | 13 |
| `punching_person__boxing_` | 14 |
| `hugging` | 16 |
| `shaking_hands` | 22 |

This previous, optional source is pinned to
`sanelehlabisa/kinetics-400-dataset/versions/1`. It contains 60 training, 13
validation and 14 locked test clips after the interest filter.
Setup inventories public filenames (55 pages on first use), then downloads
only the 87 matching videos: 91.1 MB, not the full 16.5 GB copy. Host mounts
are temporarily disabled for these file-only requests and restored afterward.
Inventory and size-checked clips are reused under ignored `runs/datasets/`;
partial downloads can be retried; non-video `.mp4.part` inventory entries are
ignored. A supplied local class-folder root skips
all network calls. No automatic full-download fallback exists.

Train/validation class counts and majority-class baselines are printed before
training. All 87 selected clips and the existing split are retained; ticket 041
changes input/training protocols, so its runs are not matched to older results.
Kinetics manifests are keyed by pinned version, matched classes and seed, so
changing the subset preserves older manifests and rejects old checkpoints.
These historical Kinetics workflows are separate from the current AAD
custom-search and model-family comparison profiles.

This is a tiny learnability diagnostic, not the full official Kinetics release
or surveillance evidence; held-out estimates are noisy. The broader audit in
ticket 036 remains incomplete: metadata does not establish clip quality,
source independence or absence of near duplicates.

## Implementation history and next steps

1. [x] **Architecture preparation (`010`).** Added the faithful paper model,
   sequence-returning ConvLSTM, explicit `(filters, kernel_size)` stacks, a
   lightweight adaptive-pooling head, and parameter counting.
2. [x] **Model API cleanup (`011`).** Kept `ConvLSTM`, `PaperConvLSTM`, and
   `CustomConvLSTM`; removed obsolete classes and migrated all active callers.
3. [x] **Baseline audit and alignment (`012`).** Verified the source-paper
   topology and separated its deeper comparisons from this study's three
   practical 18-layer 3D-CNN baselines.
4. [x] **Video augmentation (`013`).** Replaced augmented dataset copies with
   one optional, clip-consistent online view per training sample and added AAD
   commands for every executable module.
5. [x] **Run artifacts (`014`).** Model, training, evaluation, and comparison
   commands now create separate local runs with reusable JSON, checkpoints,
   plots, confusion matrices, and predictions.
6. [x] **Reproducible data (`015`).** Seeded the full pipeline and added a
   fixed, stratified 70:15:15 clip manifest.
7. [x] **Valid selection and metrics (`016`).** Added full-partition metrics,
   validation-loss selection and early stopping, restored selected checkpoints,
   and reserved test access for evaluation.
8. [x] **Configuration integration (`017`).** Added one validated JSON schema,
   explicit CLI overrides, stable resolved artifacts, and configuration-aware
   checkpoints for training and comparisons.
9. [x] **Architecture shortlist preparation (`018`).** Marked historical outputs
   non-comparable and added one validated initial custom candidate manifest
   with complete provenance.
10. [x] **Cloud experiment workflow (`019`).** Added one guarded local/Colab/
    Kaggle notebook that calls the existing CLIs and preserves test isolation.
11. [x] **Controlled experiment expansion (`020`).** Added the approved 11-model
    screen, practical-baseline confirmation, separate published topology, and
    validated one-factor ablation plan to the Colab workflow.
12. [x] **Optional cloud expansion (`021`).** Skipped; retain the existing
    focused architecture screen.
13. [x] **Notebook foundation (`022`).** Added the Colab VS Code runtime,
    committed-split, class-balance, training-sample, and augmentation walkthrough.
14. [x] **Model inspection (`023`).** Added inline clean and augmented videos,
    manifest-driven candidate and role tables, a readable lightweight reference,
    and one labelled random-weight GPU prediction with cleanup.
15. [x] **Experiment notebook (`024`).** Added bounded learning checks, guarded
    single-stage execution, and validation-only result displays.
16. [x] **Separate AAD search and comparison (`058`).** The active profiles
    below now search custom ConvLSTM architectures first, then compare the
    selected model with fixed settings.
17. [x] **Retire VDD-only workflow (`059`).** VDD remains optional through
    generic JSON dataset selection; its historical outputs are preserved.
18. [x] **Align dataset documentation (`060`).** These guides now describe the
    AAD-first two-stage study and distinguish old diagnostics.
19. [x] **Colab runner and artifact download (`061`).** The AAD notebook runs
    the modular profiles and downloads one archive of current run artifacts.
20. [ ] **Analyze verified results (`053–054`).** Prepare paper tables and
    figures only after the new AAD runs are complete and reviewed.
21. [ ] **Final training/evaluation (`026–027`).** Confirm the selected model
    across seeds, then open the locked test set once.
22. [ ] **Optional VDD generalization (`028`).** Consider separately after AAD
    results only if a cross-dataset study is needed; it is not active now.
23. [ ] **Kinetics transfer (`032`, conditional).** Consider only if justified
    by verified AAD evidence and a separate transfer protocol.

Generated datasets, environments, checkpoints, and experiment runs remain
untracked. New command artifacts live under
`runs/{model,train,evaluate,experiments}/<datetime>_<dataset>_<label>/`; each
leaf owns its `run.json` and all related evidence. Legacy output directories
were removed from Git; retained pre-014 experiment and prediction files are
archived locally under the matching `runs/` purpose. The ignored `models/`
directory remains available to active legacy jobs. Do not change manuscript
results until versioned evidence exists.

The split manifest records relative clip paths, class labels, indices, and
assignments. It is a clip-level protocol because the source dataset provides no
reliable subject or source-video grouping metadata.

## Setup and commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` contains only direct runtime dependencies and pins each one
exactly. Pip resolves platform-specific transitive packages. Development tools
remain separate in `pyproject.toml`; IPython is supplied by the notebook host.

Run the following commands from this `implementation/` directory. They use the
larger Abnormal Activities Dataset (AAD).

Preview clean and online-augmented pairs (writes MP4 files under `outputs/`):

```bash
.venv/bin/python -m src.dataset \
  --dataset_dir "datasets/abnormal-activities-dataset/abnormal-activities-dataset" \
  --sequence_length 32 \
  --height 64 \
  --width 64 \
  --num_samples 2 \
  --fps 8 \
  --seed 42 \
  --augment
```

Check source videos without creating a processed dataset (read-only, but slow):

```bash
.venv/bin/python -m src.preprocess_dataset \
  --dataset_dir "datasets/abnormal-activities-dataset/abnormal-activities-dataset" \
  --output_name "abnormal-activities-dataset" \
  --frame_size 256 \
  --dry_run
```

Smoke-test both random-weight ConvLSTM models (writes sample predictions):

```bash
.venv/bin/python -m src.model \
  --dataset-dir "datasets/abnormal-activities-dataset/abnormal-activities-dataset" \
  --runs_dir "runs" \
  --sequence-length 64 \
  --height 8 \
  --width 8 \
  --seed 42 \
  --convlstm-layer 8 3 3 \
  --convlstm-layer 16 3 3
```

Output: `runs/model/<run>/`.

Inspect the AAD screening reference without loading data or creating a run:

```bash
.venv/bin/python -m src.train \
  --config configs/train/aad_screening_reference.json \
  --print-config
```

Train the configured custom reference. This starts expensive training:

```bash
.venv/bin/python -m src.train \
  --config configs/train/aad_screening_reference.json
```

Output: `runs/train/<run>/`. Training restores the lowest-validation-loss
checkpoint and never opens the test split.

The older multi-file plan commands below remain available to the legacy notebook
workflow only. Use the three active single-JSON profiles above for the current
paper study; they all share `src.experiments --config <json>`.

Run legacy practical-baseline confirmation only after replacing `REFERENCE` with the
validation-selected custom candidate. This runs both confirmation seeds and
starts expensive training:

```bash
.venv/bin/python -m src.experiments \
  --plan-config configs/experiments/aad_controlled_experiment_plan.json \
  --run-plan-stage baseline-confirmation \
  --reference-candidate REFERENCE
```

Run the focused weight-decay, augmentation, spatial-size, and sequence-length
comparisons after freezing the same reference. Each non-reference run changes
one factor and both seeds are run:

```bash
.venv/bin/python -m src.experiments \
  --plan-config configs/experiments/aad_controlled_experiment_plan.json \
  --run-plan-stage focused-ablations \
  --reference-candidate REFERENCE
```

The audited published topology is a separate, very expensive native-input run:

```bash
.venv/bin/python -m src.experiments \
  --plan-config configs/experiments/aad_controlled_experiment_plan.json \
  --run-plan-stage published-topology
```

It uses 50 frames at `50x50` and is not a one-factor comparison with the
lightweight models. If resources are insufficient, record that limitation; do
not reduce the model and still call it faithful. All comparison outputs live
under `runs/experiments/<run>/`, rank validation evidence only, and keep test
access locked.

After freezing a configuration, deliberately evaluate its validation-selected
checkpoint on the AAD test split (writes metrics, a confusion matrix, and
up to three correct and three incorrect clips by default). All four CLI
commands use JSON `prediction_samples_per_category` (default `3`, `0` disables
videos). Training, experiment candidates, and random-weight model smoke use
validation; only final evaluation uses test. The evaluation config names the
training run, and the evaluator loads that run's best checkpoint automatically.
The resolved config is saved with each evaluation run. Clips are saved
under the run/model's `predictions/correct/` and `predictions/incorrect/`
directories; the evaluation report links them to the exact checkpoint. Test
access stays in this evaluation step only.

Each prediction manifest records configured/actual counts, source paths, labels,
confidence, and partition. A missing category is reported as zero; finding it
can require scanning the full partition. These illustrative clips are not a
metric sample. The old `prediction_samples` field remains readable for historical
configs but does not control the new per-category exports. Model smoke predictions
are random-weight checks, not research evidence. Training no longer eagerly
caches the entire dataset, which would decode locked test clips before splitting.

```bash
.venv/bin/python -m src.evaluate \
  --config configs/evaluate/aad_evaluation_reference.json
```

To evaluate another training run, update `training_run_dir` in the JSON file.
Keep its input size and split aligned with that run; set `prediction_samples_per_category`
there to change how many prediction clips are saved. The command needs no other
arguments.

Output: `runs/evaluate/<run>/`. Evaluation reads the selected checkpoint and
never saves or overwrites training checkpoints.

Legacy multi-file AAD plan/config files remain for the older plan CLI and
notebook helper functions; the active AAD notebook does not reference them.
Use the three active profiles documented above. Standalone single-model training remains available through
`configs/train/aad_screening_reference.json` and is separate from those study
profiles.

The standalone profile retains an earlier exploratory run, not the latest search
winner: custom `[32, 16, 8]`, 16 frames at `64x64`, batch 32, learning rate `0.002`,
weight decay `0.0001`, and 128 epochs. That run's highest validation accuracy
was 46.9% at epoch 125; validation loss selected its checkpoint at epoch 122
(45.0% accuracy). This is a single exploratory result, not a controlled winner
or a promise of 90% accuracy; keep final selection validation-only and run the
predeclared comparisons before making claims.
