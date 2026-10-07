# Paper 001 Implementation

Code for the [journal paper](../README.md). It now provides a small, explicit
family of ConvLSTM architectures for controlled comparison—not an unrestricted
hyperparameter grid.

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

The comparison runner uses `r3d_18`, `mc3_18`, and `r2plus1d_18` with
`weights=None` as this study's practical 18-layer 3D-CNN baselines. The source
paper reports 3D ResNet-50, 3D ResNet-101, and 3D ResNet-152. These groups are
not architecture-equivalent reproductions.

Inspect the five approved entries without a dataset or model allocation:

```bash
.venv/bin/python -m src.experiments --list-models
```

List the eleven custom AAD candidates without loading data or allocating models:

```bash
.venv/bin/python -m src.experiments \
  --config configs/aad_architecture_screen_reference.json \
  --candidates-config configs/aad_architecture_candidates.json \
  --list-models
```

Inspect every controlled stage, model, protocol, factor, run count, and command:

```bash
.venv/bin/python -m src.experiments \
  --plan-config configs/aad_controlled_experiment_plan.json \
  --list-plan
```

Start the full architecture screen only after reviewing both safe listings:

```bash
# Expensive: trains eleven candidates on AAD.
.venv/bin/python -m src.experiments \
  --plan-config configs/aad_controlled_experiment_plan.json \
  --run-plan-stage architecture-screen
```

An interrupted screen leaves evidence only for models whose training completed;
do not treat an incomplete ranking as the final shortlist. Historical local
outputs lack the current complete protocol and remain non-comparable
exploratory evidence.

## Experiment protocol

- Fixed 70:15:15 stratified clip split using the committed
  [AAD manifest](splits/abnormal-activities-dataset_seed42.json): 748 training,
  160 validation, and 161 test clips.
- AAD for architecture screening and most ablations; VDD for final
  generalisation evaluation.
- `--augment` applies no transform or one randomly selected, clip-consistent
  transform per training sample; it never combines transforms or changes
  dataset length. Validation and test remain clean.
- Validation-only model and checkpoint selection; test data remains locked.
- Seeds `42` and `2026` for reported confirmation runs; both reuse the same
  split created with split seed `42`.
- Validation-loss early stopping with patience `10` and restoration of the
  selected checkpoint.
- Sample-weighted loss plus full-partition accuracy and macro
  precision/recall/F1, confusion matrix, trainable parameters, runtime, and
  uncertainty across confirmation seeds.
- One factor changed at a time with the split and training budget fixed.

## Modular Colab notebooks

Notebook-only helpers and the Colab bootstrap live in `notebooks/utils/`;
reusable model, dataset, training, evaluation and CLI modules remain in `src/`.
Helper filenames are concise (`config.py`, `data.py`, `workflows.py`); their
directory already identifies them as notebook support.

If an old export reports missing `src/notebook_data.py` or similar files, reopen
and re-export the current notebook from master: the helpers moved to
`notebooks/utils/`. Pulling inside an old setup cell updates repository files,
not that cell's source. All current notebooks use `notebooks.utils` imports.

[Ticket 047](../../../agents/work/047-larger-colab-training/prompt.md) is ready
for approval for a larger notebook-03-only model/input profile. Its 200-epoch,
eight-hour budget is planned, not implemented; the current defaults below remain.

Integration [ticket 042](../../../agents/work/042-merge-expanded-kinetics-suite/prompt.md)
combines Kinetics-600 (>2,000 unique videos before splitting) with the expanded
eight-hour suite. Real Colab inventory/training remains pending.

Open one notebook and choose **Run All** on a Colab GPU. All four use
`SELECTED_DIAGNOSTIC_DATASET = "kinetics600-subset"` in
[shared configuration](notebooks/utils/config.py). Preparation downloads only the
five Kinetics-600 training archives (about 3 GB compressed), extracts them
safely, prints actual per-class/total counts and stops unless it finds over
2,000 unique clips. The exact extracted total remains to be observed on Colab.
This is an exploratory five-activity diagnostic, not surveillance-paper evidence.
Change the shared setting to `"vdd"` for the unchanged VDD labels or
`"kinetics-subset"` for the previous 87-clip Kinetics-400 copy. Existing saved
notebook outputs describe earlier VDD/Kinetics-400 runs.

| Notebook | End-to-end workflow |
|---|---|
| [01 Dataset](notebooks/01_dataset_setup.ipynb) | Split summary; the same training video at native FPS, sampled FPS, then augmented |
| [02 Model](notebooks/02_model_inspection.ipynb) | Architecture, parameter count, one random-weight prediction, playable labelled video and probabilities |
| [03 Training](notebooks/03_train_model.ipynb) | Train/validate one model, live epoch curves, restore the selected checkpoint, show five validation predictions |
| [04 Experiments](notebooks/04_run_experiments.ipynb) | Visible candidate screen, validation winner, longer/finer retraining, freeze, final test metrics then test videos |

Notebook 04 creates one ZIP for its current suite and triggers a Colab browser
download after completion or a handled time-budget timeout. It includes run and
per-model configs, histories, metrics, plots, checkpoints, and generated
predictions; source datasets are excluded. Original run folders remain under
`runs/`. The download needs the Colab notebook session to reach its final cell.

The exported one-cell experiment workflow checks actual NumPy and notebook
imports before loading project code. If the pinned NumPy wheel is inconsistent,
setup repairs it and asks for one kernel restart. Dataset setup omits per-archive
cache-reuse lines while retaining download progress and the final unique-video count.

Notebooks 03/04 have fresh filenames to work around VS Code's logged
`notebook controller is DISPOSED` error: copies ran while the old paths did not.
Their cells, metadata and saved outputs are preserved. Close the old tabs, open
the files linked above and select the working Colab GPU kernel for each.
This is a filename-based workaround; local tests cannot verify the editor's Run button.

Notebooks 01–03 retain 16 frames at 16 FPS, batch 8, seed 42 and 32×32 input;
notebook 03 trains for 128 epochs. Notebook 04 has separate `SUITE_*` settings:
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
macro-F1, accuracy, parameter count, then name. Batch progress is printed,
loss/accuracy plots update each epoch, and artifacts go under ignored `runs/`.

These are **exploratory diagnostics**, not the paper's controlled
AAD protocol or evidence of cross-dataset generalisation. Notebook 04 follows
the selected diagnostic dataset. The
[original reference notebook](notebooks/aad_experiment_workflow.ipynb), controlled
AAD configuration and `src.experiments` CLI remain unchanged.

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

`CLASSES_OF_INTEREST` contains the existing AAD/VDD labels and these exact
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
Controlled AAD and the original reference notebook remain unchanged.

This is a tiny learnability diagnostic, not the full official Kinetics release
or surveillance evidence; held-out estimates are noisy. The broader audit in
ticket 036 remains incomplete: metadata does not establish clip quality,
source independence or absence of near duplicates.

## Implementation tasks

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
12. [ ] **Optional cloud expansion (`021`, pre-screen decision).** Before
    screening, add the
    entire predeclared five-model block when a timing check shows enough runtime.
13. [x] **Notebook foundation (`022`).** Added the Colab VS Code runtime,
    committed-split, class-balance, training-sample, and augmentation walkthrough.
14. [x] **Model inspection (`023`).** Added inline clean and augmented videos,
    manifest-driven candidate and role tables, a readable lightweight reference,
    and one labelled random-weight GPU prediction with cleanup.
15. [x] **Experiment notebook (`024`).** Added bounded learning checks, guarded
    single-stage execution, and validation-only result displays.
16. [ ] **Controlled AAD runs (`025`).** Run the frozen screen, baselines,
    published topology, and ablations; select from validation only.
17. [ ] **Selected-model training (`026`).** Confirm the selected architecture
    and validated input/regularisation choices across both seeds.
18. [ ] **Final AAD evaluation (`027`).** Open test once for the frozen
    checkpoint and export complete metrics plus prediction examples.
19. [ ] **VDD and evidence (`028–029`).** Validate generalisation, then export
    paper-ready aggregate tables and figures.
20. [ ] **Manuscript alignment (`030–031`).** Rewrite results and align all
    affected claims only from versioned evidence.
21. [ ] **Kinetics transfer decision (`032`, conditional).** Consider an exact
    pretraining/transfer protocol only if frozen AAD/VDD evidence remains weak.
22. [ ] **Stateful streaming (`033`, later).** Carry custom ConvLSTM state across
    video chunks and predict after each chunk without altering `PaperConvLSTM`.

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
  --num-samples 2 \
  --convlstm-layer 8 3 3 \
  --convlstm-layer 16 3 3
```

Output: `runs/model/<run>/`.

Inspect the AAD screening reference without loading data or creating a run:

```bash
.venv/bin/python -m src.train \
  --config configs/aad_screening_reference.json \
  --print-config
```

Train the configured custom reference. This starts expensive training:

```bash
.venv/bin/python -m src.train \
  --config configs/aad_screening_reference.json
```

Output: `runs/train/<run>/`. Training restores the lowest-validation-loss
checkpoint and never opens the test split.

Run practical-baseline confirmation only after replacing `REFERENCE` with the
validation-selected custom candidate. This runs both confirmation seeds and
starts expensive training:

```bash
.venv/bin/python -m src.experiments \
  --plan-config configs/aad_controlled_experiment_plan.json \
  --run-plan-stage baseline-confirmation \
  --reference-candidate REFERENCE
```

Run the focused weight-decay, augmentation, spatial-size, and sequence-length
comparisons after freezing the same reference. Each non-reference run changes
one factor and both seeds are run:

```bash
.venv/bin/python -m src.experiments \
  --plan-config configs/aad_controlled_experiment_plan.json \
  --run-plan-stage focused-ablations \
  --reference-candidate REFERENCE
```

The audited published topology is a separate, very expensive native-input run:

```bash
.venv/bin/python -m src.experiments \
  --plan-config configs/aad_controlled_experiment_plan.json \
  --run-plan-stage published-topology
```

It uses 50 frames at `50x50` and is not a one-factor comparison with the
lightweight models. If resources are insufficient, record that limitation; do
not reduce the model and still call it faithful. All comparison outputs live
under `runs/experiments/<run>/`, rank validation evidence only, and keep test
access locked.

After freezing a configuration, deliberately evaluate its validation-selected
checkpoint on the AAD test split (writes metrics, a confusion matrix, and
prediction clips):

```bash
.venv/bin/python -m src.evaluate \
  --dataset_dir "datasets/abnormal-activities-dataset/abnormal-activities-dataset" \
  --checkpoint_path "runs/train/<run>/checkpoints/best_model.pth" \
  --runs_dir "runs" \
  --split_manifest "splits/abnormal-activities-dataset_seed42.json" \
  --seed 42 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --convlstm-layer 8 3 3 \
  --convlstm-layer 16 3 3 \
  --batch_size 32 \
  --sequence_length 16 \
  --height 32 \
  --width 32 \
  --num_workers 2 \
  --pin_memory \
  --num_samples 8
```

Output: `runs/evaluate/<run>/`. The checkpoint is read as input and is never
copied or overwritten by evaluation.

The AAD plan names `custom_depth_8_8_8` only as the preliminary reference used
to make the plan executable; replace it only with validation-screen evidence
before confirmation or ablation. The 24-epoch screen and 64-epoch confirmation
are compute stages, not an epoch ablation. Random-weight inference and the
512-step learning sanity check remain pipeline checks, not research evidence.

The controlled AAD screen uses `configs/aad_architecture_screen_reference.json`:
eleven candidates, 16 frames at `32x32`, and a 24-epoch maximum. The separate
`aad_screening_reference.json` remains a standalone training profile and is not
read by the controlled plan. Ticket 026 reserves up to 64 epochs for training
the validation-selected model.

The standalone profile now reproduces the strongest local exploratory run so
far: custom `[32, 16, 8]`, 16 frames at `64x64`, batch 32, learning rate `0.002`,
weight decay `0.0001`, and 128 epochs. That run's highest validation accuracy
was 46.9% at epoch 125; validation loss selected its checkpoint at epoch 122
(45.0% accuracy). This is a single exploratory result, not a controlled winner
or a promise of 90% accuracy; keep final selection validation-only and run the
predeclared comparisons before making claims.
