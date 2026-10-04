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
  --config configs/aad_screening_reference.json \
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
- `--augment` applies one fresh, clip-consistent online view per training
  sample without changing dataset length; validation and test remain clean.
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

Open one notebook and choose **Run All** on a Colab GPU. All four use
`SELECTED_DIAGNOSTIC_DATASET = "vdd"` in [shared configuration](src/notebook_config.py),
with `sanelehlabisa/violence-detection-dataset` and exact `non-violent` /
`violent` class folders. There are no execution toggles or run-directory placeholders.

| Notebook | End-to-end workflow |
|---|---|
| [01 Dataset](notebooks/01_dataset_setup.ipynb) | Split summary; the same training video at native FPS, sampled FPS, then augmented |
| [02 Model](notebooks/02_model_inspection.ipynb) | Architecture, parameter count, one random-weight prediction, playable labelled video and probabilities |
| [03 Training](notebooks/03_model_training.ipynb) | Train/validate one model, live epoch curves, restore the selected checkpoint, show five validation predictions |
| [04 Experiments](notebooks/04_controlled_experiments.ipynb) | Visible candidate screen, validation winner, longer/finer retraining, freeze, final test metrics then test videos |

Defaults: 16 frames at 16 FPS, batch 8, seed 42, 32×32 input. Notebook 03 trains
for 12 epochs. Notebook 04 compares `8`, `8-16`, and `8-8-8` stacks for eight
epochs each, then retrains the winning architecture **from scratch** for 24
epochs at 64×64. Budgets and candidates live beside the dataset selection.
Checkpoints use minimum validation loss; candidate ranking uses validation
macro-F1, accuracy, parameter count, then name. Batch progress is printed,
loss/accuracy plots update each epoch, and artifacts go under ignored `runs/`.

These are **single-seed exploratory diagnostics**, not the paper's controlled
AAD protocol or evidence of cross-dataset generalisation. Despite its retained
filename, notebook 04 now follows the selected diagnostic dataset. The
[original reference notebook](notebooks/aad_experiment_workflow.ipynb), controlled
AAD configuration and `src.experiments` CLI remain unchanged.

Preparation inventories filenames/classes and reuses the stratified 70:15:15
manifest without opening test videos. The diagnostic loader decodes only the
first input window, samples by timestamps and keeps a bounded 64 MiB cache.
Corrupt clips fail explicitly instead of substituting another split's sample.
This preprocessing is recorded separately from the legacy AAD loader.
The split is clip-level; subject/source independence and near-duplicate absence
are **not** established.

Notebook 04 opens test automatically only after all candidates finish and a
compatible longer-trained checkpoint is frozen. Metrics precede example videos;
rerunning its final cell displays the saved report. Do not tune or rerank from
test feedback. Notebooks 01–03 never evaluate test. No accuracy threshold is promised.

Every setup cell safely fast-forwards its checkout, verifies the required source
files and prints the code revision before installing `requirements.txt` and
reloading modules. Git preserves unrelated edits and refuses updates that would
overwrite conflicting edits; no reset, clean or automatic stash is used.
If dependency versions change, restart the runtime once and
rerun from the top. Updated local code must be pushed/synced before Colab can use it.
`src.notebook_data` is a repository module, not a pip package: a kernel restart
cannot download source files that have not been committed and pushed.
The previous notebooks and saved outputs are preserved locally at
`runs/notebook-backups/pre-039/`; revised cells start without stale outputs.

### Later datasets

All four notebooks follow the same selection. Kinetics remains disabled until
ticket 036 identifies the **actual variant and exact accepted class directories**.
Configure that subset in `DIAGNOSTIC_DATASETS`, then change the one selection.
Class filtering limits training inputs, but a full Kaggle download may still be
large. Do not silently use every class or call an unaudited copy official Kinetics.

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
