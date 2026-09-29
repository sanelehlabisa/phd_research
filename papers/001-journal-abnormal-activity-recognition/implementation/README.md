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

## Colab VS Code notebook

Use [the AAD workflow](notebooks/aad_experiment_workflow.ipynb) from VS Code
connected to an A100 through the official Google Colab extension.
Tickets 022–023 provide the runtime, AAD walkthrough, playable augmentation
preview and model tables. Ticket 024 begins with three visual untrained
inference checks, then adds guarded experiment execution without duplicating research logic.

1. Connect the notebook to the named Colab A100 runtime and run its GPU check.
2. Run the Python setup cell. It clones the code into `/content/phd_research`
   once or fast-forwards an existing clean checkout, reloads local `src`
   modules, installs missing packages quietly, and reuses cached AAD or
   downloads the public KaggleHub dataset when a fresh runtime has no copy.
3. Inspect the split summary, class balance, sample metadata, a side-by-side
   native/model-ready comparison, and the augmented model input.
4. Inspect all custom candidates and comparison roles, then run three labelled
   random-weight A100 checks with tensor shapes and playable prediction cards.

5. Optionally enable the 512-step learning sanity check (about 11 epochs). It
   uses the controlled reference optimizer and loss settings, saves a checkpoint
   and JSON history, plots training and validation loss, and shows three clean
   validation-video predictions while keeping the test split locked.

The notebook uses normal Python cells for interactive tables and embedded video.
Its bounded training section is a learning sanity check, not evidence for
ranking or selecting a model. The controlled experiment runner remains the
source of research evidence, and no notebook inspection opens test clips.

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
15. [ ] **Experiment notebook (`024`, next).** Add the bounded training check,
    guarded plan-stage execution, and validation-only result displays.
16. [ ] **Architecture and baseline runs.** Execute the frozen plan through the
    completed notebook and retain complete validation evidence.
17. [ ] **Validation and handoff (`025–026`).** Confirm the frozen model on VDD,
    aggregate evidence, and export paper-ready tables and figures.
18. [ ] **Stateful streaming (`029`, later).** Carry custom ConvLSTM state across
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
are compute stages, not an epoch ablation. Random-weight smoke outputs remain
pipeline checks, not research evidence.
