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
8. [ ] **Configuration integration (`017`, next).** Revise the stashed
   ticket-009 work around the stable model layer specification and reconnect the
   runner.
9. [ ] **Architecture shortlist (`018–019`).** Audit historical runs, predeclare
   a small model set, and confirm it under one protocol.
10. [ ] **Focused ablations (`020`).** Test depth, width, kernel size, input size,
   frame count, head design, augmentation, and regularisation one factor at a
   time—never as a Cartesian grid.
11. [ ] **Validation and handoff (`021–022`).** Confirm the frozen model on VDD,
   aggregate evidence, and export paper-ready tables and figures.
12. [ ] **Stateful streaming (`025`, later).** Carry custom ConvLSTM state across
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

Train the same reference without augmentation (starts a full training run):

```bash
.venv/bin/python -m src.train \
  --dataset_dir "datasets/abnormal-activities-dataset/abnormal-activities-dataset" \
  --runs_dir "runs" \
  --split_manifest "splits/abnormal-activities-dataset_seed42.json" \
  --seed 42 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --convlstm-layer 8 3 3 \
  --convlstm-layer 16 3 3 \
  --batch_size 32 \
  --weight_decay 0.0001 \
  --learning_rate 0.001 \
  --epochs 64 \
  --early_stopping_patience 10 \
  --sequence_length 16 \
  --height 32 \
  --width 32 \
  --num_workers 2 \
  --pin_memory
```

Output: `runs/train/<run>/`. Training reports validation results from the
restored lowest-loss checkpoint and does not access test clips.

Train with one online augmented view per sample (starts a full training run):

```bash
.venv/bin/python -m src.train \
  --dataset_dir "datasets/abnormal-activities-dataset/abnormal-activities-dataset" \
  --runs_dir "runs" \
  --split_manifest "splits/abnormal-activities-dataset_seed42.json" \
  --seed 42 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --convlstm-layer 8 3 3 \
  --convlstm-layer 16 3 3 \
  --batch_size 32 \
  --weight_decay 0.0001 \
  --learning_rate 0.001 \
  --epochs 64 \
  --early_stopping_patience 10 \
  --sequence_length 16 \
  --height 32 \
  --width 32 \
  --augment \
  --num_workers 2 \
  --pin_memory
```

Output: another unique `runs/train/<run>/`; test clips remain locked.

Inspect the comparison registry without loading AAD or allocating models:

```bash
.venv/bin/python -m src.experiments --list-models
```

Run the registered AAD comparisons (starts a full experiment):

```bash
.venv/bin/python -m src.experiments \
  --dataset_dir "datasets/abnormal-activities-dataset/abnormal-activities-dataset" \
  --runs_dir "runs" \
  --split_manifest "splits/abnormal-activities-dataset_seed42.json" \
  --seed 42 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --epochs 24 \
  --early_stopping_patience 10 \
  --batch_size 16 \
  --sequence_length 16 \
  --height 32 \
  --width 32 \
  --augment \
  --num_workers 2
```

Output: `runs/experiments/<run>/`, with one validation-selected checkpoint per
model. Ranking uses validation macro-F1, validation accuracy, then parameters;
it does not use test results.

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

The current training and experiment scripts remain exploratory until task 017
is complete. Random-weight smoke outputs are pipeline checks, not
research evidence.
