# Paper 001 Implementation

Code for the [journal paper](../README.md). It now provides a small, explicit
family of ConvLSTM architectures for controlled comparison—not an unrestricted
hyperparameter grid.

## Model context

`ConvLSTM` returns either its full sequence or final hidden state.
`CustomConvLSTM` stacks explicit recurrent layers and uses a
resolution-independent adaptive-pooling head. `PaperConvLSTM` remains separate
because it preserves the published full-sequence, flattened classification
topology. Legacy architecture names and compatibility wrappers are removed.

The [2022 source paper](https://www.mdpi.com/1424-8220/22/8/2946) instead reports
50 RGB frames at `50×50`, keeps the full sequence through `ConvLSTM2D(64)` and
the second `Conv2D(16)`, then flattens `(50, 50, 50, 16)` into `Dense(256)`.
That produces over 512 million parameters and must remain a separate faithful
baseline rather than being silently made lightweight.

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

The current comparison runner uses `r3d_18`, `mc3_18`, and `r2plus1d_18` as
this study's practical 3D-CNN baselines. The source paper reports 3D ResNet-50,
3D ResNet-101, and 3D ResNet-152. These groups are not architecture-equivalent;
ticket 012 audits and labels them clearly before a paper comparison.

## Experiment protocol

- Fixed 70:15:15, stratified, group-aware split with a saved manifest.
- AAD for architecture screening and most ablations; VDD for final
  generalisation evaluation.
- Validation-only model and checkpoint selection; test data remains locked.
- Seeds `42` and `2026` for reported confirmation runs.
- Validation-loss early stopping with restoration of the selected checkpoint.
- Accuracy, macro precision/recall/F1, confusion matrix, trainable parameters,
  runtime, and uncertainty across confirmation seeds.
- One factor changed at a time with the split and training budget fixed.

## Implementation tasks

1. [x] **Architecture preparation (`010`).** Added the faithful paper model,
   sequence-returning ConvLSTM, explicit `(filters, kernel_size)` stacks, a
   lightweight adaptive-pooling head, and parameter counting.
2. [x] **Model API cleanup (`011`).** Kept `ConvLSTM`, `PaperConvLSTM`, and
   `CustomConvLSTM`; removed obsolete classes and migrated all active callers.
3. [ ] **Baseline audit and alignment (`012`, ready).** Verify the published
   ConvLSTM and distinguish the source paper's deeper comparisons from this
   study's three practical 18-layer 3D-CNN baselines.
4. [ ] **Reproducible data (`013`).** Seed the full pipeline and reuse a
   stratified, group-aware 70:15:15 split manifest.
5. [ ] **Valid selection and metrics (`014`).** Isolate full-partition metrics,
   restore the validation-selected checkpoint, and separate final test access.
6. [ ] **Run artifacts (`015`).** Save resolved configuration, split, seed,
   revision, history, parameters, metrics, checkpoint, confusion matrix, and
   runtime under a unique experiment ID.
7. [ ] **Configuration integration (`016`).** Revise the stashed ticket-009 work
   around the stable model layer specification and reconnect the runner.
8. [ ] **Architecture shortlist (`017–018`).** Audit historical runs, predeclare
   a small model set, and confirm it under one protocol.
9. [ ] **Focused ablations (`019`).** Test depth, width, kernel size, input size,
   frame count, head design, augmentation, and regularisation one factor at a
   time—never as a Cartesian grid.
10. [ ] **Validation and handoff (`020–021`).** Confirm the frozen model on VDD,
   aggregate evidence, and export paper-ready tables and figures.

Generated datasets, environments, checkpoints, and experiment runs remain
untracked. Do not change manuscript results until versioned evidence exists.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.model \
  --dataset-dir datasets/violence-detection-dataset/violence-detection-dataset
```

Run commands from this directory. The current training and experiment scripts
remain exploratory until tasks 012–016 are complete. The model smoke command
uses random weights and its outputs are pipeline checks, not research evidence.
