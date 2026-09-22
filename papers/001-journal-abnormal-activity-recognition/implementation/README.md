# Paper 001 Implementation

Code for the [journal paper](../README.md), including custom ConvLSTM variants,
the original ConvLSTM baseline, and three 3D-CNN comparison models. The current
priority is a reproducible, focused ablation study—not a large hyperparameter
grid.

## Current state

The runner already accepts dataset, epoch, batch, sequence-length, resolution,
learning-rate, weight-decay, and augmentation-copy options. Model variants are
still hard-coded, dropout and optimiser choices are fixed, and depth is not
configurable. The current split is random rather than stratified and
group-aware; only its generator is seeded. The grid runner also evaluates every
variant on the test set, does not restore the best validation checkpoint, and
saves incomplete run provenance. These issues must be fixed before new results
are treated as paper evidence.

## Intended protocol

- Fixed 70:15:15, stratified, group-aware split with a saved manifest.
- AAD for architecture screening and most ablations; VDD for final
  generalisation evaluation.
- Validation-only model and checkpoint selection; test data remains locked.
- Seeds `42` and `2026` for reported confirmation runs.
- Validation-loss early stopping with restoration of the selected checkpoint.
- Accuracy plus macro precision, recall, F1, confusion matrix, trainable
  parameters, and training time.
- One factor changed at a time with the split and optimisation budget fixed.

## Preparation tasks

1. [ ] **Experiment configuration (`009`, next).** Introduce one typed,
   validated configuration covering dataset and split identifiers; seed; image
   size and frame count; convolution, ConvLSTM, and dense widths; supported
   depth; dropout; augmentation; optimiser; learning rate; weight decay;
   scheduler; batch size; epochs; and early stopping. Load it from a JSON file
   with explicit CLI overrides and save the fully resolved form. Preserve the
   current defaults and do not create one script per experiment.
2. [ ] **Reproducible data (`010`).** Seed Python, NumPy, PyTorch, CUDA, data
   loading, and augmentation. Generate and reuse a stratified, group-aware
   70:15:15 split manifest so related source videos or camera views never cross
   splits. Evaluation must consume the manifest instead of recreating a split.
3. [ ] **Valid selection and metrics (`011`).** Compute metrics over each full
   partition with isolated state, select by validation loss, restore the chosen
   checkpoint, and make final test evaluation an explicit separate action.
   Augmentation comparisons must use an equal, documented training budget.
4. [ ] **Run artifacts (`012`).** Give every run a unique ID and save its
   resolved configuration, split reference, seed and determinism settings, code
   revision/environment, histories, parameter count, validation metrics,
   checkpoint, final test metrics when authorised, confusion matrix, and run
   time. Add focused tests and a CPU smoke run.
5. [ ] **Configurable factors (`013`).** Connect the shared configuration to
   smaller/selected/larger widths, dense width, one justified extra-convolution
   variant, input resolution, frame count, augmentation, weight decay, and
   dropout. First assess stacked ConvLSTM feasibility because the current layer
   returns only its final state. Do not build a Cartesian grid.
6. [ ] **Historical shortlist (`014`).** Catalogue the 30+ runs with their known
   configurations and comparability limits. Shortlist 3–5 smaller, selected,
   and larger ConvLSTM candidates using validation performance and parameter
   count; never fill missing metadata by assumption.

## Experiment and reporting tasks

7. [ ] **Controlled architecture comparison (`015`).** Re-run the shortlist,
   original ConvLSTM, and three 3D-CNNs using the same AAD split and protocol.
   Choose the reference model using validation performance and parameter cost.
8. [ ] **Focused reference ablations (`016`).** Against the frozen reference,
   test resolution `{32, 64, 128}`, frame count `{8, 16, 32}`,
   smaller/selected/larger ConvLSTM width, smaller/selected/larger dense width,
   one justified extra-convolution variant, augmentation off/on, and a small
   weight-decay comparison. Test dropout or scheduling only when a stated
   hypothesis justifies it.
9. [ ] **Second-dataset validation (`017`).** Run the frozen selected
   configuration on VDD and document dataset-specific preprocessing or protocol
   differences instead of silently changing them.
10. [ ] **Aggregation and paper handoff (`018`).** Produce JSON/CSV summaries
    and plots with experiment ID, dataset/split, seed, architecture/input/
    training configuration, parameter count, best validation epoch and metrics,
    final test metrics, run time, checkpoint, and code revision. Flag incomplete
    or incomparable runs.

Each run should live under a stable path such as:

```text
experiments/ablation/<experiment-id>/
├── config.json
├── split.json
├── history.json
├── metrics.json
├── confusion_matrix.png
└── checkpoint.pt
```

Generated runs, datasets, environments, and checkpoints remain untracked. A
tracked aggregate may be added later only when the paper task explicitly
approves it.

## Setup and commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.experiments --dataset_dir /path/to/dataset
```

Run commands from this directory. Tasks 009–013 will update the CLI and examples;
until then, treat the existing runner as exploratory.
