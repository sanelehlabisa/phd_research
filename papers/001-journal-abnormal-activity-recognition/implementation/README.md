# Paper 001 Implementation

Research code for the
[journal paper on abnormal activity recognition](../README.md). It implements
the original and custom ConvLSTM models plus three 3D-CNN comparison models.
The paper currently uses the Abnormal Activities Dataset (AAD) and AIRTLab
Violence Detection Dataset (VDD).

## Setup and commands

Create an environment and install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The current scripts expose their configurations through command-line options:

```bash
python -m src.experiments --dataset_dir /path/to/dataset
python -m src.train --dataset_dir /path/to/dataset --model_dir models
python -m src.evaluate --dataset_dir /path/to/dataset --model_dir models
```

Run each command from this `implementation/` directory. Datasets, environments,
checkpoints, and generated outputs remain local and untracked.

## Current validity problems

- Randomness and dataset splits are not fully reproducible.
- Related views or source videos may cross splits.
- Model selection can expose test results too early.
- Metrics and best-checkpoint restoration require correction.
- Augmentation changes the training budget.
- Saved results lack complete configuration and provenance.

## Agreed experiment protocol

- Use fixed, stratified, group-aware 70:15:15 splits.
- Select with validation data and lock test data until final comparisons.
- Use seeds `42` and `2026` for confirmation runs.
- Use validation-loss early stopping and restore the best checkpoint.
- Save the split, seed, configuration, histories, checkpoint, environment, and
  code commit for every reported run.
- Screen controlled ConvLSTM width and depth variants on AAD, then confirm five
  custom models, the original ConvLSTM, and three 3D-CNNs on both datasets.
- Run training, input-size, and transfer ablations only on the selected model.

## Tickets

- [ ] `009-transfer-local-experiment-work` — **Next**
- [ ] `010-repair-experiment-runner` — blocked by 009
- [ ] `011-define-architecture-registry` — blocked by 010
- [ ] `012-screen-aad-architectures` — blocked by 011
- [ ] `013-run-focused-ablations` — blocked by 012
- [ ] `014-confirm-and-export-results` — blocked by 013

Paper rewriting starts only after ticket 014 produces versioned evidence.
