# ConvLSTM Abnormal Human Activity Recognition

Replication and extension of *"A New Approach for Abnormal Human Activities Recognition
Based on ConvLSTM Architecture"*, with multi-dataset support and structured evaluation.

## Model

CNN + LSTM architecture. A per-frame CNN backbone extracts spatial features;
an LSTM models temporal dynamics across the clip. Output: one of 14 activity classes
(13 anomalies + normal), plus a derived binary normal/abnormal verdict.

## Supported Datasets

The model targets **14 canonical classes** drawn from UCF-Crime, the standard benchmark
for surveillance-based abnormal activity recognition. All other datasets are mapped to
this vocabulary via `labels.json` aliases — no retraining required.

| Dataset | Classes | Type | Link |
|---|---|---|---|
| LoDVP Abnormal Activities | 11 | Staged surveillance anomalies | [link](https://kmikt.uniza.sk/ds/abnormal_activities.zip) |
| AirtLab Violence | 2 | Binary violent / non-violent | [link](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos) |
| UCF-Crime *(primary benchmark)* | 14 | Real-world surveillance crime | [link](https://www.crcv.ucf.edu/projects/real-world/) |
| UCF50 / UCF101 | 50 / 101 | General human actions (backbone eval) | [link](https://www.crcv.ucf.edu/data/UCF101.php) |
| Kinetics-400/700 | 400 / 700 | Large-scale action recognition (pretraining) | [link](https://github.com/cvdfoundation/kinetics-dataset) |

## Canonical Classes (14)

Sourced from UCF-Crime. All dataset folder names resolve to one of these via `labels.json`.

| ID | Class | Abnormal |
|---|---|---|
| 0 | normal | ✗ |
| 1 | abuse | ✓ |
| 2 | arrest | ✓ |
| 3 | arson | ✓ |
| 4 | assault | ✓ |
| 5 | burglary | ✓ |
| 6 | explosion | ✓ |
| 7 | fighting | ✓ |
| 8 | road accident | ✓ |
| 9 | robbery | ✓ |
| 10 | shooting | ✓ |
| 11 | shoplifting | ✓ |
| 12 | stealing | ✓ |
| 13 | vandalism | ✓ |

## Usage
```bash
# Train
python -m src.train --dataset_dir dataset_clean --epochs 64 --height 64 --width 64

# Evaluate
python -m src.evaluate --dataset_dir dataset_clean --model_dir models
```

## Journal experiment roadmap

This implementation supports the
[starter journal paper](https://github.com/sanelehlabisa/phd_work/tree/master/starter_journal).
Final paper results must record the exact code commit, configuration, split, and
seed that produced them.

### Current state

The repository can train the original ConvLSTM, custom-width ConvLSTM variants,
and three 3D-CNN baselines on multiple datasets. Earlier runs made **Proposed
Lightweight ConvLSTM (`custom_64_32_16_64`)** the provisional leading candidate,
but the result is not a confirmed optimum until the controlled experiments are
rerun.

### Problems to fix

- Make all randomness reproducible and save fixed, stratified, group-aware
  70:15:15 split manifests.
- Keep related camera views or source-video groups in one split.
- Correct accumulated metrics and restore the best validation-loss checkpoint
  before final evaluation.
- Keep test data locked during model selection.
- Give augmentation comparisons equal training budgets.
- Save complete run provenance, including configuration, seeds, splits,
  checkpoints, environment, histories, and the code commit.

### Experiment protocol

- Use fixed, stratified, group-aware 70:15:15 splits and keep related camera
  views or source videos together.
- Select models using validation results and keep test sets locked until the
  final comparisons are declared.
- Use seeds `42` and `2026` for confirmation runs.
- Use a fixed maximum epoch count, validation-loss early stopping, and the best
  validation-loss checkpoint.
- Save the configuration, split, seed, histories, checkpoint, environment, and
  code commit for every reported run.

Screen a bounded set of controlled ConvLSTM width and depth variants on AAD.
Select five custom models, then compare them with the original ConvLSTM and the
three 3D-CNN baselines on both datasets.

Run the focused ablations only on the selected proposed model: augmentation on
or off with equal budgets; weight decay `0` or `0.0001`; lower-priority Adam or
AdamW; and VDD training from scratch, frozen-feature transfer, or full-network
fine-tuning from AAD. The three input settings are 16 frames at 32x32, 32 frames
at 32x32, and 16 frames at 64x64. Control and record the initial learning rate
even when a scheduler is used.

### Experiment tickets

- [ ] `EXP-001-repair-experiment-runner` — **Next.** Repair splitting,
  reproducibility, metrics, checkpointing, augmentation fairness, and run
  manifests.
- [ ] `EXP-002-define-architecture-registry` — define controlled width and
  depth variants. **Blocked by EXP-001.**
- [ ] `EXP-003-screen-aad-architectures` — screen candidates and select five
  custom models using validation results. **Blocked by EXP-002.**
- [ ] `EXP-004-run-focused-ablations` — run the selected-model training, input,
  and transfer ablations. **Blocked by EXP-003.**
- [ ] `EXP-005-confirm-and-export-results` — run two-seed final comparisons and
  export versioned evidence for the paper. **Blocked by EXP-004.**

## References

1. LoDVP / ConvLSTM paper — Ullah et al.
2. AirtLab — Castaldi et al.
3. UCF-Crime — Sultani, Chen, Shah (CVPR 2018)
