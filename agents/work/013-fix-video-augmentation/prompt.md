# Task Prompt

- Ticket: `013-fix-video-augmentation`
- Status: Done
- Aim: Replace copy-based training-set expansion with optional, temporally
  consistent online video augmentation and document fast copy-paste commands
  that use the larger AAD dataset.
- Scope:
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/dataset.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/train.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiments.py`
  - root, paper, and implementation guidance affected by commands or task status
- Changes:
  - Replace `--aug_copies` with an opt-in `--augment` flag in both training
    runners. Without the flag, return clean training samples; with it, generate
    one fresh augmented view whenever a training sample is loaded.
  - Keep the training subset length equal to the number of original training
    samples. Remove the clean-plus-copies `ConcatDataset` construction and do
    not save augmented dataset copies to disk.
  - Centralize one readable video-augmentation policy in `dataset.py` so
    `train.py` and `experiments.py` cannot define different transforms.
  - Sample each random decision and parameter set once per clip, then apply it
    identically to every frame. Do not call `torch.manual_seed` inside dataset
    access or otherwise reset global random state to synchronize frames.
  - Use one conservative surveillance-video policy: horizontal flip at `0.5`;
    affine at `0.3` with at most 8-degree rotation, 5% translation, and
    `0.95–1.05` scale; brightness/contrast adjustment up to `0.15` at `0.3`;
    and light blur at `0.1`. Remove vertical flip, perspective, grayscale,
    random erasing, sharpness adjustment, and the current strong colour jitter.
  - Keep validation and test samples clean in every mode. Make terminal output
    state whether augmentation is enabled and confirm that the training-set
    length is unchanged.
  - Extend the dataset preview command to save clearly named clean and augmented
    MP4 pairs so temporal consistency and transform strength can be inspected.
  - Add concise copy-paste commands to the implementation README for every
    executable module: dataset preview, preprocessing dry run, model smoke,
    training with and without augmentation, baseline registry inspection,
    experiment running, and evaluation. Use `.venv/bin/python`, the larger AAD
    path `datasets/abnormal-activities-dataset/abnormal-activities-dataset`, and
    label commands that train or write outputs.
  - Update task status and concise repository guidance when complete.
- Acceptance criteria:
  - `--augment` is the only training augmentation control; `--aug_copies` and
    copy-based expansion are absent from active code and commands.
  - Enabling augmentation does not change the training subset length. Each load
    produces at most one online view of an original sample.
  - Spatial and photometric parameters are consistent across all frames of one
    clip, and augmentation code does not reset global RNG state.
  - Training data is clean when `--augment` is absent; validation and test data
    are always clean.
  - `train.py` and `experiments.py` use the same centralized policy and report
    its enabled/disabled state.
  - The preview command writes matched clean and augmented, viewable video files
    without changing the source dataset.
  - The implementation README contains valid, clearly labelled copy-paste
    commands for all executable scripts and consistently uses the larger AAD
    dataset path.
  - Active Python files compile and `git diff --check` reports no errors.
- Out of scope:
  - Training or evaluating models, comparing augmentation performance, or
    changing reported results.
  - Creating persistent augmented datasets or multiple copies per sample.
  - Implementing the fixed split manifest, complete worker seeding, or global
    determinism; ticket 014 owns those changes.
  - Changing models, losses, optimizers, schedulers, checkpoint selection,
    metrics, manuscript text, or existing generated artifacts.
- Open questions: None.
- Verification:
  - Compile every modified Python file.
  - Check both runner help screens and confirm `--augment` is present while
    `--aug_copies` is rejected or absent.
  - Use a synthetic repeated-frame clip to confirm one sampled transform is
    applied consistently across time and that the output shape/range is valid.
  - Verify clean and augmented wrappers have exactly the base-subset length and
    that validation/test construction never receives the augmentation policy.
  - Run the AAD preview command on a small sample and inspect the paired output
    names; do not start model training.
  - Search active code and guidance for duplicated transform pipelines,
    `aug_copies`, augmentation `ConcatDataset`, and per-frame reseeding.
  - Confirm models, splits, training logic, manuscript, checkpoints, existing
    outputs, and the ticket-009 stash are unchanged.
  - Run `git diff --check` and review the final diff for unrelated changes.

## Execution Prompt

Execute ticket `013-fix-video-augmentation` exactly as written in
`agents/work/013-fix-video-augmentation/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
