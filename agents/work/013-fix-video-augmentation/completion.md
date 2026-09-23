# Completion

- Status: Done
- Summary: Replaced expanded augmentation copies with one optional online view
  per training sample, applied one parameter set across each clip, and added
  copy-paste commands for the larger AAD dataset.
- Changes:
  - `implementation/src/dataset.py`: added the centralized conservative video
    policy, removed per-frame RNG resets, updated all dataset transforms to
    receive complete clips, and added timestamped clean/augmented MP4 previews.
  - `implementation/src/train.py` and `src/experiments.py`: replaced
    `--aug_copies` and `ConcatDataset` expansion with the opt-in `--augment`
    flag while leaving validation and test subsets clean.
  - Repository guidance: documented the online policy, added AAD commands for
    every executable module, completed ticket 013, and made ticket 014 next.
- Verification:
  - Modified Python files compiled and passed Black formatting checks.
  - Both runner help screens and parser checks exposed opt-in `--augment`,
    defaulted it off, and contained no `aug_copies` option.
  - A forced all-branches synthetic check retained `(5, 3, 24, 24)`, stayed in
    `[0, 1]`, kept repeated frames equal, and preserved subset length.
  - Twelve normal policy calls produced nine distinct online views while
    retaining temporal consistency within every clip.
  - The AAD preview loaded 1,069 videos and wrote two timestamped clean/augmented
    pairs; all four outputs decoded as 32-frame, `64×64` RGB videos.
  - Obsolete-pipeline search, scope review, and `git diff --check` passed;
    models, split ratios/indices, training logic, manuscript, checkpoints,
    existing outputs, and the ticket-009 stash were unchanged.
- Remaining issues: None.
