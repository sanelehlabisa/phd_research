# Completion

- Status: Done
- Summary: Simplified the ConvLSTM API to one reusable layer and two model
  classes, migrated active callers, and verified a real-data CPU smoke run.
- Changes:
  - Replaced legacy model classes and type aliases with `ConvLSTM`,
    `PaperConvLSTM`, `CustomConvLSTM`, and private `_ConvLSTMCell`.
  - Added explicit custom-layer CLI parsing, reconstructable checkpoint
    metadata, and clear rejection of incompatible legacy checkpoints.
  - Reduced the comparison registry to the paper model, one custom reference,
    and the existing three 3D-CNN models.
  - Reworked `model.py` to save two-model random-weight prediction videos,
    confusion matrices, and a JSON summary under `outputs/model_samples`.
  - Removed the tracked unittest directory and updated repository guidance.
- Verification:
  - Compiled `model.py`, `train.py`, `evaluate.py`, `experiments.py`, and
    `demo.py` successfully.
  - Verified sequence/final-state shapes, a two-layer custom stack, preserved
    spatial dimensions, parameter counting, CLI conversion, and new checkpoint
    reconstruction with direct Python checks.
  - Confirmed a legacy checkpoint fails with an explicit incompatibility error.
  - Confirmed `train.py`, `evaluate.py`, and `demo.py` help output includes the
    repeatable layer option and optional hidden-classifier width.
  - Ran the required CPU command on two VDD samples at `T=64` and `16x16`;
    both models returned `(1, 2)` logits and produced four MP4s, two confusion
    matrices, and a valid summary using `non-violent` and `violent` labels.
  - Searched active Python files for removed names and aliases; none remain.
  - `git diff --check` passed and the ticket-009 stash remained unchanged.
- Remaining issues: Comparison-baseline alignment remains for ticket 012.
