# Completion

- Status: Done
- Summary: Audited the published ConvLSTM against the primary paper and Keras
  defaults, aligned its PyTorch implementation, and made the five approved
  comparison roles inspectable without data or training.
- Changes:
  - `implementation/src/model.py`: matched the paper's full-sequence topology,
    Keras gate behaviour and initialization, Batch Normalization settings, and
    recorded parameter categories while preserving custom-model defaults.
  - `implementation/src/experiments.py`: added a five-model registry, explicit
    comparison roles, sequential model construction, and `--list-models`.
  - Root, paper, implementation, and agent guidance: documented the evidence
    boundary, baseline distinction, exact counts, inspection command, and next
    task.
- Verification:
  - The default paper model produced `(1, 11)` logits on a meta input shaped
    `(1, 50, 3, 50, 50)` and matched the source counts: `512,197,467` trainable,
    `128` non-trainable state values, and `512,197,595` total.
  - Reduced CPU forwards passed for `PaperConvLSTM`, `CustomConvLSTM`, `r3d_18`,
    `mc3_18`, and `r2plus1d_18`, with each model allocated sequentially.
  - `.venv/bin/python -m src.experiments --list-models` listed exactly the five
    approved names and roles without dataset access or training.
  - `python -m py_compile src/model.py src/experiments.py` passed.
  - Source-claim search, checkpoint compatibility check, scope review, and
    `git diff --check` passed; data, training, augmentation, manuscript,
    checkpoints, generated results, and the ticket-009 stash were unchanged.
- Remaining issues: None.
