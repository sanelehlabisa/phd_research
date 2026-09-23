# Task Prompt

- Ticket: `012-audit-and-align-baselines`
- Status: Done
- Aim: Verify the published ConvLSTM baseline against its primary paper and
  clearly separate it from this study's three practical 18-layer 3D-CNN
  baselines before any comparison training.
- Scope:
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/model.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiments.py`
  - root, paper, and implementation guidance affected by baseline names,
    commands, or task status
  - primary source: <https://www.mdpi.com/1424-8220/22/8/2946>
- Changes:
  - Audit `PaperConvLSTM` against the source paper's architecture description,
    algorithm, figures, framework defaults, input shape, and disclosed training
    setup. Check temporal return mode, padding, activations, normalization,
    dropout placement, flattening, and dense layers instead of assuming them.
  - Correct `PaperConvLSTM` where the primary source supports a different
    implementation. Preserve the published default input of 50 RGB frames at
    `50x50` and keep it separate from `CustomConvLSTM`.
  - Document concise implementation notes for details that the source does not
    report. Call the model a faithful implementation only to the extent
    supported by the paper; do not invent missing settings.
  - Verify the default output shape and trainable-parameter count without
    performing a full training run or allocating avoidable large tensors.
  - Keep `r3d_18`, `mc3_18`, and `r2plus1d_18` as this study's clearly named
    practical 3D-CNN baselines. Keep `weights=None` so this ticket does not
    silently introduce pretraining.
  - Distinguish those 18-layer study baselines from the source paper's reported
    3D ResNet-50/101/152 comparisons. Do not describe either group as an
    architecture-equivalent reproduction of the other.
  - Refactor the experiment registry only as needed to expose the faithful
    paper baseline, one explicitly named custom reference, and the three
    approved 3D-CNN baselines with unambiguous names and roles.
  - Add a list-only or equivalent dry inspection command that reports the
    registered model names and roles without loading a dataset, allocating all
    models, or starting training.
  - Keep repository guidance concise and update the next-task status when the
    audit is complete.
- Acceptance criteria:
  - Every implemented `PaperConvLSTM` stage is traceable to the primary paper or
    is marked as an explicit implementation choice when the source is silent.
  - `PaperConvLSTM` retains defaults of 50 RGB frames at `50x50`, returns logits
    shaped `(B, num_classes)`, and has a recorded calculated parameter count.
  - The audit resolves whether the recurrent layer returns the full sequence or
    final state before the following convolution; the decision is supported by
    the paper and the original framework's documented behaviour.
  - The experiment registry contains `PaperConvLSTM`, one `CustomConvLSTM`
    reference, `r3d_18`, `mc3_18`, and `r2plus1d_18`, with no 3D ResNet-50/101/152
    implementation or equivalence claim.
  - Registry labels distinguish the original paper baseline, this study's
    custom reference, and this study's practical 3D-CNN baselines.
  - The dry inspection command completes without a dataset and without starting
    training. Reduced-input forward smoke checks for all model families pass on
    CPU where practical.
  - Active Python files compile and `git diff --check` reports no errors.
- Out of scope:
  - Training or evaluating any model.
  - Implementing 3D ResNet-50, 3D ResNet-101, or 3D ResNet-152.
  - Selecting custom depth, width, kernel, or filter-position variants.
  - Changing data splits, augmentation, seeds, early stopping, metrics,
    checkpoint policy, or run artifacts; the following tickets own those items.
  - Updating manuscript results, tables, conclusions, or performance claims.
  - Modifying or deleting existing checkpoints, curves, or experiment outputs.
- Open questions: None.
- Verification:
  - Record the primary-paper evidence used for each material architecture
    decision, including any reliance on original framework defaults.
  - Compile `model.py` and `experiments.py`.
  - Run the registry's list-only command and confirm the five approved entries
    and their roles without dataset access or training.
  - Run direct CPU shape and parameter-count checks for `PaperConvLSTM`,
    `CustomConvLSTM`, and the three 3D-CNN wrappers using safe inputs and
    sequential allocation.
  - Search active code and guidance for claims that the 18-layer models are the
    source paper's 50/101/152 models.
  - Confirm training, datasets, augmentation, manuscript results, checkpoints,
    generated curves, and the ticket-009 stash are unchanged.
  - Run `git diff --check` and review the final diff for unrelated changes.

## Execution Prompt

Execute ticket `012-audit-and-align-baselines` exactly as written in
`agents/work/012-audit-and-align-baselines/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
