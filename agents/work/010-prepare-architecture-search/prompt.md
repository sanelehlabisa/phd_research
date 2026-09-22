# Task Prompt

- Ticket: `010-prepare-architecture-search`
- Status: Done
- Aim: Add a faithful implementation of the paper's proposed ConvLSTM and a
  configurable stacked ConvLSTM family for controlled depth, width, and kernel
  experiments without changing the training pipeline.
- Research source:
  - Vršková et al., “A New Approach for Abnormal Human Activities Recognition
    Based on ConvLSTM Architecture,” *Sensors*, 2022.
  - https://www.mdpi.com/1424-8220/22/8/2946
  - The paper's Table 1 keeps 50 time steps through the second convolution and
    flattens `(50, 50, 50, 16)` before `Dense(256)`. Its Table 2 reports
    512,197,467 trainable parameters. The implementation must calculate its own
    count and use the table only as a fidelity check.
- Scope:
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/model.py`
  - focused model tests inside the same implementation
  - implementation and paper READMEs only when model status or commands change
- Changes:
  - Extend `ConvLSTM2DCell` to accept an integer or two-dimensional kernel size
    while retaining the existing gate equations and same-size spatial output.
  - Extend `ConvLSTM2D` with `return_sequences`. When true, return
    `(B, T, C, H, W)`; otherwise return the final hidden state `(B, C, H, W)`.
    Preserve spatial feature maps between recurrent layers.
  - Add `OriginalPaperConvLSTM` as a separate, faithful baseline with default
    RGB input `(3, 50, 50)` and sequence length 50:
    time-distributed `Conv2D(16, 3x3)` → `ConvLSTM2D(64, 3x3)` returning the full
    sequence → time-distributed batch normalisation → time-distributed
    `Conv2D(16, 3x3)` → dropout 0.5 → flatten all time and spatial dimensions →
    `Linear(256)` → dropout 0.5 → classifier.
  - Infer same padding and full-sequence output from the paper's published tensor
    shapes. Do not add unreported ReLU layers to this baseline. Return logits in
    PyTorch and document that the loss applies the classification normalisation.
    Record all remaining Keras-to-PyTorch ambiguities in the model docstring.
  - Add `StackedConvLSTM` configured by explicit layer specifications of the form
    `[(filters, (kernel_h, kernel_w)), ...]`, for example:
    `[(8, (3, 3))]`, `[(8, (3, 3)), (16, (3, 3))]`, or
    `[(8, (3, 3)), (16, (3, 3)), (32, (5, 5))]`.
  - Validate that the layer list is non-empty, filters are positive, and kernel
    dimensions are positive odd integers so same padding preserves height and
    width. Intermediate recurrent layers return sequences; the final layer
    returns only its final spatial representation.
  - Give `StackedConvLSTM` a lightweight adaptive-average-pooling head with
    explicit dropout and an optional small hidden classifier width. Do not use a
    resolution-dependent flattened dense layer in the stacked model.
  - Add `count_trainable_parameters(model)` and use it for all parameter reports;
    do not store claimed totals in model classes.
  - Preserve the existing public model classes and imports used by current
    scripts. Mark the existing `ConvLSTMOriginal` clearly as a legacy
    approximation rather than silently changing its behaviour.
  - Do not alter or remove the current 3D-CNN comparison models. Document that
    `r3d_18`, `mc3_18`, and `r2plus1d_18` are not the paper's 3D ResNet-50/101/152;
    ticket 011 will align the comparison baseline registry.
  - Add focused CPU tests using small tensors and short sequences. Do not allocate
    the paper baseline's full 512-million-parameter dense layer in ordinary test
    memory; use a safe metadata/meta-device check or a documented reduced-shape
    topology test for fidelity.
- Acceptance criteria:
  - `ConvLSTM2D(return_sequences=True)` returns `(B, T, filters, H, W)` and
    `return_sequences=False` returns `(B, filters, H, W)` for the same input.
  - A two-layer specification such as `[(8, (3, 3)), (16, (5, 5))]` passes a
    sequence from the first layer to the second and produces class logits of
    shape `(B, num_classes)`.
  - One-, two-, and three-layer stacked configurations run on small CPU tensors,
    preserve spatial dimensions inside the stack, and have distinct calculated
    parameter counts.
  - The paper baseline's documented shapes and calculated parameter total are
    reconcilable with Tables 1–2 without silently replacing its flatten/dense
    head with pooling.
  - Invalid empty, zero-width, even-kernel, or malformed layer specifications
    fail with field-specific messages before a forward pass.
  - Existing model class imports still work, no 3D-CNN or experiment definition
    is removed, and no training/evaluation script is changed.
  - Focused tests pass and `git diff --check` reports no errors.
- Out of scope:
  - Training models, selecting a best model, or reporting experimental results.
  - Editing dataset, training, evaluation, experiment, or manuscript code.
  - Implementing paper-matched 3D ResNet-50/101/152 baselines; ticket 011 owns
    that work.
  - Reapplying the stashed ticket-009 experiment-configuration work. It must be
    revised after this model schema is stable.
- Open questions: None. For quick tests, use reduced spatial sizes and sequence
  lengths; these are engineering checks, not paper results. Later screening may
  use a common smaller input for speed only if every compared model uses it and
  the run is labelled preliminary.
- Verification:
  - Run focused unit tests for return modes, one/two/three-layer stacks, kernel
    handling, validation failures, output shapes, and parameter counting.
  - Trace tensor shapes through a two-layer example and record them in
    `completion.md`.
  - Compare the calculated faithful-baseline structure with the paper's Tables
    1–2 without allocating an unsafe CPU tensor.
  - Confirm existing model imports, `git diff --check`, and no changes outside
    the approved scope.

## Execution Prompt

Execute ticket `010-prepare-architecture-search` exactly as written in
`agents/work/010-prepare-architecture-search/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
