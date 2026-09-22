# Task Prompt

- Ticket: `011-simplify-model-api`
- Status: Ready
- Aim: Reduce the implementation to one clear ConvLSTM layer and two model
  classes, then migrate every current caller without retaining compatibility
  wrappers for obsolete architectures.
- Scope:
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/model.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/train.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/evaluate.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiments.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/demo.py`
  - focused tests in the same implementation
  - root, paper, and implementation guidance when names, commands, or task status
    change
- Changes:
  - Keep exactly three public neural-network classes in `model.py`:
    `ConvLSTM` for the reusable sequence layer, `PaperConvLSTM` for the faithful
    published topology, and `CustomConvLSTM` for configurable stacks. Keep the
    cell implementation private as `_ConvLSTMCell`.
  - Remove `ConvLSTM2DCell`, `ConvLSTM2D`, `ConvLSTMOriginal`,
    `OriginalPaperConvLSTM`, `StackedConvLSTM`, `ConvLSTMPooledModel`,
    `ConvLSTMPooledModelV1`, `ConvLSTMModel`, and `ConvLSTMCustom`. Do not leave
    aliases or compatibility wrappers.
  - Remove `TypeAlias` and the `KernelSize`/`LayerSpec` aliases. Use direct,
    readable type hints such as `int | tuple[int, int]` and
    `list[tuple[int, tuple[int, int]]]`.
  - Preserve ticket 010's validated behaviour: integer or two-dimensional odd
    kernels, full-sequence or final-state output, same spatial dimensions,
    field-specific layer validation, adaptive pooling for the custom model, and
    calculated trainable-parameter counts.
  - Give `CustomConvLSTM` an explicit `layers` list containing
    `(filters, (kernel_height, kernel_width))` entries. Keep dropout explicit and
    retain an optional positive hidden-classifier width. Use `input_channels`
    rather than a full input shape because its pooled head is resolution
    independent.
  - Keep `PaperConvLSTM` structurally faithful and separate. Its defaults remain
    50 RGB frames at `50x50`; reduced inputs are allowed only for smoke checks or
    clearly labelled reduced-input comparisons.
  - Rewrite model docstrings in a consistent junior-friendly form: a one-line
    description, then concise `Parameters` and `Returns` sections. Add complete
    parameter and return type hints without introducing aliases.
  - Replace the current dataset-based `model.py` command with a deterministic CPU
    smoke check using small random tensors. Run both `PaperConvLSTM` and
    `CustomConvLSTM`, use inference mode, and print each model's input shape,
    logits shape, and calculated trainable parameters. Do not load a dataset,
    checkpoint, or write media from this command.
  - Migrate `train.py`, `evaluate.py`, and `demo.py` to `CustomConvLSTM`. Replace
    `--custom_filters` with a repeatable argument:
    `--convlstm-layer FILTERS KERNEL_HEIGHT KERNEL_WIDTH`. Convert repeated
    values directly to the model's `layers` list and use one small single-layer
    default only when the option is omitted. Add an optional
    `--hidden-classifier-width` argument and keep matching architecture options
    consistent across all three callers.
  - Update new checkpoints and metadata to identify `CustomConvLSTM`, its exact
    layer list, hidden-classifier width, and class count. If a supplied legacy
    checkpoint uses the deleted architecture keys, fail with a concise
    incompatibility message; never silently continue with random weights.
    Preserve all existing checkpoint files on disk.
  - Simplify `experiments.py` to construct `PaperConvLSTM`, one explicitly named
    `CustomConvLSTM` reference configuration, and the three existing 3D-CNN
    models. Remove the obsolete original/light/pooled and four-value custom
    configurations. Label a paper model using non-default sequence or spatial
    dimensions as a reduced-input topology check, not a faithful paper result.
  - Update focused tests for the new names and API. Remove compatibility tests
    for deleted classes and assert that no current Python caller imports or
    constructs an obsolete model.
- Acceptance criteria:
  - `model.py` exposes `ConvLSTM`, `PaperConvLSTM`, and `CustomConvLSTM`; its only
    other neural-network class is the private `_ConvLSTMCell`.
  - No type alias or deleted model name remains in active implementation Python
    files. Historical ticket records and the user's `new_prompt.md` remain
    untouched.
  - `CustomConvLSTM(layers=[(8, (3, 3)), (16, (5, 5))], ...)` produces logits
    shaped `(B, num_classes)` while preserving spatial dimensions through the
    recurrent stack.
  - `python -m src.model` completes on CPU with both models and uses reduced
    tensors that cannot allocate the faithful baseline's full dense layer.
  - `train.py`, `evaluate.py`, and `demo.py` show the repeatable layer argument in
    `--help`, resolve identical inputs to identical layer lists, and instantiate
    `CustomConvLSTM` with the same architecture options.
  - A new-format checkpoint can be reconstructed from its saved architecture
    metadata. A representative legacy checkpoint dictionary fails clearly
    before training or evaluation and is not replaced with random weights.
  - `experiments.py` contains only the two approved ConvLSTM model families plus
    the unchanged three 3D-CNN comparison models; it does not start training
    during verification.
  - Model and caller imports compile, focused tests pass, and `git diff --check`
    reports no errors.
- Out of scope:
  - Loading or converting old model weights.
  - Running training, evaluation, architecture search, or reporting results.
  - Selecting the final custom layer configurations for the research study.
  - Replacing or claiming equivalence for the current 3D-CNN baselines; ticket
    012 owns comparison-baseline alignment.
  - Reapplying the stashed ticket-009 configuration work.
  - Changing dataset splitting, metrics, checkpoint selection policy, or the
    manuscript.
- Open questions: None. The removal is intentionally breaking; historical
  checkpoints and outputs stay on disk but are not compatible with the new API.
- Verification:
  - Run focused tests for kernels, return modes, stack shapes, validation,
    parameter counts, both retained models, CLI layer conversion, checkpoint
    compatibility handling, and absence of deleted imports.
  - Run `python -m src.model` and record both output shapes and parameter counts.
  - Run `python -m src.train --help`, `python -m src.evaluate --help`, and
    `python demo.py --help` without accessing a dataset.
  - Compile `model.py`, `train.py`, `evaluate.py`, `experiments.py`, `demo.py`,
    and the focused tests.
  - Search active implementation Python files for deleted names and `TypeAlias`.
  - Confirm the user's `new_prompt.md`, old checkpoints, generated runs, the
    ticket-009 stash, 3D-CNN definitions, and manuscript are unchanged.
  - Run `git diff --check` and review the final diff for unrelated changes.

## Execution Prompt

Execute ticket `011-simplify-model-api` exactly as written in
`agents/work/011-simplify-model-api/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
