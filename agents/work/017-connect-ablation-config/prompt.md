# Task Prompt

- Ticket: `017-connect-ablation-config`
- Status: Done
- Aim: Connect one validated JSON configuration to Paper 001's training and
  comparison runners so controlled ablation runs use the same explicit model,
  input, data, and training settings.
- Scope:
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiment_config.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/train.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiments.py`
  - one concise example under
    `papers/001-journal-abnormal-activity-recognition/implementation/configs/`
  - `AGENTS.md` and the root, paper, and implementation READMEs only where
    commands, status, or the next ticket changes
- Changes:
  - Reimplement the useful configuration ideas from the preserved ticket-009
    stash against the current APIs. Do not apply, pop, drop, or edit the stash,
    and do not restore its obsolete runner code.
  - Add a typed, JSON-serialisable configuration covering:
    - dataset path, runs directory, split manifest, fixed split seed, 70:15:15
      ratios, and non-negative run seed;
    - sequence length, frame height, and frame width;
    - an explicit `CustomConvLSTM` layer list in the current
      `(filters, (kernel_height, kernel_width))` form and optional hidden
      classifier width;
    - epochs, validation-loss early-stopping patience, batch size, learning
      rate, weight decay, online augmentation flag, worker count, and pinned
      memory;
    - the currently fixed Adam, cross-entropy, and ReduceLROnPlateau choices as
      recorded protocol values, not silently ignored free-form controls.
  - Validate unknown fields, types, ratios, dimensions, layer tuples, kernel
    sizes, probabilities or rates, seeds, and positive training values before
    dataset loading or run-directory creation. Errors must name the bad field.
  - Add `--config` and `--print-config` to `src.train` and `src.experiments`.
    JSON provides the base values; only CLI options explicitly supplied by the
    user override matching JSON values. Repeated `--convlstm-layer` options
    replace the complete JSON layer list rather than appending to it.
  - Keep both runners' existing no-config CLI behavior available. Do not rename
    established CLI flags or reinterpret historical outputs as controlled runs.
  - Make `--print-config` emit stable, normalised JSON and exit before loading a
    dataset, allocating a model, creating a run, or opening a test split.
  - Use the resolved custom architecture in both runners. In
    `src.experiments`, keep the audited paper topology and the `r3d_18`,
    `mc3_18`, and `r2plus1d_18` registry entries and roles unchanged; only its
    custom reference entry is supplied by the resolved configuration.
  - Save the exact normalised configuration in each run and embed it in every
    validation-selected checkpoint created by these runners. Preserve the
    ticket-016 metric protocol, validation-loss selection, restored best
    weights, early stopping, ranking, and complete test isolation.
  - Add an AAD screening-reference JSON using the committed seed-42 split
    manifest, 70:15:15 ratios, seed 42, 16 frames at 32x32, online augmentation,
    and the documented two-layer `8 -> 16` custom ConvLSTM. Label it as a
    screening reference, not an optimal or final model.
  - Document short copy-paste commands that print the example configuration and
    run it through training or comparisons. State clearly that the latter two
    commands start expensive training.
  - Use temporary pytest checks when helpful; do not add a permanent test suite
    or commit generated runs, datasets, checkpoints, environments, or caches.
- Acceptance criteria:
  - The example JSON alone resolves the complete AAD screening reference without
    editing Python source.
  - Both runners resolve the same shared fields identically, and an explicitly
    supplied CLI value wins over JSON while unspecified JSON values remain.
  - The custom model instantiated from the example has exactly the two requested
    recurrent layers; the baseline registry remains the audited five-entry set.
  - No-config CLI commands retain their current defaults and behavior.
  - Invalid JSON, unknown fields, invalid ratios, malformed layers, and invalid
    training values fail with field-specific messages before dataset access or
    run creation.
  - Stable JSON round-trips without loss and the saved run/checkpoint metadata
    contains the exact resolved configuration.
  - `--print-config` for both runners is side-effect free and does not require
    the AAD dataset or model allocation.
  - Training and comparisons still never construct or read the test loader, and
    all ticket-016 metric, selection, checkpoint-restoration, and ranking checks
    continue to pass.
  - The ticket-009 stash and existing experiment artifacts remain unchanged.
  - Formatting, focused checks, Python compilation, lint error checks, and
    `git diff --check` pass.
- Out of scope:
  - Applying or deleting the ticket-009 stash.
  - Choosing the architecture shortlist; ticket 018 owns that decision.
  - Running full training, comparisons, ablations, or final test evaluation.
  - Changing the fixed split manifest, model implementations, baseline
    identities, augmentation behavior, metrics, selection rules, or manuscript
    results.
  - Adding configuration-driven final test access; evaluation remains an
    explicit, checkpoint-based command.
- Open questions: None.
- Verification:
  - Run temporary pytest checks for JSON loading, stable round-trip,
    field-specific validation, no-config defaults, explicit CLI precedence,
    complete layer-list replacement, and side-effect-free `--print-config`.
  - Print the example through both runners and compare their shared resolved
    fields without loading AAD.
  - Instantiate only the configured custom model and inspect its layer
    configuration; list the baseline registry and confirm the same five names.
  - Re-run the focused ticket-016 synthetic checks for full-partition metrics,
    validation-only selection/restoration, ranking, and test-access isolation.
  - Run Black checks on changed Python files, compile them, run Pylint in
    error-only mode, and run `git diff --check`.
  - Review `git status`, the stash identifier, and generated-file exclusions.

## Execution Prompt

Execute ticket `017-connect-ablation-config` exactly as written in
`agents/work/017-connect-ablation-config/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
