# Task Prompt

- Ticket: `018-shortlist-architecture-results`
- Status: Done
- Aim: Predeclare and validate one small AAD custom-architecture screen so the
  user can launch a fair comparison of depth and filter placement with one
  command.
- Scope:
  - local read-only audit of historical JSON metadata under Paper 001's ignored
    `models/` and `runs/experiments/legacy-pre-ticket-014/` directories
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiment_config.py`
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiments.py`
  - one explicit candidate manifest under
    `papers/001-journal-abnormal-activity-recognition/implementation/configs/`
  - `AGENTS.md` and the root, paper, and implementation READMEs only where
    experiment context, commands, status, or the next ticket changes
- Changes:
  - Treat all historical outputs as non-comparable exploratory evidence because
    they do not provide the current fixed split, complete seeding, full metric
    protocol, validation-only ranking, or checkpoint provenance. Do not use
    their test metrics to select candidates or copy generated outputs into Git.
  - Preserve `64-32-16` as the **selected lightweight ConvLSTM candidate** from
    earlier work, but give it no privileged rank in the new validation screen.
  - Add one validated, explicit candidate manifest containing exactly these six
    `CustomConvLSTM` models, all with `3x3` kernels and no hidden classifier:
    - `custom_reference_8`: `[(8, (3, 3))]`;
    - `custom_depth_8_8_8`: `[(8, (3, 3)), (8, (3, 3)), (8, (3, 3))]`;
    - `custom_capacity_early`: `[(16, (3, 3)), (8, (3, 3)), (8, (3, 3))]`;
    - `custom_capacity_middle`: `[(8, (3, 3)), (16, (3, 3)), (8, (3, 3))]`;
    - `custom_capacity_late`: `[(8, (3, 3)), (8, (3, 3)), (16, (3, 3))]`;
    - `custom_selected_64_32_16`: `[(64, (3, 3)), (32, (3, 3)), (16, (3, 3))]`.
  - Give every candidate a unique stable name and a concise research question.
    Keep this as an explicit list; do not generate a Cartesian hyperparameter
    grid or silently add more architectures.
  - Add typed loading and field-specific validation for the candidate manifest:
    reject unknown fields, duplicate or unsafe names, empty lists, malformed
    layers, even or non-positive kernels, non-positive filters, and invalid
    hidden widths before dataset access or run creation.
  - Add `--candidates-config` to `src.experiments`. When supplied, use the six
    manifest entries as the active custom-only screening registry. Keep the
    default no-manifest five-entry registry and all four audited baseline
    identities and roles unchanged.
  - Make `--list-models` work with the shared experiment config and candidate
    manifest without loading AAD or allocating models. Show each architecture
    and research question clearly.
  - Build every manifest model through the existing `CustomConvLSTM`; do not add
    new model classes or one script per architecture.
  - Save the exact candidate manifest and its SHA-256 hash in the run metadata,
    resolved configuration, each selected checkpoint, per-model metrics, and
    final summary. Preserve ticket 017's exact shared configuration provenance.
  - Keep the AAD screening protocol fixed at the ticket-017 JSON values: seed
    42, the committed 70:15:15 manifest, 16 frames at 32x32, online
    augmentation, 24 maximum epochs, and validation-loss early stopping.
  - Continue ranking only by validation macro-F1, validation accuracy, then
    parameter count. Retain the selected checkpoint and runtime for every model;
    never construct or read the test loader.
  - Document concise copy-paste commands to list the six candidates and start
    the full AAD architecture screen. Clearly label the full command as
    expensive and state that interruption leaves only completed model evidence.
  - Use temporary pytest checks; do not add a permanent test suite or commit
    datasets, environments, checkpoints, generated runs, or legacy artifacts.
- Acceptance criteria:
  - One committed candidate manifest contains exactly the six approved names,
    layer stacks, and research questions—no implicit combinations.
  - Supplying both JSON files resolves the same shared protocol from ticket 017
    and replaces only the custom candidate set; default no-manifest behavior
    still lists the original five audited entries.
  - Candidate listing is side-effect free, and each listed custom architecture
    can be instantiated and complete a small CPU forward pass.
  - Invalid or duplicate candidate definitions fail with field-specific errors
    before AAD access or run creation.
  - Run, checkpoint, per-model, and summary metadata retain the exact shared
    configuration plus candidate-manifest content and hash.
  - Training, selection, early stopping, restored weights, full-partition
    metrics, validation-only ranking, and complete test isolation remain intact.
  - Historical files, the ticket-009 stash, baseline definitions, model classes,
    split manifest, augmentation, evaluation command, and manuscript remain
    unchanged.
  - Temporary tests, candidate CPU smoke checks, Black, Python compilation,
    Pylint error-only, and `git diff --check` pass.
- Out of scope:
  - Running the full AAD screen inside this ticket; the user will launch the
    documented expensive command after reviewing the candidate list.
  - Comparing the shortlist with `PaperConvLSTM`, `r3d_18`, `mc3_18`, or
    `r2plus1d_18`; ticket 019 owns the controlled baseline comparison.
  - Selecting a final model from test results or opening the test split.
  - Kernel-size, input-size, frame-count, augmentation, optimiser, weight-decay,
    head-design, or other training ablations; ticket 020 owns one-factor
    ablations after the reference architecture is selected.
  - Rewriting manuscript results or claims.
- Open questions: None.
- Verification:
  - Parse and stable-round-trip the candidate manifest; test unknown fields,
    duplicate names, unsafe names, malformed layers, and invalid widths.
  - Resolve `--config` plus `--candidates-config` and list the six candidates
    without dataset access, model allocation, or run creation.
  - Instantiate each candidate and run a tiny CPU forward pass; confirm unique
    parameter counts are recorded where architectures differ.
  - Confirm the no-manifest registry still contains exactly the source-paper
    topology, one default custom reference, and the three practical 3D-CNNs.
  - Use a temporary miniature comparison or focused mocks to verify manifest
    provenance reaches checkpoints and summaries while the test split remains
    untouched; do not start the full AAD run.
  - Run ticket-016/017 regression checks, Black, Python compilation, Pylint in
    error-only mode, `git diff --check`, and a final scope review.

## Execution Prompt

Execute ticket `018-shortlist-architecture-results` exactly as written in
`agents/work/018-shortlist-architecture-results/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
