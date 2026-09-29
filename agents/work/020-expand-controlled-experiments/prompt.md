# Task Prompt

- Ticket: `020-expand-controlled-experiments`
- Status: Done
- Aim: Prepare one reproducible AAD experiment suite that expands the custom
  architecture screen and supports controlled baseline confirmation and
  one-factor ablations from Colab.
- Scope:
  - `papers/001-journal-abnormal-activity-recognition/implementation/src/experiments.py`
  - shared configuration validation only where the experiment runner requires it
  - AAD experiment JSON files under `implementation/configs/`
  - `implementation/notebooks/aad_experiment_workflow.ipynb`
  - focused tests and concise repository, paper, and implementation guidance
- Changes:
  - Expand the custom manifest from six to these eleven predeclared `3x3`
    ConvLSTM stacks, each with a stable name and short research question:
    `[8]`, `[8, 16]`, `[16, 8]`, `[32, 8]`, `[64, 8]`, `[8, 8, 8]`,
    `[16, 8, 8]`, `[8, 16, 8]`, `[8, 8, 16]`, `[64, 16, 8]`, and
    `[64, 32, 16]`.
  - Keep architecture screening fixed at AAD, split seed `42`, run seed `42`,
    16 frames, `32x32`, learning rate `0.001`, weight decay `0.001`, online
    augmentation enabled, maximum 24 epochs, patience 10, and validation-only
    ranking. Do not treat the earlier eight-epoch smoke run as a completed screen.
  - Extend the existing runner instead of adding one script per study. Provide
    safe inspection output showing exact models, protocols, changed factors,
    run counts, and commands before allocating data or models.
  - Support an explicit validation-selected custom candidate in a comparable
    confirmation with `r3d_18`, `mc3_18`, and `r2plus1d_18`.
  - Keep `PaperConvLSTM` as the audited source-paper topology. Its native
    50-frame `50x50` protocol and very large classifier must run separately and
    must not be presented as a one-factor comparison with the lightweight
    screen. If native training is infeasible, retain topology and parameter
    evidence and report the resource limitation rather than substituting a
    smaller model.
  - Add a focused ablation plan for one explicitly named custom reference. Keep
    learning rate fixed at `0.001` with the scheduler and change one factor at a
    time: weight decay `0`, `0.0001`, `0.001`; augmentation disabled/enabled;
    spatial size `32x32`/`64x64`; and sequence length 16/32 frames.
  - Treat 24 epochs for screening and 64 maximum epochs for confirmation and
    ablations as compute stages, not an epoch-count ablation. Preserve
    validation-loss early stopping and selected-checkpoint restoration.
  - Keep the fixed AAD split, seeds, batch settings, architecture, and other
    values unchanged within each one-factor comparison. Reject a plan changing
    several factors unless it is explicitly labelled a new protocol.
  - Update the Colab-first notebook with separately guarded screening, baseline,
    and ablation stages through existing CLIs. Continue using ticket 019's
    KaggleHub AAD download and `/content/...` defaults; do not duplicate logic.
  - Preserve provenance, validation-only selection, locked test access,
    timestamped artifacts, and confirmation seeds `42` and `2026`.
  - Keep all costly actions disabled. Do not run the full suite in this ticket.
- Acceptance criteria:
  - Candidate validation and `--list-models` show exactly the eleven approved
    stacks in order, without duplicate names or research questions.
  - Safe plan inspection separates custom screening, practical-baseline
    confirmation, the published topology, and focused ablations without AAD or
    model allocation.
  - The focused plan has one named reference; every non-reference run differs
    from it by exactly one approved factor.
  - Learning rate stays `0.001`; factor values match the approved set; epoch
    count is not reported as an ablation factor.
  - The three practical 3D CNNs and faithfully labelled `PaperConvLSTM` remain
    present in their distinct comparison roles.
  - The notebook stays valid nbformat 4, output-free, Colab-first, and guarded;
    all network, training, test, and archive actions default to false.
  - No dataset, checkpoint, generated run, notebook output, or result claim is
    committed.
- Out of scope:
  - Running expensive experiments or final test evaluation.
  - Selecting a final model, changing learning rate, testing optimizers, treating
    epochs as an ablation, or adding ticket 021's optional candidates.
  - Rewriting manuscript results before versioned evidence exists.
- Open questions: None.
- Verification:
  - Run focused configuration/runner tests, including duplicate-model,
    multi-factor-plan, and test-lock failures.
  - Exercise safe config printing, candidate listing, and plan inspection without
    AAD access or model allocation.
  - Parse and compile notebook cells; verify unique IDs, empty outputs, null
    execution counts, false action flags, paths, and the double test guard.
  - Run relevant existing tests, `git diff --check`, and a scope/status review.

## Execution Prompt

Execute ticket `020-expand-controlled-experiments` exactly as written in
`agents/work/020-expand-controlled-experiments/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
