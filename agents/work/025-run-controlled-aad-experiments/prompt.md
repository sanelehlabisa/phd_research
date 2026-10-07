# Task Prompt

- Ticket: `025-run-controlled-aad-experiments`
- Status: Draft
- Aim: Execute the frozen AAD comparison plan and select a reference
  configuration using validation evidence only.
- Scope:
  - Paper 001 controlled plan, candidate manifest, and Colab experiment workflow
  - local ignored `runs/experiments/` evidence
  - concise Paper 001 and implementation status notes
- Changes:
  - Use the fixed 24-epoch screening profile from ticket 020; reserve 64 epochs
    for the later selected-model training ticket. Do not use a longer local
    training profile for candidate screening.
  - Execute only after ticket 024 is complete and the ticket-021 supplemental
    candidate-block decision is recorded before screening.
  - Run the complete architecture screen with the fixed split, seed, 24-epoch
    budget, online augmentation, optimizer, scheduler, learning rate, weight
    decay, input size, and validation-only ranking.
  - Do not select from an interrupted screen. Resume or rerun missing candidates
    until every predeclared candidate has comparable provenance and artifacts.
  - Select the reference using validation macro-F1, then validation accuracy,
    parameter count, and runtime as documented tie-break information. Never use
    test results or three example predictions.
  - Update the controlled plan and confirmation configuration to the selected
    architecture with a short evidence link; do not silently preserve the
    preliminary `custom_depth_8_8_8` reference if another candidate wins.
  - Run two-seed practical-baseline confirmation and the two-seed one-factor
    ablations for weight decay, augmentation, spatial size, and sequence length.
  - Attempt the native published topology separately when resources allow.
    Record an out-of-memory/resource limitation rather than reducing it while
    still calling it faithful.
  - Keep every test partition locked and retain all configurations, histories,
    validation metrics, selected checkpoints, confusion matrices, parameters,
    runtimes, seeds, split hashes, and code revisions.
- Acceptance criteria:
  - The architecture screen is complete and its selected reference is traceable
    to validation-only evidence.
  - Baseline confirmation and each declared ablation have both planned seeds or
    an explicit incomplete/resource-failure record.
  - Augmentation on/off, weight decay `0/0.0001/0.001`, `32/64` spatial
    size, `16/32` frames, three 3D-CNNs, and the original topology retain
    distinct documented roles.
  - No test metric influences architecture or factor selection.
- Out of scope:
  - Final test evaluation, VDD, unplanned hyperparameter grids, Kinetics,
    manuscript rewriting, or committing generated datasets/checkpoints/runs.
- Open questions: Superseded by ticket 052's updated JSON-driven AAD plan; do not
  execute this older run matrix. Ticket 021 remains recorded as skipped.
- Verification:
  - Validate the plan and list exact commands before execution.
  - Inspect every produced run manifest and validation-only summary.
  - Verify expected run counts, seeds, split hashes, configurations, and locked
    test access; run the focused suite and `git diff --check`.

## Superseded

Do not execute this older prompt. Use the updated AAD experiment plan in
`agents/work/052-json-controlled-aad-experiments/prompt.md` after the earlier
foundation tickets are reviewed and completed.
