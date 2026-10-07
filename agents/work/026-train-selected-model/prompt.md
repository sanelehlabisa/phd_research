# Task Prompt

- Ticket: `026-train-selected-model`
- Status: Blocked
- Code preparation: Ticket 055 supplies notebook 05 and its script export; actual GPU execution remains pending.
- Execution check (2026-10-07): No completed ticket-052 study/selection exists locally. Await ticket 053's current validation evidence and an authorised GPU runtime; do not select from the older six-model screen.
- Aim: Train and freeze the validation-selected architecture and input protocol
  with the full confirmation budget.
- Scope:
  - Paper 001 selected experiment configuration and training workflow
  - local ignored study/final and linked `runs/experiments/` evidence (shared runner)
  - concise Paper 001 and implementation status notes
- Changes:
  - Execute only after ticket 053 (current 052 study, superseding 025) has comparable architecture and
    one-factor validation evidence.
  - Freeze the selected architecture, augmentation state, weight decay, spatial
    size, and sequence length from validation evidence. Do not assume that
    larger frames, more frames, more layers, or wider layers are automatically
    better.
  - If individually favourable factors are combined, confirm that combination
    as one named validation-only configuration before calling it selected.
  - Train the frozen configuration for the planned 64-epoch maximum with
    validation-loss early stopping, learning-rate scheduling, seeds `42` and
    `2026`, and restoration of each seed's best validation checkpoint.
  - Save the full configuration, split and code provenance, histories, learning
    rates, validation metrics, confusion matrices, checkpoints, parameter count,
    runtime, and uncertainty across seeds.
  - Choose the checkpoint used by the next ticket using validation evidence
    before opening the test partition.
- Acceptance criteria:
  - The final configuration is reconstructable and each chosen factor has
    ticket-053 validation evidence.
  - Both confirmation seeds complete or have an explicit failure record, and
    their aggregate validation results are reported without cherry-picking.
  - Reusable best checkpoints and complete histories exist under timestamped
    training runs with the test partition still locked.
- Out of scope:
  - Test evaluation, VDD, Kinetics, architecture expansion, additional
    hyperparameter search, manuscript claims, or committed generated runs.
- Open questions: None. Concrete selected values are outputs of ticket 053, not
  assumptions made before it.
- Verification:
  - Validate resolved configurations and checkpoint reconstruction.
  - Check seed, split hash, selected epoch, early stopping, scheduler, metrics,
    runtime, and locked-test provenance for both runs.
  - Run focused tests and `git diff --check`.

## Execution Prompt

Execute ticket `026-train-selected-model` exactly as written in
`agents/work/026-train-selected-model/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
