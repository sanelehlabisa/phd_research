# Task Prompt

- Ticket: `046-separate-aad-screen-config`
- Status: Done
- Depends on: Ticket 020; ticket 021 is skipped by user decision.
- Aim: Keep the fixed 24-epoch AAD architecture screen independent from the
  user's longer-training configuration.
- Scope: Paper 001 experiment-plan JSON, a dedicated screening config, focused
  validation, and concise Paper 001/implementation guidance.
- Changes:
  - Add a dedicated AAD architecture-screen config matching ticket 020:
    24 maximum epochs, batch size 16, 16 frames, `32x32`, learning rate
    `0.001`, weight decay `0.001`, online augmentation, Adam with
    `reduce_on_plateau`, cross-entropy, seed/split seed `42`, the existing
    70:15:15 split manifest, and early-stopping patience 10. Candidate layer
    specifications come from the existing eleven-model manifest.
  - Point the controlled plan's screening stage to the dedicated config.
  - Preserve the existing user-modified `aad_screening_reference.json` and all
    run/checkpoint files unchanged; do not stage or commit them.
  - Document that the AAD architecture screen stays at 24 epochs and 64 epochs
    are for the later selected-model training stage, not a screening factor.
- Acceptance criteria:
  - Safe plan listing succeeds and reports the fixed ticket-020 screening
    protocol and eleven candidates.
  - The dedicated screening config validates independently of the local
    longer-training config.
  - No training, evaluation, dataset download, or test access is performed.
  - Focused tests and `git diff --check` pass; unrelated local config and run
    artifacts remain untouched.
- Out of scope: Changing candidate architectures, adding ticket 021's five
  optional candidates, changing confirmation/ablation budgets, running any
  experiment, or modifying paper claims.
- Open questions: `None`.
- Verification: Validate JSON and protocol values; run the safe plan listing and
  relevant tests; compare before/after hashes for the preserved user config and
  run artifacts; run `git diff --check`.

## Execution Prompt

Execute ticket `046-separate-aad-screen-config` exactly as written in
`agents/work/046-separate-aad-screen-config/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`.
