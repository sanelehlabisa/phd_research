# Task Prompt

- Ticket: `021-optional-cloud-architecture-expansion`
- Status: Ready
- Aim: Optionally add one predeclared supplemental custom-architecture block
  before the full AAD screen when a Colab timing check shows sufficient runtime.
- Scope:
  - the Paper 001 custom architecture manifest and focused validation tests
  - concise Paper 001 and implementation roadmap notes
- Changes:
  - Execute only after ticket 020 and before any full architecture screen. Base
    the decision on measured Colab time and memory, never favourable results.
  - Add all five supplemental `3x3` stacks as one block, or add none:
    - `[8, 8]` — two-layer depth bridge;
    - `[16, 16]` — uniform wider two-layer stack;
    - `[32, 16]` — moderate descending two-layer stack;
    - `[32, 16, 8]` — moderate three-layer funnel;
    - `[8, 8, 8, 8]` — deeper narrow stack.
  - Give each candidate a stable unique name and concise, non-overlapping
    research question.
  - Use the exact ticket-020 screening protocol, split, seed, budget,
    augmentation, optimizer, scheduler, learning rate, weight decay, and
    validation-only ranking. Do not add another hyperparameter grid.
  - Update safe model/plan listing and documentation counts. Do not change the
    baselines, `PaperConvLSTM`, ablation factors, manuscript, or notebook flow
    beyond reflecting the expanded candidate count.
- Acceptance criteria:
  - The manifest contains the eleven ticket-020 models plus all five approved
    supplemental models in a deterministic documented order.
  - Validation rejects duplicate names and malformed stacks; safe listing shows
    the full set without allocating models or accessing AAD.
  - No results determine whether the optional block is added, and no full
    experiment runs during implementation.
- Out of scope:
  - Adding only a favourable subset, changing the screening protocol, expanding
    kernel sizes or classifier heads, running the screen, or modifying claims.
- Open questions: None. If the ticket-020 screen has started, do not execute
  this ticket; retain it as a skipped contingency record.
- Verification:
  - Run candidate validation and safe list/plan inspection.
  - Verify exact model order and protocol equality with ticket 020.
  - Run `git diff --check` and a final scope/status review.

## Execution Prompt

Execute ticket `021-optional-cloud-architecture-expansion` exactly as written in
`agents/work/021-optional-cloud-architecture-expansion/prompt.md`. Follow
`AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the approved
changes, verify every acceptance criterion, set the ticket status to `Done`,
and create `completion.md` from `agents/templates/completion.md`.
