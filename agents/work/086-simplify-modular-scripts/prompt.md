# Task Prompt

- Ticket: `086-simplify-modular-scripts`
- Status: Ready
- Aim: Plan a smaller, clearer modular-script surface without breaking the working research pipeline.
- Scope: Paper 001's `implementation/src/`, notebook helpers, configs, tests and documented commands; inventory and cleanup plan first.
- Dependencies: Do not delay [084](../084-diagnose-transformer-learning/prompt.md) or [083](../083-comparison-only-recovery/prompt.md); defer structural changes until the final comparison path is verified.
- Changes:
  - Inventory scripts and callers, including imports, dynamic/CLI entry points, notebooks/exports, tests and config/document links. Classify active core modules, reusable helpers, legacy workflows and genuinely unreferenced files; small files are not automatically unnecessary.
  - Keep one training/comparison implementation behind simple JSON commands. Identify actual duplication and propose only useful merges/removals, preserving clear model/dataset/train/evaluate/experiments entry points.
  - Record the exact proposed move/delete paths, replacement imports/commands, historical workflow impact and verification in a short plan. Obtain approval for that manifest before deletion or renaming; no broad cleanup from guessed usage.
  - After approval, a scoped follow-up executes the manifest with import/CLI/notebook parity checks and concise README/context updates. Keep datasets, reports, checkpoints, existing results and provenance untouched; do not invent compatibility guarantees.
- Acceptance criteria:
  - Every proposed removal has a documented caller audit and replacement or retirement reason. Required active paths and useful helpers are not deleted merely to reduce file count.
  - The user has a concise exact-path cleanup plan and verification/rollback strategy; actual deletion remains gated on its approval.
  - Existing commands, configs, saved evidence and running experiments are unchanged during this planning ticket.
- Out of scope: Unapproved deletion/moves, a new framework, algorithm/protocol changes, training, generated-run cleanup, commit or push.
- Open questions: None for the inventory. Exact removal/rename approval is a deliberate follow-up gate after the evidence exists.
- Verification: Read-only caller/import/CLI/config/notebook audit, path/link checks and `git diff --check`; no broad training run required.

## Execution Prompt

Execute ticket `086-simplify-modular-scripts` exactly as written here. Follow
`AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Produce the inventory and
cleanup plan only; do not delete/move files before manifest approval. Verify,
set the status to `Done`, and create `completion.md` from the completion template.
Do not train, commit or push.
