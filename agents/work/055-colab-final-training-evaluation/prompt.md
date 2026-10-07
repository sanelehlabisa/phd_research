# Task Prompt

- Ticket: `055-colab-final-training-evaluation`
- Status: Done
- Approval: User confirmed code-only Colab notebook/script preparation on 2026-10-07.
- Aim: Prepare an exportable AAD notebook for the sequential 026 -> 027 workflow.
- Scope: New notebook/script and shared helper, existing evaluator, tests and guides.
- Changes:
  - Require completed 052/053 validation evidence; never use the historical screen.
  - Select the best validated single-factor configuration of the screen winner;
    rank seed means by validation accuracy, loss and parameter count. Do not combine factors.
  - Confirm seeds 42/2026 for up to 64 epochs with the existing runner, then freeze
    the lowest-validation-loss checkpoint before test access.
  - Display curves, final overall-micro/per-class metrics and prediction examples;
    preserve provenance and reuse completed results on reruns.
  - Support custom, practical CNN and Swin winners without changing their topology.
- Acceptance criteria:
  - Plain-Python export matches notebook code and uses the requirements bootstrap.
  - Missing/incompatible evidence prevents training/testing; partial test attempts
    cannot silently repeat. Existing notebooks and saved outputs remain unchanged.
  - Tickets 026/027 remain pending actual authorised Colab execution.
- Out of scope: Real GPU runs, manuscript changes, Kinetics changes, commit/push.
- Open questions: None; code preparation explicitly approved.
- Verification: Synthetic end-to-end and guard tests, script/notebook syntax,
  full CPU suite, protected-file hashes and `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and
`agents/config.md`; verify, update status and create `completion.md`.
