# Task Prompt

- Ticket: `095-simple-demo-spaces`
- Status: Done
- Aim: Simplify the demo UI and provide a small, ready-to-upload Spaces bundle.
- Scope: Demo layout, deployment template/exporter, tests and concise instructions.
- Approval: User requested simpler UI, emojis, attribution and deployment steps.
- Changes:
  - Short upload/predict interface with “By Sanele Hlabisa”; details collapsed.
  - Preserve trained weights, prediction API and conspicuous random-mode warning.
  - Prepare CPU Gradio Spaces entry/config and a local upload folder, excluding research data.
- Acceptance criteria:
  - Existing predictions work; local app stays localhost-only.
  - Bundle contains source, pinned dependencies and real weights/metadata, not an LFS pointer.
  - Deployment instructions require no source editing; no automatic publication.
- Out of scope: Hosting/account actions, commit/push, retraining, metrics or manuscript changes.
- Open questions: None.
- Verification: UI/API regression tests, offline bundle checks and local prediction.

## Execution Prompt

Complete the approved work above, follow repository rules, verify the result and create `completion.md`. Preserve existing work; do not commit, push or deploy.
