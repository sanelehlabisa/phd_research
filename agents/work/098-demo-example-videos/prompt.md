# Task Prompt

- Ticket: `098-demo-example-videos`
- Status: Done
- Aim: Bundle three clickable example videos with the working Gradio demo.
- Scope: `demo_assets/examples/`, demo example discovery, Spaces entry/export and checks.
- Approval: User requested copying and including three examples after successful deployment.
- Changes:
  - Copy three small original validation clips; record sources, hashes and split evidence.
  - Show examples locally and in Spaces without automatic GPU predictions.
  - Include only these named samples in a fresh upload ZIP.
- Acceptance criteria:
  - Exactly three playable, unchanged source videos appear as clickable examples.
  - Upload predictions and working ZeroGPU setup remain unchanged; test data stays locked.
  - Export includes examples; original files and earlier exports are preserved.
- Out of scope: Retraining, result changes, commit/push or remote upload.
- Open questions: None.
- Verification: Split membership, copy hashes, video decoding, example wiring and export tests.

## Execution Prompt

Complete the approved work above, verify the result and create `completion.md`. Preserve existing work; do not commit, push or deploy.
