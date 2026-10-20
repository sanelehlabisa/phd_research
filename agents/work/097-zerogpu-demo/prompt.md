# Task Prompt

- Ticket: `097-zerogpu-demo`
- Status: Done
- Aim: Run the existing trained Gradio demo on the user's ZeroGPU-only Space.
- Scope: Spaces entry/dependencies, a reusable UI prediction hook, deployment tests and guide.
- Approval: User needs the deployment fixed for ZeroGPU; CPU hardware is unavailable.
- Changes:
  - Use supported CUDA-enabled PyTorch/torchvision and pinned `spaces` in one requirements file.
  - Import Spaces before model code, load the model once on CUDA, and decorate real inference.
  - Expose a module-level Gradio app; preserve local CPU use and checkpoint validation.
  - Prepare a fresh upload folder/ZIP without overwriting old exports.
- Acceptance criteria:
  - ZeroGPU decorator reaches actual model inference; no dummy GPU function or CPU-package workaround.
  - Existing UI, weights, class order, sampling and research results remain unchanged.
  - Tests verify startup/import order, decorated callback, packaging and local CPU behavior.
- Out of scope: Retraining, manuscript/results edits, commit/push, account/hardware changes or remote upload.
- Open questions: None.
- Verification: Unit tests, published dependency/source compatibility checks, clean export/weight hash and local CPU prediction. Hosted ZeroGPU execution remains user-run.

## Execution Prompt

Complete the approved work above, follow repository rules, verify the result and create `completion.md`. Preserve existing work; do not commit, push or deploy.
