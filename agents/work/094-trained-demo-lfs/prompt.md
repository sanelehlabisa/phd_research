# Task Prompt

- Ticket: `094-trained-demo-lfs`
- Status: Done
- Aim: Use the imported Colab checkpoint in the local demo and prepare it for Git LFS cloning.
- Scope: Curated demo assets, exact LFS rule, demo defaults/guards, tests and guides.
- Approval: User requested using the newly added model and adding Git large-file storage.
- Changes:
  - Verify the imported checkpoint against its sidecar, recipe and class evidence.
  - Copy the checkpoint unchanged into `demo_assets/winner/` with matching metadata.
  - Default the app to trained weights; retain explicit random/custom modes.
  - Configure local Git LFS for this file only; verify its pointer and object hash.
- Acceptance criteria:
  - Real CPU inference returns eleven finite probabilities from the trained model.
  - Missing LFS data fails clearly without random fallback.
  - Clone instructions retain class order/config; original artifacts remain untouched.
- Out of scope: Retraining, test evaluation, manuscript/slides, hosting deployment, commit/push or history rewrite.
- Open questions: None.
- Verification: Artifact/hash checks, demo regressions, live local API prediction and LFS inspection.

## Execution Prompt

Complete the approved work above. Follow the repository rules, preserve prior work, verify the result, update status and create `completion.md`. Do not commit or push.
