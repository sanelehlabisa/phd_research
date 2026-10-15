# Task Prompt

- Ticket: `089-colab-campaign-readiness`
- Status: Done
- Aim: Make one thin experiment notebook and matching Python export safely run the revised study through existing modular scripts.
- Scope: Existing AAD experiment notebook/export, minimal workflow-helper integration, smoke tests and concise run commands. Do not build another trainer.
- Dependencies: [088 pooled outputs](../088-pooled-metrics-everywhere/prompt.md), [087 fixed-64px search](../087-expand-frame-depth-search/prompt.md), and [083 comparison preparation](../083-comparison-only-recovery/prompt.md) must be verified before declaring full readiness. 084's diagnosis is complete; a successful Swin3D-T diagnostic does not establish Swin3D-S convergence or approve a final family-specific recipe.
- Changes:
  - Point a clearly named notebook and matching Python setup/run export at a separately versioned fast JSON derived from 087: 32 customs, 32x32, 8/16 frames, 8/16 FPS, 128 search jobs capped at 128 epochs (minimum 64/patience 24 retained). Preserve 087/077 profiles and evidence; never reuse their incompatible scores.
  - Reuse existing clone/fast-forward setup, dependency/bootstrap checks, GPU validation, public dataset resolution and the reviewed split. Run modular commands/import their existing orchestration; keep notebook logic minimal.
  - Show the actual dataset/input/seed, resolved model families and learning rates, job count, full-step memory preflight, measured workload estimate and tqdm progress. Never silently shrink inputs/skip models or claim a runtime guarantee.
  - Before expensive comparison, require a recorded, owner-approved recipe and short training-only learnability checks for both transformers at the intended input/batch. Do not automatically transplant either the failed common LR 0.001 or the successful Swin3D-T-only LR 0.00001 to every model. Keep all models from scratch unless separately approved and labelled.
  - After complete validation-only search, pass the verified five finalists and selected frames/FPS to the same modular runner. First train those five from scratch at 64x64, cap/minimum/patience 128/64/24, validation only; save their 64px ranking alongside their 32px search order. Keep all five regardless of rank changes, with no extra tuning/accuracy threshold. Then freshly train five customs, adapted paper model, R3D-18, MC3-18 and Swin3D-T/S at the same 64px input, cap/minimum/patience 256/128/32. Freeze all checkpoints/best custom, then guarded test/confusion/examples. No test-based tuning or arbitrary fallback shortlist.
  - Invoke 087's existing results-only backup path before/after input blocks and comparison stages, on completion/handled failure. Preflight owner-authorized Drive access; retain a verified local ZIP, clearly report failed delivery, and keep checkpoints until evaluation completes. Do not promise protection from abrupt VM deletion.
- Acceptance criteria:
  - Notebook/export parity and a synthetic Run All traverse new JSON -> search -> verified shortlist -> five validation-only resolution checks -> ten-model comparison -> freeze -> test/confusion/examples -> compact backup, without a second training implementation.
  - Missing dependencies/recipe/evidence/Drive access, bad hashes, stale modules and genuine dataset/split mismatches stop clearly before expensive work; no false "ready" status.
  - New saved/reported results follow 088; old campaign values/protocols remain unchanged. Smoke success is not claimed as real A100 results or confirmed Drive durability.
- Out of scope: Real research training/testing during implementation, unapproved recipe/pretraining changes, streaming, unrelated notebook/helper cleanup and manuscript edits.
- Approved faster revision (2026-10-15): 32px search + five 64px finalist checks + ten 64px comparisons = 143 trainings, plus the two bounded transformer sanity checks. Rank transfer is measured only for shortlisted models; do not claim every architecture was searched at 64px. Epoch caps are not guaranteed runtime or convergence.
- Approved update (2026-10-15): Swin3D-T/S use LR 0.00001, other models LR 0.001, all from scratch. Both transformers must pass 128 training-only updates at the selected input/common batch (finite gradients/updates, lower loss and 100% accuracy on a fixed two-class training subset); failure saves diagnostics and stops. This is a sanity gate, not evidence of convergence.
- Delivery update: Mount Drive with owner authorization first; retain a visible dummy text file and verify its contents before training. Automatically save small, verified results-only ZIPs after completed jobs and tests, including every search/comparison architecture, resolved hyperparameters, metrics/confusion and provenance; exclude weights/videos/cache. Preserve full evidence locally. No runtime or Drive durability guarantee.
- Open questions: None. Owner approved implementation and commit/push after verification; Drive authentication remains interactive before launch.
- Verification: Notebook/export parsing/parity, JSON/list-plan, no duplicate training logic, synthetic end-to-end/handoff and backup failure checks, unchanged old profiles/artifact hashes, `git diff --check`. Real Colab execution remains owner-run work.

## Execution Prompt

Execute ticket `089-colab-campaign-readiness` exactly as written here after its
dependencies are verified. Follow `AGENTS.md`, `agents/rules.md`, and
`agents/config.md`. Make only approved changes, verify every acceptance
criterion, mark `Done`, and create `completion.md` from the template. Do not
launch real search/comparison/test evaluation. The latest owner instruction authorizes commit/push after verification.
