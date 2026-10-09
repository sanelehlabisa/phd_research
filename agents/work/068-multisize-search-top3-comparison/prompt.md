# Task Prompt

- Ticket: `068-multisize-search-top3-comparison`
- Status: Done
- Aim: Select three custom architectures across multiple resolutions, then run a fixed, longer eight-model comparison with complete downloadable evidence.
- Scope: Paper 001 AAD profiles, existing experiment/evaluation runners, cached-data path, active Colab notebook/export, artifact helper, tests and guides.
- Decisions:
  - Two stages only: architecture/input search, then final model comparison.
  - One fixed model seed `42` and the existing AAD split in both stages. No seed sweep or seed-2026 confirmation; label findings single-seed.
  - Prioritise resolution coverage: retain the existing 12 architectures, not the proposed four additions.
  - Keep downloaded `configs/experiments/temp/`, historical runs, saved notebook outputs and manuscript unchanged.
- Changes:
  - Search: run every architecture at `32x32`, `48x48` and `64x64`, always with 8 frames: exactly 36 training jobs, up to 128 epochs each.
  - Retain the current 12 stacks: `[16,16]`, `[16,16,16]`, `[24,24,24]`, `[32,32,32]`, `[32,16]`, `[32,16,16]`, `[16,32,16]`, `[16,16,32]`, `[8,16]`, `[24,16]`, `[16,24]`, `[16,16,16,16]`; keep 3x3 kernels and current heads.
  - Replace winner-only input checks and second-seed confirmation with the complete architecture-by-resolution matrix. No frame-count, augmentation, weight-decay or optimiser sweep.
  - Hold search data, split, seed, augmentation, optimiser/LR, regularisation, epoch cap and early-stopping rule fixed. Choose one memory-safe batch size before the search and retain it across every architecture/resolution.
  - Reuse bounded, configuration-keyed train/validation caches; never cache/decode test clips during search or selection. Record cache footprint and avoid stale inputs across resolutions.
  - Require all three resolutions for every candidate. Rank distinct architectures by equally weighted mean validation accuracy across resolutions, then mean loss, parameters and name. Save per-resolution scores/ranks and worst-case accuracy; resolution variation is not seed uncertainty.
  - Automatically export/import the top three architectures with their exact layers, heads, source runs and split/config hashes. Reject incomplete or incompatible selections; do not fall back to a placeholder custom model.
  - Comparison: train each of those three plus `paper_convlstm_published`, `r3d_18`, `mc3_18`, `swin3d_t` and `swin3d_s` exactly once from scratch: eight jobs, seed 42, up to 256 epochs.
  - Keep one comparison input for all eight: 50 frames at 50x50, batch 1. Preserve the current shared comparison optimiser/LR, weight decay, augmentation and validation-loss selection/stopping rules. No per-model tuning, seed repeats or search-checkpoint warm starts. Caps are budgets, not promises of that many completed epochs.
  - Freeze all eight validation-selected checkpoints, configurations and the best-custom identity before any test access. Evaluate each frozen model once on the same complete test partition; test results never select the best custom model or trigger retraining.
  - Export validation/test loss, accuracy, explicitly labelled micro precision/recall/F1, per-class metrics, confusion matrices and full prediction records for all eight. Retain parameter counts, weight-only model size versus full checkpoint bytes, training/evaluation timing definitions, histories/curves, actual epochs and provenance. Include a combined JSON/CSV comparison table.
  - For slides, save up to three correct and three incorrect playable test examples from the validation-selected best custom model, with source clip, true/predicted class, confidence and correctness. Report missing categories honestly; avoid test-based model reselection.
  - Persist every run's config, metrics, history, progress and selected checkpoint as it completes. Save incomplete status on handled failures; safely reuse verified completed work without silently repeating training/test evaluation.
  - ZIP-only delivery: include all completed/partial run evidence, group selections, exact split manifest, logs, checkpoints, tables and example videos. Exclude raw datasets/caches; avoid per-epoch checkpoint duplication.
  - Check available disk space before packaging; verify archive contents/integrity and report its size/path. Keep failures visible and provide an independent retry-download step that never retrains/retests. Do not claim a download succeeded merely because it was requested.
  - Keep notebook/export code aligned and search/comparison explicit. No Drive integration. Explain that local ZIPs are lost if the Colab runtime is deleted before download.
- Acceptance criteria:
  - Validated plan lists 36 search jobs, then eight comparison jobs and eight gated final test evaluations; no hidden confirmation or expansion.
  - Top-three ranking uses every declared resolution equally and is traceable to complete validation evidence. No global-optimum or unseen-data-generalisation claim follows solely from resolution testing.
  - All comparison models share the frozen protocol, restore validation-selected checkpoints and retain complete metrics/artifacts. The best-custom examples use that frozen model.
  - ZIP inventory covers every planned/partial run and selected checkpoint; space/integrity/download failures are actionable and cannot mark incomplete work successful.
  - Existing data, outputs and reference workflows are preserved; smoke/guard tests pass. The unarchived reported >90% result is context only, not paper evidence.
- Out of scope: Real Colab/GPU execution, new architectures/datasets/splits, temporal or optimiser sweeps, multiple seeds, recovering the prior session, Drive uploads, manuscript edits, commit/push.
- Open questions: None; the concrete matrix/ranking above was approved and implemented.
- Verification:
  - Test exact job counts, all-resolution ranking, missing-result rejection and correct top-three handoff.
  - Test cache isolation, fixed protocol, checkpoint/split compatibility, frozen-test guards and safe reruns.
  - Synthetic end-to-end search/comparison/evaluation; metric recalculation and example/source checks.
  - Test ZIP completeness/integrity, insufficient space, interrupted runs and retry-download without compute.
  - Focused/full CPU tests, notebook/export parity, local smoke, preserved-output hashes and `git diff --check`.

## Execution Prompt

Complete ticket `068-multisize-search-top3-comparison`. Follow this approved prompt,
`AGENTS.md`, `agents/rules.md` and `agents/config.md`. Implement and verify the two-stage
workflow, preserve existing artifacts and saved outputs, update status and create
`completion.md`. Do not run real Colab experiments or commit/push.
