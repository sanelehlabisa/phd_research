# Completion

- Status: Done
- Summary: One new Colab notebook/export runs the approved faster campaign through the existing runner, with verified results-only Drive backups.
- Changes:
  - Separate versioned JSON: 32 architectures x 8/16 frames x 8/16 FPS at 32px; 128 search jobs, cap/minimum/patience 128/64/24. Original 087/077 profiles remain unchanged.
  - Five fresh validation-only 64px checks retain all finalists and record rank transfer; ten fresh 64px comparisons use 256/128/32. Continuous experiment numbers 1-143; validation freeze precedes ten guarded tests and best-custom examples.
  - Approved scratch LR 0.00001 for Swin3D-T/S and 0.001 for others. Both transformers must pass bounded train-only checks at the selected input/common batch; failed checks save diagnostics and stop.
  - Authorize Drive first and verify a persistent visible text file. After each completed job/test, copy a verified JSON-only ZIP containing all search/check/comparison results, architectures, settings, counts and provenance; exclude weights/videos/cache. Uncompressed safety cap: 128 MiB.
  - Updated root/paper/implementation guides with the new entry point, backup limits and the uncertain 12-18-hour A100 estimate.
- Verification:
  - 173 targeted pytest checks passed: 107 campaign/profile/metrics/handoff checks and 66 transformer-diagnostic/bootstrap checks. The 12 fast new unit cases also passed on rerun after the final error-message compatibility correction.
  - New synthetic Run All completed 143 trainings, both transformer gates, ten frozen tests, compact backup verification and compatible resume without retraining/retesting. No research training or actual Drive authentication occurred.
  - Notebook/export source parity and AST parsing, the 128-job `--list-plan`, Black checks on ten changed Python files, and `git diff --check` passed.
  - A separate read-only check confirmed the completed legacy fixture rejects a changed learning rate with the expected frozen-recipe error.
  - All 179 previously tracked report/notebook files retain their initial byte hashes; old profiles and saved evidence are unchanged.
- Remaining issues:
  - One additional legacy end-to-end stress rerun was stopped under local memory pressure after its initial 64-search/ten-comparison/ten-test cycle. Its final repeated-resume assertions were not completed; the new campaign's full resume check passed. This is not a claim that the entire repository test suite passed.
  - Real A100 fit, transformer learnability, runtime and owner-authorized Drive delivery remain Colab checks. Results ZIPs preserve analysis, not checkpoint recovery after VM deletion. No new research scores or manuscript claims.
