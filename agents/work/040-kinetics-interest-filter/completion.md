# Completion

- Status: Done
- Summary: Notebooks 01–04 default to the five-class Kinetics diagnostic subset; one setting switches back to unchanged VDD classes.
- Changes:
  - Shared hardcoded interests; Kinetics-only exact intersection, skipped-name reporting, separate activity labels and clear insufficient-class/split failures.
  - Version-1 filename inventory and selective per-file downloads; cached reruns, partial-download recovery, safe paths and temporary host-mount bypass. Non-video `.part` entries excluded.
  - Version/class/seed-specific manifests, compatible-checkpoint checks and train/validation counts and majority baselines. No model, preprocessing or epoch-budget changes.
  - Retained positional video-display fix and save-before-display evaluation recovery. Updated notebook introductions/guides without changing code cells or saved outputs.
- Verification:
  - `python -m pytest tests -q`: 68 passed, including two-/five-class synthetic workflows, filtering/cache failures, incompatible checkpoints, locked tests and display recovery.
  - Real public Kaggle version-1 download: exactly 87 files / 91,112,992 bytes; split 60 train / 13 validation / 14 test. Cached rerun succeeded with network calls forbidden and identical manifest; video decoding forbidden throughout.
  - Live verification caught unrelated `.mp4.part` inventory entries; added the exclusion and a regression test.
  - Black, Python compilation, notebook code compilation, `pip check` and `git diff --check` passed.
  - All four notebooks' code-cell/output hashes unchanged. Reference notebook SHA-256 remains `cbe430531729e6d0444c783cea799b99467928ab5964a526adb58dad986b078b`; controlled AAD, model/training code and dependencies unchanged.
- Remaining issues:
  - No real GPU training or accuracy claim; 87 clips give noisy held-out estimates. Broader clip-quality/source-independence audit remains ticket 036.
  - Real download verification used a temporary cache outside Git; no datasets, checkpoints or generated runs are included in the commit.
