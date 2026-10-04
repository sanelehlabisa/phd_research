# Task Prompt

- Ticket: `040-kinetics-interest-filter`
- Status: Done
- Aim: Try a small Kinetics activity subset through the existing modular notebooks.
- Scope: Paper 001 shared notebook configuration/data helpers, tests and guides; retain the pending display/recovery fix.
- Context: Saved VDD validation results (66.04% accuracy, 50% macro recall) suggest majority-class prediction; switching datasets is a diagnostic, not a promised fix.
- Input: `sanelehlabisa/kinetics-400-dataset/versions/1`; verified filename metadata in [dataset-inventory.json](dataset-inventory.json).
- Changes:
  - Define one hardcoded `CLASSES_OF_INTEREST`: existing AAD/VDD class names plus the five Kinetics folders below. Keep original, separate activity labels; no violent/non-violent remapping.
  - Apply this list only to Kinetics: intersect with available folders, print matched/skipped names, and tolerate absent interests. Fail clearly if fewer than two classes or insufficient samples for stratification remain.
  - Download/cache only matching Kinetics files, preserving class folders and showing progress/size. Never silently download the full 16.5 GB dataset; reuse supplied local roots without downloading.
  - Set shared selection to `kinetics-subset` for notebooks 01–04; retain the one-value switch back to VDD. Leave VDD classes and controlled AAD configuration unchanged.
  - Keep seed 42, 70:15:15 stratification, model candidates, preprocessing and epoch budgets unchanged. Use all matching clips without new balancing, caps or augmentation changes.
  - Key Kinetics manifests/provenance by version and matched classes; preserve old manifests/runs and reject incompatible checkpoints. Print train/validation class counts and majority-class baselines.
  - Preserve saved notebook outputs and the original reference notebook. Keep the local positional-video-path and save-before-display fixes.
  - Update guides and completion record; commit the agreed changes only after implementation and verification, never during task setup. No dataset/checkpoint artifacts.
- Acceptance criteria:
  - Version 1 matches `headbutting` (22), `slapping` (13), `punching_person__boxing_` (14), `hugging` (16), `shaking_hands` (22): 87 clips / 91,112,992 bytes.
  - Missing interests are skipped; unrelated classes are neither downloaded nor loaded; labels/head match the resulting subset.
  - All four workflows follow shared selection; validation alone selects models/checkpoints, and the frozen compatible winner gates final test access.
  - Existing display/recovery tests pass, and saved notebook outputs remain intact.
- Out of scope: Full Kinetics training, binary relabeling, architecture/optimizer tuning, real GPU training here, manuscript claims, and accuracy guarantees. This tiny subset is not surveillance-domain evidence; its held-out estimates will be noisy.
- Open questions: None; implementation approved, with subsequent approval to push master after verification.
- Verification: Mocked download/cache and filtering tests (including missing/zero/one matches), multiclass synthetic-video workflows, manifest/checkpoint isolation, test locks, display recovery, output preservation, full pytest, formatting and `git diff --check`.

## Execution Prompt

Complete ticket `040-kinetics-interest-filter`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Implement only the listed changes, preserve saved notebook outputs and the pending display fix, keep test data locked until the frozen final evaluation, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`. Commit only after implementation and verification.
