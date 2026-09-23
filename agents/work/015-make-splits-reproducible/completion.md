# Completion

- Status: Done
- Summary: Added one validated AAD split manifest and deterministic seeding for
  model smoke, training, evaluation, and comparison commands without training
  models or changing the manuscript.
- Changes:
  - Added shared Python, NumPy, PyTorch, CUDA, data-loader, worker, and
    augmentation seeding with deterministic PyTorch settings.
  - Added atomic creation and strict reuse validation for stratified clip-level
    manifests, then replaced independent random splits in all three consumers.
  - Added `splits/abnormal-activities-dataset_seed42.json`, covering all 1,069
    AAD clips as 748 training, 160 validation, and 161 test samples.
  - Recorded the exact manifest, split metadata, run seed, split seed, and
    determinism settings in run metadata and resolved configurations.
  - Updated runnable commands and repository guidance; ticket 016 is next.
- Verification:
  - Black left all six modified Python files unchanged; `py_compile` and Pylint
    error-only passed.
  - Parser checks confirmed seed `42`, optional manifests, and 70:15:15 defaults;
    model smoke uses the shared seeding helper.
  - Synthetic manifest checks confirmed stable membership under reordered input,
    changed membership for a different split seed, and clear rejection of seed,
    ratio, class-map, duplicate, omitted, unknown, and relabelled corruption.
  - A two-worker synthetic loader produced identical order and augmented tensors
    for seed `42`, while seed `2026` changed the stochastic sequence.
  - The real manifest validated with zero overlap and all classes in every split:
    Begging 93/20/20; Drunkenness 63/13/14; Fight 67/14/14; Harassment
    62/14/13; Hijack 54/12/11; Knife Hazard 70/15/15; Normal Videos 80/17/18;
    Pollution 80/17/17; Property Damage 67/14/15; Robbery 60/13/13; Terrorism
    52/11/11 (train/validation/test).
  - Three repeated consumers reconstructed identical indices, and temporary
    train, evaluate, and experiments run records contained the exact manifest
    hash and deterministic settings. Run seed `2026` reused split seed `42`.
  - Manifest JSON validation and tracking-eligibility checks passed.
- Remaining issues: None.
