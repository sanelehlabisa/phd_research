# Completion

- Status: Done
- Summary: One validated AAD JSON drives a staged, validation-only comparison.
- Changes:
  - Added `aad_staged_experiments.json` and `--study-config` to the existing runner;
    `--list-plan` lists all jobs without model allocation or dataset access.
  - Defaults: seven models over two seeds (14 screen runs), then four separate
    input/weight-decay variants of the frozen validation winner (8 runs).
  - Added scratch `swin3d_t`; native PaperConvLSTM is separately labelled,
    opt-in and excluded from the screen ranking.
  - Persisted generated commands/configs, progress, split/input identity,
    parameters/runtime, selection and seed mean/standard deviation summaries.
  - Retained old plan APIs for historical notebooks; Kinetics settings untouched.
- Verification:
  - Full pytest suite: 187 passed; additive counts, invalid schemas/overrides,
    fixed budgets, multi-seed validation selection and test-lock guards covered.
  - All seven declared models completed tiny CPU forward checks without weight
    downloads; Swin has 27,858,929 parameters for 11 classes under pinned
    PyTorch 2.9.1 / Torchvision 0.24.1. Native topology passed a full-shape meta
    check (512,197,467 parameters), not a memory-intensive real training run.
  - Four-job synthetic-video study completed screen then ablation, reusing the
    existing runner and preserving test lock.
  - Listing, Black, Python parsing and diff checks passed. All notebook,
    historical-run and manuscript hashes unchanged.
- Remaining issues: Full AAD/GPU experiments are intentionally unrun; 22 jobs
  can take substantial time. Native topology may exceed available memory.
