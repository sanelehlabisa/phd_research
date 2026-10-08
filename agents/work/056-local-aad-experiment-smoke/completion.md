# Completion

- Status: Done
- Summary: Added flat one-JSON local and A100 experiment profiles with validation-only selection and a reusable selected training config.
- Changes:
  - `aad_local_smoke.json` runs two tiny custom models at 32×32 with 8 frames for 8 epochs; `aad_colab_a100.json` runs 32 combinations across four models, `frame_sizes: [32, 48]` and `num_frames: [8, 16]`, 16 epochs, and two weight-decay values. Both use four-epoch early-stopping patience.
  - Added flat-list grid parsing and one-argument `--config` execution in `src.experiments`.
  - Preserved `aad_staged_experiments.json` for the full two-seed study and notebook workflow.
  - Saved selected config/selection metadata and documented smoke, longer-training, and frozen-evaluation commands in the implementation README.
  - Added coverage for grid dimensions/count, fixed split and test lock, CLI listing, ranking, and selected-config output. Retained the plan-runner CWD fix and regression test.
- Verification:
  - Implementation pytest suite: `208 passed` (two expected NVML warnings in notebook workflow tests).
  - Local and A100 `--list-plan`: confirmed 2 and 32 configurations, with the A100 Cartesian product of the two frame-size and two frame-count values, and no training started.
  - `src.experiments --help`: confirmed `--config` and backward-compatible `--study-config` options.
  - `git diff --check`: passed.
- Remaining issues: Neither training profile was launched here. Run the two-job smoke locally first; the one-seed A100 screen requires later confirmation before paper claims.
