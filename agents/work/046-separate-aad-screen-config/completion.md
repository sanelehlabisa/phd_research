# Completion

- Status: Done
- Summary: Separated the fixed AAD architecture-screen protocol from the local longer-training profile.
- Changes:
  - Added `aad_architecture_screen_reference.json` for the eleven-candidate 24-epoch screen and pointed the controlled plan to it.
  - Updated controlled-experiment tests and README guidance; ticket 021 remains skipped and ticket 025 now depends on this separation.
- Verification:
  - `tests/test_controlled_experiments.py`: 12 passed.
  - Safe `--list-plan` succeeded and reported 11 candidates, 24 screening epochs, 16 frames, `32x32`, and the validation-only protocol.
  - Hashes for the local `aad_screening_reference.json` and the older run artifacts matched before/after. The newer run's `config.json`, `resolved_config.json`, and `run.json` hashes also matched; its checkpoint and history files changed during verification while its manifest remained `running`.
  - `git diff --check` passed. No training, evaluation, download, or test-set access was performed.
- Remaining issues: The newer run's checkpoint/history continued changing during this task; these files were not edited or staged here and remain local.
