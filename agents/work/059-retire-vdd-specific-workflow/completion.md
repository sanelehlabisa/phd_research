# Completion

- Status: Done
- Summary: Retired the VDD-only learnability command while preserving VDD as an optional dataset using generic class-folder discovery.
- Changes:
  - Removed `src.vdd_diagnostic`, its notebook entry point, and its VDD-specific tests.
  - Moved downloaded-root resolution and balanced training-subset selection into dataset-neutral helpers.
  - Changed the configured VDD notebook dataset to infer all classes from its folders; preserved existing runs and local data.
- Verification:
  - `.venv/bin/python -m pytest -p no:cacheprovider -q` — 217 passed, 2 environment warnings.
  - Notebook JSON validated; the retired diagnostic cells and executable references are absent.
  - Search found no VDD diagnostic module imports or entry points in active implementation code.
  - `git diff --check` passed.
- Remaining issues: None.
