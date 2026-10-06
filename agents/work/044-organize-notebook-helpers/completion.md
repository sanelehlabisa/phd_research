# Completion

- Status: Done
- Summary: Moved notebook-only helper modules into an importable `notebooks/utils/` package and updated their callers.
- Changes:
  - Relocated notebook config/data/diagnostics/display/model/suite/workflow helpers, Kinetics loaders, and Colab bootstrap from `src/` into `notebooks/utils/`, using concise filenames (`config.py`, `data.py`, `workflows.py`, etc.).
  - Updated all four modular notebooks, helper imports, setup checks, focused tests, and README paths; left the controlled AAD notebook and reusable CLI modules in `src/`.
  - Kept the ticket 045 NumPy bootstrap behavior intact at its new path.
- Verification:
  - Focused notebook and bootstrap tests: 102 passed; 2 expected local NVML warnings.
  - All four modular notebooks parsed as JSON and their code cells compiled as Python.
  - Fresh-process NumPy/bootstrap helper import passed in the implementation virtual environment.
  - `src.train`, `src.evaluate`, and `src.dataset` `--help` checks passed; `src.experiments --list-models` passed.
  - `git diff --check` passed.
- Remaining issues: The optional `--list-plan` check hit an existing controlled-plan validation mismatch (`controlled epochs must be 24, got 64`); its configuration was left untouched. No experiment config or run artifacts were changed.
