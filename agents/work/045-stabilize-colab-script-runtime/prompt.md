# Task Prompt

- Ticket: `045-stabilize-colab-script-runtime`
- Status: Done
- Depends on: None; ticket 044's helper relocation remains a separate follow-up.
- Aim: Make the experiment notebook reliable when exported to a Python script and run in one Colab cell on an A100.
- Scope: Paper 001 notebook 04, its Colab dependency bootstrap, pinned requirements only if needed, focused tests, and concise setup guidance.
- Changes:
  - Keep notebook 04 as the source of truth; do not add a separately maintained runner or commit an exported copy.
  - Diagnose and prevent the reported NumPy `_blas_supports_fpe` import failure by validating the actual importable NumPy stack, not package metadata alone; choose a compatible pin or safe repair only when evidence supports it.
  - Run dependency setup before repository imports. If a repair changes already-loaded binary packages, stop with one concise, actionable restart instruction; otherwise continue in the same cell.
  - Keep startup output concise: retain essential checkout/GPU and dataset summary, suppress per-archive cache-reuse messages, and preserve download progress, errors, and final results.
  - Keep ordinary CLI scripts and the separate controlled AAD workflow unchanged.
- Acceptance criteria:
  - A clean Python subprocess can import NumPy, `numpy.testing`, and the notebook's required dependencies after bootstrap without the reported `AttributeError`.
  - A detected broken/mixed install is repaired or reported before experiment imports, with at most one clear restart instruction and no long traceback from the known failure.
  - The notebook remains exportable as plain Python and its Colab one-cell workflow continues through preparation and the existing experiment path.
  - Startup no longer prints per-archive cache-reuse listings; essential GPU, dataset totals, download progress, failures, and final results remain visible.
  - Focused bootstrap/notebook tests pass; notebook JSON and Python cells validate; existing CLI smoke checks pass; `git diff --check` is clean.
- Out of scope: A new runner script, helper relocation from ticket 044, changing experiment candidates/budgets, changing the dataset, or running a full GPU experiment.
- Open questions: `None`.
- Verification: Focused `pytest`; isolated-process NumPy/dependency import smoke test; notebook JSON/export validation; relevant CLI `--help` checks; `git diff --check`. Live A100 execution is a user-side check.

## Execution Prompt

Complete ticket `045-stabilize-colab-script-runtime`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the approved changes, verify every acceptance criterion, set the ticket status to `Done`, and create `completion.md` from `agents/templates/completion.md`.
