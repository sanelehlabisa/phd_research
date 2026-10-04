# Task Prompt

- Ticket: `034-colab-requirements-install`
- Status: Done
- Aim: Make Colab install the implementation's complete, strictly versioned direct runtime dependencies from `requirements.txt` on every setup run.
- Scope: Paper 001 implementation requirements, package metadata, Colab notebook setup, dependency documentation, and focused verification.
- Changes:
  - Replace the frozen transitive package list in `requirements.txt` with only the directly required runtime packages, each pinned to an explicit version.
  - Include every runtime dependency imported by the implementation or required by the Colab workflow, including `torchmetrics` and `kagglehub`.
  - Keep development and transitive packages out of the runtime requirements unless the implementation directly needs them.
  - Align `pyproject.toml` runtime dependencies with the direct dependency set where applicable.
  - Replace the notebook's hard-coded missing-package installer with an unconditional `python -m pip install -r requirements.txt` call from the implementation directory.
  - Preserve the notebook's existing uncommitted metadata removal and all unrelated work.
  - Update the implementation README with the Colab dependency behavior and any compatibility caveat.
- Acceptance criteria:
  - `requirements.txt` contains only complete, direct, explicitly pinned runtime dependencies.
  - The Colab setup cell installs that file every time and no longer maintains a second dependency list.
  - `torchmetrics` imports successfully after installation.
  - PyTorch, TorchVision, and CUDA availability are checked after installation so an incompatible A100 environment fails clearly before experiments.
  - Notebook JSON remains valid and the existing metadata-only user change is preserved.
  - Relevant implementation tests and dependency consistency checks pass.
- Out of scope: Running experiments, changing models or research protocols, opening the test split, or updating manuscript results.
- Open questions: None.
- Verification: Validate notebook JSON; compare imports against declared direct dependencies; run the requirements installation in an isolated environment when practical; import runtime packages; run focused tests; inspect the final diff and `git diff --check`.

## Execution Prompt

Complete ticket `034-colab-requirements-install`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, preserve the notebook's existing metadata removal, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
