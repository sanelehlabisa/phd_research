# Task Prompt

- Ticket: `064-colab-first-run-setup`
- Status: Ready
- Aim: Make first-run setup consistent and less confusing across Paper 001 Colab notebooks while preserving any restart required for safe dependency changes.
- Scope: Colab notebook setup cells and matching exported scripts, `implementation/notebooks/utils/colab_bootstrap.py`, its tests, and concise setup guidance.
- Changes:
  - Audit every active Colab notebook/export and route dependency checks through the shared bootstrap helper where applicable.
  - Skip package installation and restart when the required versions and fresh-process import check already pass.
  - When pins must change or NumPy needs repair, stop before model imports and show one short, consistent instruction explaining the one-time runtime restart and rerun; avoid an alarming traceback or ambiguous “reset” message.
  - Do not automatically restart/disconnect a runtime, replace a working CUDA PyTorch build unnecessarily, or change dataset/training/experiment behavior.
  - Record that Colab may require one restart after actual package changes; do not promise restart-free setup when the environment needs repair.
- Acceptance criteria:
  - All active notebook setup paths use the same tested dependency behavior and compatible pinned requirements.
  - A matching healthy environment skips pip and restart; an environment needing repair requests restart once; rerunning after restart proceeds without reinstalling or requesting another restart.
  - Setup confirms the attached GPU after restart and fails concisely if CUDA/dependencies remain unhealthy.
  - Automated tests cover healthy, repaired, and rerun states; notebook training and experiment cells remain unchanged apart from setup handling.
- Out of scope: Rewriting notebook workflows, changing model/training configs, changing experiment inputs, or promising a zero-restart flow when Colab packages change.
- Open questions: `None`.
- Verification: Run bootstrap and notebook setup tests; inspect every active notebook and export for consistent handling; perform a Colab A100 smoke check if available and record the observed first-run/restart behavior.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
