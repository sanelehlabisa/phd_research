# Task Prompt

- Ticket: `036-kinetics-subset-audit`
- Status: Ready
- Aim: Define a small, surveillance-relevant subset of the available Kinetics-400 copy without downloading or training the full dataset prematurely.
- Scope: Dataset audit and planning for `sanelehlabisa/kinetics-400-dataset`; concise Paper 001 roadmap documentation.
- Changes:
  - Inspect dataset version, size, directory and annotation structure, available classes, clip counts, formats, corrupt files, and download/cache requirements.
  - Identify a balanced candidate subset of clearly observable classes relevant to human-activity recognition and distinct enough for a learnability check.
  - Estimate download size, preprocessing time, GPU time, class balance, and storage for the subset before approving any training.
  - Define deterministic train/validation splits and a configuration that reuses the existing ConvLSTM pipeline.
  - Return the proposed class list and resource estimate for user approval before implementing or running training.
  - Record that any additional future dataset requires its link, labels, size, and intended research role before a task is created.
- Acceptance criteria:
  - The audit is reproducible and names the exact Kaggle dataset version.
  - The proposed subset has explicit inclusion criteria, class counts, estimated resources, and no test access.
  - No claim treats a general Kinetics subset as surveillance-domain evidence.
  - No dataset download or model training begins without approval of the subset and budget.
- Out of scope: Full Kinetics-400 training, opportunistic class selection from results, replacing AAD/VDD, final transfer learning, or manuscript changes.
- Open questions: None; exact classes are an output of the audit and require approval before a later implementation ticket.
- Verification: Save dataset metadata and class inventory; validate the proposed split/configuration; review resource estimates and `git diff --check`.

## Execution Prompt

Complete ticket `036-kinetics-subset-audit`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Perform audit and planning only, do not train or download the full dataset, return the proposed subset for approval, update this status, and create `completion.md` from `agents/templates/completion.md`.
