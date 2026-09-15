# Completion

- Status: Done
- Summary: Moved both manuscripts into the numbered paper workspace and updated repository guidance for manuscripts and future implementations.
- Changes:
  - `papers/001-journal-abnormal-activity-recognition/`: Moved the journal README and manuscript, renamed its entry file to `manuscript/main.tex`, and added the unified ticket queue.
  - `papers/002-review-abnormal-activity-recognition/`: Moved the review manuscript, renamed its entry file to `manuscript/main.tex`, and added a concise paper README.
  - `README.md`: Added the numbered paper index, workspace structure, and current build commands.
  - `AGENTS.md`, `agents/rules.md`, and `agents/config.md`: Added the mixed manuscript/code layout and concise README-maintenance rule.
  - `.gitignore`: Added local per-paper implementation exclusions.
  - `agents/work/007-create-numbered-paper-layout/prompt.md`: Advanced the ticket from `Ready` through `Approved` and `In progress` to `Done`.
- Verification:
  - Confirmed the old `starter_journal/` and `review_paper/` paths are absent and both numbered papers contain their README and manuscript entry point.
  - SHA-256 hashes matched for both manuscripts, both bibliographies, and all three journal figures before and after the moves.
  - Journal `main.tex` built successfully to a 14-page PDF in `/tmp/phd-journal-build.0GX7zc`.
  - Review `main.tex` built successfully to a 3-page PDF in `/tmp/phd-review-build.QAFBb2`; its empty bibliography warning is expected because the scaffold has no citations.
  - `git diff --check` passed, historical tickets 001 through 006 were unchanged, and generated manuscript artifacts remained ignored.
  - Confirmed the separate implementation repository and its preserved working changes were untouched.
- Remaining issues: Ticket `008-import-journal-implementation` is next. Repository and local workspace renaming remain deferred until the consolidated structure is verified.
