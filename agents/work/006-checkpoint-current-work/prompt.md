# Task Prompt

- Ticket: `006-checkpoint-current-work`
- Status: Done
- Aim: Create local rollback checkpoints for the completed roadmap work before restructuring the repositories.
- Scope:
  - `agents/work/004-starter-journal-experiment-roadmap/`
  - `agents/work/005-separate-paper-code-roadmaps/`
  - `agents/work/006-checkpoint-current-work/`
  - `starter_journal/README.md`
  - `/home/it-software/Desktop/projects/2026/convlstm_abnormal_human_activity_recognition/README.md`
  - Git staging and local commits in both repositories
- Changes:
  - Do not revise the approved README content or any research file.
  - In the implementation repository, stage and commit only `README.md` with the message `Document journal experiment roadmap`.
  - Leave `models/meta.json`, `src/experiments.py`, all untracked experiment results, and all untracked curve images unstaged and unchanged.
  - In the PhD repository, finish this ticket record, then stage and commit only the completed ticket 004, ticket 005, ticket 006, and `starter_journal/README.md` with the message `Checkpoint journal migration roadmaps`.
  - Do not stage ignored LaTeX artifacts such as `texput.log`.
  - Do not push either commit.
- Acceptance criteria:
  - The implementation README is committed in a local commit containing no other file.
  - The implementation repository retains its pre-existing modified and untracked experiment work unchanged and unstaged.
  - The PhD repository has one local commit containing the completed roadmap files and no unrelated files.
  - The PhD repository is clean after its commit.
  - Neither repository is pushed.
- Out of scope:
  - Moving or renaming files and directories.
  - Updating repository structure, rules, paths, or README content.
  - Importing Git history or implementation files.
  - Committing implementation code, metadata, results, curves, datasets, checkpoints, or environments.
- Open questions: None.
- Verification:
  - Inspect `git status --short --branch` before and after each commit.
  - Inspect each staged diff with `git diff --cached --check` and `git diff --cached --name-only` before committing.
  - Inspect both new commits with `git show --stat --oneline --decorate HEAD` and `git show --format= --name-only HEAD`.
  - Confirm the implementation repository still lists the preserved modified and untracked experiment files but not `README.md`.
  - Confirm the PhD repository is clean and remains ahead of `origin/master` without pushing.

## Execution Prompt

Execute ticket `006-checkpoint-current-work` exactly as written in `agents/work/006-checkpoint-current-work/prompt.md`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the approved changes, verify every acceptance criterion, set the ticket status to `Done`, create `completion.md` from `agents/templates/completion.md`, and create the two approved local commits without pushing.
