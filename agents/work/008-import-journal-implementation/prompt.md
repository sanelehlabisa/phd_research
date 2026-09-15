# Task Prompt

- Ticket: `008-import-journal-implementation`
- Status: Done
- Aim: Import the journal implementation and its committed Git history into paper 001 without transferring uncommitted or local-only experiment work.
- Scope:
  - `/home/it-software/Desktop/projects/2026/convlstm_abnormal_human_activity_recognition` as the read-only import source
  - `papers/001-journal-abnormal-activity-recognition/implementation/`
  - `papers/001-journal-abnormal-activity-recognition/README.md`
  - `README.md`
  - `agents/work/008-import-journal-implementation/`
  - Git history, staging, and local commits in the PhD repository
- Changes:
  - Import the implementation repository's local `master` at commit `6683e649ca5684e64109d81ee2e5362005856114` into `papers/001-journal-abnormal-activity-recognition/implementation/` using `git subtree` without squashing so its committed history remains reachable.
  - Do not import a nested `.git` directory or configure the old repository as a permanent dependency.
  - Import only content committed at `6683e64`; do not copy the source repository's modified `models/meta.json`, modified `src/experiments.py`, untracked grid-search JSON, untracked curves, datasets, checkpoints, outputs, `.venv`, or other local-only files.
  - Leave the separate implementation repository and all of its working changes untouched as a temporary backup.
  - Rewrite only the imported implementation `README.md` to be concise and current:
    - state that it supports paper 001 and link to `../README.md`;
    - describe the ConvLSTM/custom-model and 3D-CNN experiment code briefly;
    - mention only the AAD and VDD datasets currently used by this paper;
    - remove outdated claims about unused UCF-Crime, UCF50/UCF101, Kinetics, canonical 14-class mapping, or public readiness;
    - retain a short setup/usage section based on commands the imported project actually supports;
    - retain the controlled experiment protocol and validity problems;
    - replace `EXP-*` names with the consolidated global ticket sequence and mark `009-transfer-local-experiment-work` as `Next`.
  - Update the paper 001 README to link `implementation/README.md`, mark ticket 008 checked, and mark ticket 009 as next.
  - Update the root README paper index so paper 001 states that its implementation has been imported and local experiment work is pending transfer.
  - Keep the root and paper READMEs concise and avoid duplicating detailed implementation guidance outside the implementation README.
  - Allow the standard subtree import commit plus one local documentation/ticket completion commit; do not push either commit.
- Acceptance criteria:
  - `papers/001-journal-abnormal-activity-recognition/implementation/` contains the 79 files tracked at source commit `6683e64`, subject only to the approved imported README revision.
  - Commit `6683e64` remains reachable from the PhD repository history, and no nested `.git` exists under `implementation/`.
  - The imported README is concise, links paper 001, discusses only AAD and VDD, and shows ticket 009 as next.
  - The paper and root READMEs accurately show that implementation import is complete and local experiment transfer is next.
  - Imported Python source compiles without syntax errors.
  - No manuscript, bibliography, figure, result, model metadata, or research claim is changed.
  - The separate implementation repository has exactly the same branch, commit, modified files, and untracked files as before this ticket.
  - The PhD repository is clean after the approved local commits, is ahead of its remote, and is not pushed.
- Out of scope:
  - Transferring or committing any uncommitted implementation work or local experiment output.
  - Moving datasets, checkpoints, environments, or ignored generated outputs.
  - Repairing the experiment runner or defining new model architectures.
  - Deleting, renaming, pushing, or archiving the old implementation repository.
  - Renaming the PhD workspace directory or GitHub repository before the consolidated workspace is verified.
  - Editing LaTeX, BibTeX, figures, results, or research claims.
- Open questions: None.
- Verification:
  - Record `git status --short --branch`, `git rev-parse HEAD`, and the tracked file count in both repositories before import.
  - Inspect the subtree import and verify `git merge-base --is-ancestor 6683e64 HEAD` succeeds.
  - Compare the imported tracked paths and blobs with source commit `6683e64`, allowing only the approved README difference.
  - Confirm no `.git` exists below the imported implementation directory and no dataset, environment, checkpoint, or source working change was copied.
  - Run `python -m compileall -q src` from the imported implementation directory and confirm generated caches remain ignored.
  - Run `git diff --check`, inspect staged file lists before the documentation commit, and inspect both new local commits.
  - Confirm the PhD repository is clean and ahead of `origin/master` without pushing.
  - Recheck the separate implementation repository and confirm its status and commit are unchanged.

## Execution Prompt

Execute ticket `008-import-journal-implementation` exactly as written in `agents/work/008-import-journal-implementation/prompt.md`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the approved changes, verify every acceptance criterion, set the ticket status to `Done`, create `completion.md` from `agents/templates/completion.md`, and create the approved local commits without pushing.
