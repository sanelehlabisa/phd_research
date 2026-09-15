# Task Prompt

- Ticket: `003-starter-bib-private-notes`
- Status: Done
- Aim: Mark all private paper-reading notes in the starter journal bibliography as non-rendered annotations without changing their content.
- Scope:
  - `starter_journal/references.bib`.
  - This ticket's `prompt.md` and `completion.md`.
- Changes:
  - Replace the standalone opening `abstract` block with an exact copy of the two commented private `annote` templates at the top of `review_paper/references.bib`.
  - Rename the other 22 `abstract` field keys in bibliography entries to `annote`.
  - Preserve every reading-note value, citation key, and bibliographic metadata field exactly.
- Acceptance criteria:
  - All 23 original `abstract` occurrences have been handled: one standalone template is replaced and 22 entry fields are renamed.
  - No uncommented `abstract` field remains in `starter_journal/references.bib`.
  - Exactly 22 uncommented `annote` entry fields remain, with their original note content unchanged.
  - The opening commented templates exactly match those in `review_paper/references.bib`.
  - The active `IEEEtran` bibliography style does not render the private `annote` fields.
  - No LaTeX file or unrelated bibliography content is modified.
- Out of scope:
  - Editing `starter_journal/starter_journal.tex` or any other LaTeX file.
  - Correcting spelling, formatting, claims, metadata, duplicate entries, or other pre-existing bibliography problems.
  - Editing `review_paper/references.bib`.
  - Committing or pushing the completed ticket.
- Open questions: None.
- Verification:
  - Run `git diff --check`.
  - Count original and resulting `abstract` and `annote` field keys.
  - Compare the opening template with `review_paper/references.bib`.
  - Review a focused diff to confirm only the opening template and field names changed.
  - Confirm the installed `IEEEtran.bst` contains no handler for `annote` and that private note markers do not occur in the generated bibliography.
  - Confirm all `.tex` files and `review_paper/references.bib` are unchanged.

## Execution Prompt

Execute ticket `003-starter-bib-private-notes` exactly as written in `agents/work/003-starter-bib-private-notes/prompt.md`. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the approved changes, verify every acceptance criterion, set the ticket status to `Done`, and create `completion.md` from `agents/templates/completion.md`.
