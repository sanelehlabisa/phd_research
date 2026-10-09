# Task Prompt

- Ticket: `071-evidence-guided-search-corrections`
- Status: Draft
- Disposition: Superseded by [073](../073-focused-shape-search-run-all/prompt.md); retained for history, do not execute separately.
- Aim: Correct shortlist tie handling and retain the strongest observed architecture in the new search.
- Scope: Shared validation ranking, ticket 069's refinement/profile/reporting, associated tests and guides.
- Changes:
  - Use verified correct/total counts to compare equal-weight validation accuracy across resolutions and seeds without floating-point tie noise; retain lower mean loss, fewer parameters and stable name as tie-breaks.
  - Version new ranking/selection provenance. Preserve archived 068 selections; never silently relabel previously frozen or tested models. If exact evidence is unavailable, report it rather than inventing counts.
  - Add fixed control `[16,24]` to 069's refinement, at all three resolutions under its common new training/sampling recipe. Keep shallow models eligible; deduplicate exact matches.
  - Raise the pre-deduplication upper budget from 156 to 159 jobs, including at most 21 refinement jobs. Keep confirmation/ablation bounds unchanged; do not pool old 128-epoch scores into 069.
  - Keep complete result reporting, per-resolution scores, selected/actual epochs and optimization-failure warnings. Leave final comparison settings unchanged.
- Acceptance criteria:
  - The archived 424/480 tie ranks `[32,16]` before `[16,16,16]` by lower loss in a new audit result; the original saved selection stays byte-for-byte intact.
  - Genuine small accuracy differences are not erased by display rounding; unequal partition sizes and equal-weight resolution/seed aggregation behave correctly.
  - `[16,24]` has three scheduled control jobs and can enter the custom shortlist; counts, cache identities and provenance remain deterministic.
- Out of scope: Real training/testing, source resplitting, more grid expansion, notebook-output changes, manuscript updates, commit/push.
- Open questions: None; these are bounded corrections based on the October 9 evidence.
- Verification: Unit tests for exact ties, genuine differences, loss/parameter/name tie-breaks, missing/tampered evidence and archived-selection preservation; staged dry-run/deduplication tests; existing ranking and handoff regressions; `git diff --check`.

## Execution Prompt

Complete ticket `071-evidence-guided-search-corrections`. Follow `AGENTS.md`,
`agents/rules.md` and `agents/config.md`. Make the listed ranking and control
changes without rewriting historical evidence. Verify the result, update this
status and create `completion.md` from `agents/templates/completion.md`.
Do not commit or push.
