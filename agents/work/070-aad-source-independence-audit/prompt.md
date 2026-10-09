# Task Prompt

- Ticket: `070-aad-source-independence-audit`
- Status: Draft
- Disposition: Superseded by [073](../073-focused-shape-search-run-all/prompt.md); retained for history, do not execute separately.
- Aim: Resolve suspected shared-source leakage before treating the AAD split as publication evidence.
- Scope: AAD dataset provenance, current split manifests, existing audit helpers/tests and evidence guides.
- Changes:
  - Follow up the [October 9 review](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-09-multiresolution-search-review.md): 18 apparent filename groups cross partitions. Treat names as hypotheses, not verified source IDs.
  - Inspect available source metadata, preprocessing and original clips; record the evidence mapping clips to recordings/scenes. Do not decode held-out test clips for model selection.
  - Report verified overlap, unknown identities, duplicate evidence and independent source counts per class. Check whether a grouped three-way split can actually represent every class.
  - Save an explicit audit disposition tied to dataset/split hashes, with verified findings separate from unresolved cases.
  - If overlap is confirmed or identities remain unresolved, keep final testing blocked and propose a versioned protocol decision for user approval. Do not silently resplit or overwrite historical evidence.
- Acceptance criteria:
  - The reported filename examples are reproducible; authoritative identity is not inferred solely from a suffix.
  - Known overlap cannot be reported as independent generalization; insufficient per-class groups are visible.
  - Existing manifests, notebook outputs, results and ticket 069 remain intact.
- Out of scope: Training/testing, dataset replacement, implementing a new split, manuscript claims, commit/push.
- Open questions: None for the audit; any new split or data-collection decision requires a separate approval.
- Verification: Synthetic audit tests for shared, distinct and unknown sources; reconcile counts to the fixed manifest; inspect provenance links and unchanged evidence hashes; `git diff --check`.

## Execution Prompt

Complete ticket `070-aad-source-independence-audit`. Follow `AGENTS.md`,
`agents/rules.md` and `agents/config.md`. Audit and document source independence
without changing the split or running experiments. Verify the result, update this
status and create `completion.md` from `agents/templates/completion.md`.
Do not commit or push.
