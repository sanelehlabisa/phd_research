# Task Prompt

- Ticket: `062-update-manuscript-after-aad-results`
- Status: Ready
- Aim: Refocus Paper 001's methods, results, and discussion on the completed AAD study without unsupported VDD claims.
- Scope: Paper 001 manuscript and paper README, using only verified outputs from the new experiment runs.
- Coordination: Draft tickets [079](../079-aad-results-section-structure/prompt.md), [080](../080-aad-search-figure-and-appendix/prompt.md), [081](../081-aad-ten-model-results-comparison/prompt.md) and [082](../082-aad-error-analysis-and-synthesis/prompt.md) detail the requested results revision; do not duplicate their work or execute their unresolved scope. This ticket retains whole-paper alignment after verified results.
- Changes:
  - Use the requested figure-led search narrative with the full search table in the appendix, followed by the distinct ten-model comparison table; follow 079–082 after their approval.
  - Remove VDD-specific methods, results, and claims from the paper; do not replace them with unsupported claims.
  - Update abstract, contributions, limitations, and conclusion only where the verified AAD results require it.
- Acceptance criteria:
  - Every reported value traces to a saved run artifact; no old VDD number is presented as a new result.
  - The methods accurately describe dataset selection, class set, splits, seeds, validation selection, and the locked test protocol.
  - All citations resolve and the manuscript builds successfully.
- Out of scope: New experiments, fabricating missing metrics, or claiming generalization to VDD without a new configured VDD run.
- Open questions: `None`. Execute only after tickets 057–061 and the AAD search/comparison produce verified artifacts.
- Verification: Trace each table/claim to run artifacts, check bibliography/cross-references, and build with `latexmk -pdf main.tex` from the manuscript directory.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only the listed changes, verify the result, update this status, and create `completion.md` from `agents/templates/completion.md`.
