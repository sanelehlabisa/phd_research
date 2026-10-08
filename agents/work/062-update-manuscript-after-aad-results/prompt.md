# Task Prompt

- Ticket: `062-update-manuscript-after-aad-results`
- Status: Ready
- Aim: Refocus Paper 001's methods, results, and discussion on the completed AAD study without unsupported VDD claims.
- Scope: Paper 001 manuscript and paper README, using only verified outputs from the new experiment runs.
- Changes:
  - Present custom ConvLSTM architecture search and model-family comparison as distinct result tables.
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
