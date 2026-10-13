# Task Prompt

- Ticket: `078-aad-ablation-figures`
- Status: Ready
- Aim: Make the AAD architecture/input tradeoffs clear in a small set of research figures, with tables as supporting detail.
- Scope: Paper 001 report figures/CSV-derived plotting and concise report guidance; use verified 077/076 outputs when available. Preserve the current figure and source data.
- Changes:
  - Use parameter count on a logarithmic x-axis and validation accuracy on y; small panels for each frame-size/FPS condition, with identical axes and keys. Label the selected and compact candidates and retain poor/collapsed results.
  - Try intuitive layer-count glyphs: one tally stroke per layer and a crossed tally for five. Preview readability at paper size; keep a clear legend and consistent mapping. Retain the current shapes as a readable fallback if tally markers overlap.
  - Add a compact controlled-ablation figure: depth 1–5 at width 64; three-layer flat widths 16/48/64/96/128; and early/middle/late widening against [64,64,64]. Hold input/recipe/seed fixed within every curve. Show matched training/validation gaps where supported.
  - Distinguish measured saturation, underfitting, overfitting and optimization failure. Plot condition variation as input sensitivity; do not label it seed uncertainty or generalization.
  - Save publication-size SVG/PDF/PNG and source CSV with provenance. Make figures the main results narrative; keep the top-ten table concise and all configurations in supplementary CSV. Manuscript insertion waits for ticket 062 and verified final results.
- Acceptance criteria:
  - Markers/legends remain legible at paper size; every point traces to a saved configuration and correct partition.
  - Condition panels and controlled curves use consistent mappings and avoid combining changed factors; absent/failed results are disclosed, not invented or hidden.
  - Raw data and original figures are preserved; exported figures, captions, links and `git diff --check` pass.
- Out of scope: New training, altered scores, unsupported turning-point/overfitting claims, manuscript edits before results are verified, commit or push.
- Open questions: `None`; implement after the new evidence is available, or preview marker choices on the preserved search data first.
- Verification: Reconcile plot rows with CSV/receipts, check marker mapping and common axes, visually inspect exports at paper size and run `git diff --check`.

## Execution Prompt

Complete the approved work above. Follow `AGENTS.md`, `agents/rules.md`, and
`agents/config.md`. Verify acceptance criteria, mark Done and create
`completion.md` from `agents/templates/completion.md`. Do not train, commit,
push, or change the manuscript in this ticket.
