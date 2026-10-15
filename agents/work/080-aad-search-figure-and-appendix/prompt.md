# Task Prompt

- Ticket: `080-aad-search-figure-and-appendix`
- Status: Draft
- Aim: Present verified search capacity trends, five finalists and the complete appendix table (requested Group 5).
- Scope: Paper 001 search-results manuscript content, appendix and derived figure/table assets; provenance under `implementation/reports/YYYY-MM-DD/`.
- Dependencies: [079](../079-aad-results-section-structure/prompt.md), verified search artifacts and selection receipt. Reuse [078](../078-aad-ablation-figures/prompt.md)'s plotting work; do not build a competing plotting pipeline.
- Changes:
  - Create one main figure with a subplot per tested frame resolution (currently 32×32 and 64×64), log trainable-parameter x-axis, validation-accuracy y-axis and consistent layer-count colours/markers.
  - Keep FPS/sequence-length conditions visibly distinct; never silently combine them or cherry-pick each architecture's best condition. Use common axes, highlight validation-selected trade-off candidates and annotate sparingly.
  - Put the complete campaign search table in the appendix: configuration, filters, depth, resolution, sequence length, FPS, parameters, validation loss/accuracy, pooled precision/recall/F1, selected epoch and seed, where available. Link per-class counts/metrics and confusion evidence; disclose missing/failed/reused entries.
  - Reconcile pooled metrics with verified matching-checkpoint predictions; document full-partition aggregation and per-class zero-division policy. Pooled P/R/F1 equal accuracy here. Do not compute macro averages, relabel legacy scores or fill missing values from another campaign (088).
  - Add a short trend discussion followed by the five finalists and an appendix reference. Explain saturation, under-capacity, diminishing returns or overfitting only where supported by matched conditions and train/validation evidence; distinguish optimization collapse.
  - Record validation ranking and parameter efficiency without using test. Preserve numerical top five separately from any reserved-anchor finalist list; report a turning point only if observed, otherwise state that none was established.
- Acceptance criteria:
  - Every plotted/tabulated value traces to the correct configuration, partition and checkpoint; all search conditions are accounted for.
  - Figure is legible at paper size, complete table stays out of the main text, and shortlist wording matches the saved selection rule.
  - Raw evidence and existing figures remain unchanged; no global-optimality claim or invented metric.
- Out of scope: New training, changing the running search/ranking, post-test selection, final comparison analysis, commit or push.
- Open questions: Confirm strict numerical top five versus the current reserved `[96,64,64]` anchor rule before manuscript selection wording is finalized. No protocol change is authorized by this ticket.
- Verification: Reconcile all rows/points with receipts, independently check pooled/per-class calculations and ranking, inspect figure/table PDF exports, build LaTeX and run `git diff --check`.

## Execution Prompt

After resolving the selection question, receiving verified evidence and owner
approval, complete the work above. Follow `AGENTS.md`, `agents/rules.md`, and
`agents/config.md`. Verify, update status and create `completion.md` from
`agents/templates/completion.md`. Do not train, commit or push.
