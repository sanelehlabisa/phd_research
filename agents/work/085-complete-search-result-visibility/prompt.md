# Task Prompt

- Ticket: `085-complete-search-result-visibility`
- Status: Done
- Aim: Make every saved search configuration visibly inspectable without clipping, filtering or rerunning search.
- Scope: Paper 001's existing report builder, derived figures/gallery, reporting tests and concise report guidance; no experiment-code changes.
- Owner clarification (15 October 2026, latest): Graphs, not per-model performance tables, zoom subplots or arrows. Keep all real points; enlarge the crowded accuracy range using a clearly disclosed scale. Only layer-count keys belong at the bottom. Each panel has Best, Middle, Worst model keys in that order, with two-decimal accuracy and no "tied" label.
- Evidence: [14 October report](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-14/README.md), `search_results.csv`, `search_evidence_check.json` and `build_preview.py`. All 64 results already exist: 16 architectures x two sizes x two FPS values, not 64 distinct architectures. Current plots contain all points; nearby points overlap and tables/gallery use collapsed blocks.
- Changes:
  - Reuse the existing builder and verified inputs; plot all 64 measurements in four 16-model input-condition panels. Keep the existing CSV as the numeric reference rather than displaying per-model performance tables in the new report or beside the graphs.
  - Keep every point, including the 12.5% failed-learning result, at its exact measured coordinates. Use the same monotone piecewise y-axis on all panels: compress below 90%, retaining true percentage ticks, all values and a labelled transition. No inset plots, omitted scores or unlabelled scale changes.
  - Outline the best, worst and real model nearest the median accuracy separately within each panel. Give each a named role key inside that panel, ordered Best, Middle, Worst; keep only the original layer-marker colours/shapes in the bottom legend. Resolve illustrative ties by fewer parameters then run ID; disclose this rule and keep the original across-condition shortlist/ranking unchanged. Do not invent an average-model point or call a panel winner globally optimal.
  - Keep links to all 64 original curves/confusion pairs. Do not require all 128 large images to appear inline; make the complete row list and run navigation visible without expanding hidden tables.
  - Generate revised derived outputs in a new dated report folder, linking the unchanged source evidence and previous review. State validation-only/single-seed limits and keep the five-finalist ranking unchanged. Link this evidence to later manuscript ticket [080](../080-aad-search-figure-and-appendix/prompt.md); do not edit the paper now.
- Acceptance criteria:
  - Exactly 64 unique configuration IDs occur in primary plot metadata: 16 per condition, with 128 original-image links, no side ID/performance tables and no top-N/full-panel axis filtering.
  - Each highlight identifies a real, correctly selected per-panel model; the scale is invertible, disclosed and retains every observation. No arrow/inset or role key replaces the layer-count keys. Raw evidence/selection hashes remain unchanged.
  - Markdown/GitHub and PNG/SVG outputs are readable; overlap is disclosed and the complete values remain accessible.
- Out of scope: New training, reranking, changing search values, transformer fixes, manuscript edits, a separate plotting framework, commit or push.
- Open questions: None.
- Verification: Assert IDs 1–64 exactly once per primary plot, 16 per condition, real per-panel highlight choices, unchanged coordinates, invertible scale, image-link existence and raw/selection hashes. Check removed tables/arrows/insets, complete bounds, layer-count-only bottom legend and correct panel-key order/accuracy; visually inspect exports and run `git diff --check`.

## Execution Prompt

Execute ticket `085-complete-search-result-visibility` exactly as written here.
Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only approved
changes, verify each acceptance criterion, set the status to `Done`, and create
`completion.md` from `agents/templates/completion.md`. Do not train, commit or push.
