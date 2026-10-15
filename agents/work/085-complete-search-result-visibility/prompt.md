# Task Prompt

- Ticket: `085-complete-search-result-visibility`
- Status: Ready
- Aim: Make every saved search configuration visibly inspectable without clipping, filtering or rerunning search.
- Scope: Paper 001's existing report builder, derived figures/tables/gallery, reporting tests and concise report guidance; no experiment-code changes.
- Evidence: [14 October report](../../../papers/001-journal-abnormal-activity-recognition/implementation/reports/2026-10-14/README.md), `search_results.csv`, `search_evidence_check.json` and `build_preview.py`. All 64 results already exist: 16 architectures x two sizes x two FPS values, not 64 distinct architectures. Current plots contain all points; nearby points overlap and tables/gallery use collapsed blocks.
- Changes:
  - Reuse the existing builder and verified inputs; present all 64 configuration rows visibly by default, grouped into four 16-run conditions. Include run ID, layer widths/depth, frame count/size/FPS, parameters, seed, validation accuracy/loss and available selected-epoch metrics. Mark unavailable fields, never infer them.
  - Keep every point in the four input-condition panels, including the 12.5% failed-learning result. Use full data bounds/padding and consistent legends; add stable run identifiers and a readable table/legend for overlapping points rather than altering scores or hiding rows. A zoomed view may only supplement the complete view.
  - Keep links to all 64 original curves/confusion pairs. Do not require all 128 large images to appear inline; make the complete row list and run navigation visible without expanding hidden tables.
  - Generate revised derived outputs in a new dated report folder, linking the unchanged source evidence and previous review. State validation-only/single-seed limits and keep the five-finalist ranking unchanged. Link this evidence to later manuscript ticket [080](../080-aad-search-figure-and-appendix/prompt.md); do not edit the paper now.
- Acceptance criteria:
  - Exactly 64 unique configuration IDs appear in the visible table and plots: 16 per condition, with 128 original-image links and no silent top-N/axis filtering.
  - Labels, points and tables agree with the existing CSV/receipts; all failures remain visible and raw evidence/selection hashes remain unchanged.
  - Markdown/GitHub and PNG/SVG outputs are readable; overlap is disclosed and the complete values remain accessible.
- Out of scope: New training, reranking, changing search values, transformer fixes, manuscript edits, a separate plotting framework, commit or push.
- Open questions: None.
- Verification: Assert IDs 1–64 exactly once per table/plot, 16 per condition, image-link existence, raw/selection hashes and unchanged numeric values; visually inspect exports and run `git diff --check`.

## Execution Prompt

Execute ticket `085-complete-search-result-visibility` exactly as written here.
Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`. Make only approved
changes, verify each acceptance criterion, set the status to `Done`, and create
`completion.md` from `agents/templates/completion.md`. Do not train, commit or push.
