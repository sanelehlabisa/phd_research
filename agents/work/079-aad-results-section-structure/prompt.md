# Task Prompt

- Ticket: `079-aad-results-section-structure`
- Status: Done
- Aim: Replace only Paper 001's outdated experimental results/discussion with the verified AAD search-to-comparison study.
- Scope: Only `Experimental Results and Discussion` in `papers/001-journal-abnormal-activity-recognition/manuscript/main.tex`; necessary baseline metadata in `manuscript/references.bib`; dated copies of existing report figures in `manuscript/figures/`; ticket records and concise root/paper README status.
- Evidence: `implementation/reports/2026-10-16/README.md`, `verification.json`, `search_results.csv`, `architecture_ranking.csv`, `resolution_checks.csv`, `comparison_results.csv`, `comparison_per_class.csv`, `test_confusion_matrices.json` and preserved evidence/receipts. No new training or test evaluation.
- Coordination: This revised ticket consolidates the requested main-text portions of Draft [080](../080-aad-search-figure-and-appendix/prompt.md)–[082](../082-aad-error-analysis-and-synthesis/prompt.md); do not execute their stale recipes separately or mark them Done. Additional appendix/figure work remains deferred. Later [090](../090-configurable-methodology-and-baseline-citations/prompt.md) handles methodology; [062](../062-update-manuscript-after-aad-results/prompt.md) retains broader abstract/introduction/conclusion alignment.
- Changes:
  - Keep the section concise with a short introduction and four subsections: experimental setup (including search design) → architecture-search results and selection handoff → comparative evaluation (including metrics) → discussion and confusion analysis. The owner's minor editorial follow-up removes the repeated shortlist table, retains the useful depth/width graph, uses `ConvLSTM[filters]` and author-led reference labels, and adds the raw confusion-matrix equation. Keep the numerical top five in the comparison table only.
  - State AAD's 1,069 clips/11 classes, fixed 748/160/161 clip-level split, seed 42, recorded A100/software, Adam, batch 32, no augmentation/weight decay, scheduler and validation checkpoint rule. Distinguish epoch caps, minimums and patience from actual completed epochs; do not claim actor/scene-independent splitting.
  - Explain that the search asks how recurrent depth, filters and width placement affect the accuracy/parameter trade-off: 32 architectures, depths 1–5, widths 16–128, fixed 3×3 kernels, 8/16 frames and 8/16 FPS, giving 128 jobs at 32×32 (cap/minimum/patience 128/64/24). Do not imply all shapes or kernel sizes were tested.
  - Reuse the existing all-128-point search graph and matched capacity-control graph without changing data/plotting code. Caption the compressed accuracy scale explicitly if using `search_capacity.png`; `capacity_controls.png` supplies matched depth/width context. Explain observed local saturation/decline and late-wide performance, not a universal turning point or causal proof of overfitting; sixteen-frame training scores also deteriorated and 108/128 search jobs reached the cap.
  - Report the actual numerical top five from four-condition validation ranking: [64,64,128], [96,96,96,96], [128,128,128], [128,64,64], [32,64]. Explain validation-only selection of eight frames/16 FPS, five subsequent 64px checks, then fresh training of all ten models at that shared input. Only finalists were checked at 64px; do not describe a full two-resolution search.
  - Replace legacy result tables with exactly ten verified test rows: those five customs plus input-adapted PaperConvLSTM, R3D-18, MC3-18, Swin3D-T and Swin3D-S. State scratch initialization, cap/minimum/patience 256/128/32 and LR 0.001 except Swin 0.00001 after training-only checks; freeze validation selections before test. Do not claim equal per-family tuning or exact reproduction of the source-paper protocol.
  - Use the owner's 19 October approval of macro precision/recall/F1 for this manuscript display, superseding the older no-macro planning preference for this scope only. Independently derive each macro score from all 11 saved class scores/counts; retain accuracy, exact parameter counts and loss. Keep short column titles, explain macro averaging below the table, bold column maxima/minimum parameters/minimum loss, not model names; omit latency. Preserve original pooled exports/contracts unchanged.
  - Cite one verified source beside each baseline, reusing existing keys: `vrskova2022` for adapted PaperConvLSTM; `tran2018closer` for both R3D-18/MC3-18; `liu2022videoswin` for both Swin sizes. Shared family sources are sufficient. Only fill verified missing BibTeX metadata; no duplicate entries or reliance on private `annote` claims.
  - After comparison, reuse the selected [64,64,128] model's saved test confusion matrix. Explain 160/161 correct, Hijack → Terrorism, 1,336,459 parameters, parameter savings and the smaller [32,64] alternative. Limit conclusions to tested AAD configurations; disclose single seed, small clip-level test set, short first-window coverage and baseline/input limitations. Do not claim significant superiority, universal optimum or deployment reliability.
- Acceptance criteria:
  - Only the named results/discussion section and necessary supporting assets/metadata/records change; abstract, introduction, related work, methodology and conclusion remain untouched.
  - The introduction and four subsections follow the requested order without a separate shortlist table; all 11 classes and 128 search points are retained, the confusion-matrix equation is defined, and exactly ten comparison rows match the verified split, frozen choices and test counts.
  - Macro calculations, bolding and parameter reductions independently reconcile; the text distinguishes validation search, 64px checks and final test evidence. No old VDD score/split or unsupported optimum/overfitting claim is reused.
  - Every baseline has one resolving primary-paper citation; existing citation keys/private notes are preserved and no duplicate bibliography entries are introduced.
  - LaTeX builds with figures, citations and cross-references resolved; inspect output for table/figure readability. Clearly record the intentionally deferred inconsistency in other sections rather than silently rewriting them.
- Out of scope: Training/retesting, model or split/seed selection, code/config/notebook/raw-evidence changes, new plotting pipelines/full appendix, whole-paper edits, methodology implementation, commit or push.
- Open questions: None. Complete October 16 evidence resolves the former pending-artifact and reserved-anchor questions; the actual numerical top five is fixed. The owner approved execution on 19 October 2026.
- Verification: Reconcile report/evidence and independently compute macro metrics; verify citation metadata at the primary sources below; compare section-boundary diffs and figure bytes; run `latexmk -pdf main.tex` from `manuscript/`, inspect the PDF, check undefined labels/citations and run `git diff --check`.

## Baseline sources

- `vrskova2022`: [Vrskova et al., Sensors 22(8), 2946 (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9028541/), DOI `10.3390/s22082946`.
- `tran2018closer`: [Tran et al., CVPR 2018, pp. 6450–6459](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html).
- `liu2022videoswin`: [Liu et al., CVPR 2022, pp. 3202–3211](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html).

## Execution Prompt

Execute ticket `079-aad-results-section-structure` exactly as written in
`agents/work/079-aad-results-section-structure/prompt.md`. Follow `AGENTS.md`,
`agents/rules.md`, and `agents/config.md`. Make only the approved changes,
verify every acceptance criterion, set the ticket status to `Done`, and create
`completion.md` from `agents/templates/completion.md`. Do not train, commit or push.
