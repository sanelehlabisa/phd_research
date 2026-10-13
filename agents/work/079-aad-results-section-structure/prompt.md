# Task Prompt

- Ticket: `079-aad-results-section-structure`
- Status: Draft
- Aim: Restructure Paper 001's experimental section around the AAD-only search-to-comparison narrative (requested Group 4).
- Scope: Experimental section in `papers/001-journal-abnormal-activity-recognition/manuscript/main.tex` and its cross-references; related guides.
- Dependencies: Owner approval and verified campaign configuration/split; [080](../080-aad-search-figure-and-appendix/prompt.md)–[082](../082-aad-error-analysis-and-synthesis/prompt.md) supply results. [062](../062-update-manuscript-after-aad-results/prompt.md) retains wider methods/abstract/conclusion alignment.
- Changes:
  - Use this exact subsection order: computing environment; AAD dataset and split; training configuration/hyperparameters; evaluation metrics; architecture-search setup; search results; five-model selection; final model-family comparison; confusion matrix/class analysis; overall interpretation.
  - Describe configurable stacked ConvLSTM layers/filters, frame resolution, sequence length, sampling FPS, fixed settings, validation selection and locked test before the search figure. Call the search focused and evidence-guided, not exhaustive.
  - Read actual hardware/software, class counts, split, seeds, optimizer/scheduler, augmentation and training budgets from verified records; distinguish configured caps from epochs actually completed.
  - Use AAD only in this section; remove superseded VDD/legacy-result assertions without deleting historical evidence. Define multiclass accuracy and macro P/R/F1 explicitly, retaining per-class detail separately.
  - Keep the main narrative figure-led: search setup → figure → trends → five finalists → appendix-table reference → ten-model comparison/discussion → test confusion/class analysis → synthesis. Place the full search table physically in the appendix.
- Acceptance criteria:
  - Requested order and semantic labels/references are correct; no old score or split is silently reused as current evidence.
  - Protocol statements trace to the chosen campaign; unavailable measurements are marked pending, not inferred.
  - No invented result or unsupported optimum claim; LaTeX builds with references resolved.
- Out of scope: Training, runner/config changes, new model selection, historical artifact deletion, unrelated manuscript rewrites, commit or push.
- Open questions: Before execution, confirm whether to create a clearly pending scaffold before final artifacts arrive or wait for complete results. Ticket creation alone is not execution approval.
- Verification: Check headings and campaign provenance, audit links/labels/citations, run `latexmk -pdf main.tex` from `manuscript/`, inspect PDF and run `git diff --check`.

## Execution Prompt

After resolving the open question and approving this prompt, complete only the
work above. Follow `AGENTS.md`, `agents/rules.md`, and `agents/config.md`.
Verify acceptance criteria, update status and create `completion.md` from
`agents/templates/completion.md`. Do not train, commit or push.
