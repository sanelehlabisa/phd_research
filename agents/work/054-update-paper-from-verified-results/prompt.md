# Task Prompt

- Ticket: `054-update-paper-from-verified-results`
- Status: Ready
- Aim: Update Paper 001 from verified experimental and final-evaluation evidence.
- Scope: Paper 001 manuscript, bibliography, result tables/figures and README; consolidates roadmap items 029–031.
- Changes:
  - Wait for ticket 053's analysis, ticket 026's frozen selection and ticket 027's final test report; require separate verified evidence for any VDD claims.
  - Trace each reported value/table/figure to exact runs, configs, seeds and the correct partition.
  - Update setup, comparison/ablation results, discussion, limitations and conclusions; align the abstract and claims with observed evidence.
  - Report seed variation, parameter counts and runtime; distinguish practical baselines, native topology and exploratory diagnostics.
  - Preserve the author's voice, use LaTeX labels/references, and verify citations.
- Acceptance criteria:
  - Every quantitative claim is reproducible from saved evidence; validation results are never presented as final test results.
  - Missing experiments are stated as limitations, not invented or implied complete.
  - Manuscript tables, figures and narrative agree; LaTeX builds with references/citations resolved.
- Out of scope: Running new experiments, changing model selection after test, unsupported claims, or committing generated PDFs.
- Open questions: None for ticket setup; verified result artifacts are prerequisites.
- Verification: Cross-check values against source artifacts; build `latexmk -pdf main.tex` in the manuscript directory; check unresolved references and `git diff --check`.

## Execution Prompt

Complete ticket `054-update-paper-from-verified-results` only after its evidence prerequisites are met. Follow repository rules, preserve historical results, use verified artifacts, rebuild the manuscript, update status and create `completion.md`. Do not fabricate missing results.
