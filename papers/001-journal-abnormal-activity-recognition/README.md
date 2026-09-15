# Journal Paper: Abnormal Activity Recognition

- Status: Experiments and results require controlled revision
- Target submission: December 2026
- Manuscript: [`manuscript/main.tex`](manuscript/main.tex)
- Implementation: Pending import into `implementation/`

Earlier runs made **Proposed Lightweight ConvLSTM (64-32-16-64)** the leading
candidate. It remains provisional until the controlled experiments are
complete.

## Tickets

- [x] `003-starter-bib-private-notes`
- [x] `004-starter-journal-experiment-roadmap`
- [x] `005-separate-paper-code-roadmaps`
- [x] `006-checkpoint-current-work`
- [x] `007-create-numbered-paper-layout`
- [ ] `008-import-journal-implementation` — **Next**
- [ ] `009-transfer-local-experiment-work` — blocked by 008
- [ ] `010-repair-experiment-runner` — blocked by 009
- [ ] `011-define-architecture-registry` — blocked by 010
- [ ] `012-screen-aad-architectures` — blocked by 011
- [ ] `013-run-focused-ablations` — blocked by 012
- [ ] `014-confirm-and-export-results` — blocked by 013
- [ ] `015-rewrite-experimental-results` — blocked by 014
- [ ] `016-align-paper-claims` — blocked by 015

## Planned results-section flow

1. Experimental Setup
2. Comparison and Ablation Protocol
3. Main Comparative Results
   - AAD results
   - VDD results
4. Ablation Studies
5. Error Analysis
6. Efficiency and Deployment Trade-offs
7. Overall Discussion and Limitations

Keep baselines separate from ablations, report variability, and make causal
claims only when supported by controlled comparisons. After the results are
verified, align the abstract, introduction contributions, conclusion, methods,
limitations, tables, figures, and affected claims.
