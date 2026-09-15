# Starter Journal Paper

This paper studies ConvLSTM-based abnormal human activity recognition. Training
and evaluation are maintained in the separate
[implementation repository](https://github.com/sanelehlabisa/convlstm_abnormal_human_activity_recognition).
The final paper must identify the exact implementation commit used for its
results.

## Current state

Earlier runs made **Proposed Lightweight ConvLSTM (64-32-16-64)** the leading
candidate. It remains provisional until the controlled experiments are
complete. The present Experimental Results and Discussion section also mixes
setup, comparisons, results, and interpretation and therefore needs a clearer
flow.

## Paper tickets

- [x] `003-starter-bib-private-notes` — keep reading notes out of the published
  bibliography.
- [x] `004-starter-journal-experiment-roadmap` — establish the experiment and
  paper dependencies.
- [x] `005-separate-paper-code-roadmaps` — separate paper work from
  implementation work.
- [ ] `006-rewrite-experimental-results` — rewrite the results section using
  verified outputs. **Blocked until the implementation experiment queue is
  complete.**
- [ ] `007-align-paper-claims` — update all sections affected by the new
  results. **Blocked by ticket 006.**

The next work is
[`EXP-001-repair-experiment-runner`](https://github.com/sanelehlabisa/convlstm_abnormal_human_activity_recognition#journal-experiment-roadmap)
in the implementation repository.

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

Baselines and ablations should be presented separately. Report variability,
make causal claims only when supported by controlled comparisons, and include
only figures that explain an important result.

## Sections affected after the results

Ticket 007 must align the abstract, introduction contributions, and conclusion
with the verified findings. It must also update any methods, limitations,
tables, figures, or claims whose accuracy depends on the final experiment
protocol or results.
