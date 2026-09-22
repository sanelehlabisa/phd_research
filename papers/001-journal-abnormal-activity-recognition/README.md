# Journal Paper: Abnormal Activity Recognition

- Status: Preparing controlled ablation experiments
- Target submission: December 2026
- Manuscript: [`manuscript/main.tex`](manuscript/main.tex)
- Code and detailed experiment plan: [`implementation/README.md`](implementation/README.md)

The paper proposes a lightweight ConvLSTM-based approach to abnormal
human-activity recognition from surveillance video. Earlier architecture runs
identified `64-32-16-64` as the selected lightweight ConvLSTM candidate, but the
new study must establish which changes caused its accuracy-efficiency trade-off.

## Evidence plan

1. Treat the 30+ historical runs as exploratory and shortlist only comparable
   architecture results.
2. Re-run 3–5 shortlisted ConvLSTM variants under one reproducible protocol.
3. Select the reference model by validation performance and parameter cost.
4. Change one factor at a time on that reference: input resolution, sequence
   length, widths, justified depth, augmentation, weight decay, and dropout.
5. Use AAD for screening and most ablations; use VDD to test the selected
   configuration's generalisation.
6. Export versioned tables and plots before changing reported manuscript claims.

## Tasks

- [ ] `009-define-ablation-config` — **Next:** add one validated experiment
  configuration and explicit CLI overrides.
- [ ] `010-make-splits-reproducible` — seed the full pipeline and create fixed,
  stratified, group-aware split manifests.
- [ ] `011-fix-selection-and-metrics` — correct metric accumulation, restore the
  best validation checkpoint, and keep test data out of model selection.
- [ ] `012-standardize-run-artifacts` — save complete, versioned run evidence
  and add a CPU smoke check.
- [ ] `013-expose-ablation-factors` — make focused architecture, input, and
  training factors configurable without duplicate scripts.
- [ ] `014-shortlist-architecture-results` — audit the historical runs and name
  3–5 candidates for controlled confirmation.
- [ ] `015-confirm-architecture-candidates` — re-run the shortlist and comparison
  models under the fixed protocol.
- [ ] `016-run-reference-ablations` — run the approved one-factor comparisons on
  the selected reference model.
- [ ] `017-validate-second-dataset` — evaluate the frozen configuration on VDD.
- [ ] `018-aggregate-ablation-evidence` — produce paper-ready tables, figures,
  uncertainty, efficiency comparisons, and error-analysis inputs.
- [ ] `019-rewrite-experimental-results` — revise the experiment and discussion
  section using only verified outputs.
- [ ] `020-align-paper-claims` — update the abstract, contributions, methods,
  limitations, and conclusion to match the final evidence.

## Planned results-section flow

1. Experimental Setup
2. Comparison and Ablation Protocol
3. Main Comparative Results (AAD, then VDD)
4. Ablation Studies
5. Error Analysis
6. Efficiency and Deployment Trade-offs
7. Discussion and Limitations
