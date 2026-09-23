# Journal Paper: Abnormal Activity Recognition

- Status: Aligning comparison baselines before controlled experiments
- Target submission: December 2026
- Manuscript: [`manuscript/main.tex`](manuscript/main.tex)
- Code and detailed experiment plan: [`implementation/README.md`](implementation/README.md)

The paper studies lightweight ConvLSTM-based abnormal human-activity recognition
from surveillance video. Earlier width experiments made `64-32-16-64` a useful
candidate, but not a proven optimum. The revised study will compare a faithful
paper baseline with narrow stacked ConvLSTM models before selecting a reference
architecture for ablation.

## Evidence plan

1. Implement the published ConvLSTM topology faithfully and add configurable
   stacked ConvLSTM depth, width, and kernel sizes.
2. Keep the paper baseline separate from the new pooled stacked family, and align
   the 3D-CNN comparison registry before training comparisons.
3. Make splits, seeds, metrics, checkpoint selection, and outputs reproducible.
4. Screen a small, predeclared model set using validation performance, parameter
   count, and runtime—not test results.
5. Run one-factor ablations on the selected reference, then confirm the frozen
   model on the second dataset.
6. Export versioned tables and plots before changing manuscript claims.

## Tasks

- [ ] `009-define-ablation-config` — deferred; its WIP is stashed and its schema
  must be revised after ticket 010.
- [x] `010-prepare-architecture-search` — added the faithful paper baseline,
  sequence-returning ConvLSTM, explicit layer specifications, and lightweight
  stacked model.
- [x] `011-simplify-model-api` — kept one ConvLSTM layer plus the paper and
  custom models, removed obsolete architectures, and migrated every caller.
- [ ] `012-audit-and-align-baselines` — **Ready:** verify the published
  ConvLSTM and distinguish the source paper's 3D ResNet-50/101/152 comparisons
  from this study's `r3d_18`, `mc3_18`, and `r2plus1d_18` baselines.
- [ ] `013-make-splits-reproducible` — seed the full pipeline and create fixed,
  stratified, group-aware split manifests.
- [ ] `014-fix-selection-and-metrics` — correct metrics, restore the best
  validation checkpoint, and keep test data out of model selection.
- [ ] `015-standardize-run-artifacts` — save complete, versioned run evidence and
  add a CPU smoke run.
- [ ] `016-connect-ablation-config` — revise and resume ticket 009 so one config
  controls the stable model, input, and training schema.
- [ ] `017-shortlist-architecture-results` — audit historical runs and predeclare
  a small candidate set for controlled confirmation.
- [ ] `018-confirm-architecture-candidates` — compare the shortlist and approved
  baselines under one protocol.
- [ ] `019-run-reference-ablations` — run approved one-factor comparisons on the
  selected reference model.
- [ ] `020-validate-second-dataset` — evaluate the frozen configuration on VDD.
- [ ] `021-aggregate-ablation-evidence` — produce paper-ready tables, figures,
  uncertainty, efficiency comparisons, and error-analysis inputs.
- [ ] `022-rewrite-experimental-results` — revise the experiment and discussion
  section using only verified outputs.
- [ ] `023-align-paper-claims` — align the abstract, contributions, methods,
  limitations, and conclusion with the final evidence.

## Planned results-section flow

1. Experimental Setup
2. Comparison and Ablation Protocol
3. Main Comparative Results (AAD, then VDD)
4. Ablation Studies
5. Error Analysis
6. Efficiency and Deployment Trade-offs
7. Discussion and Limitations
