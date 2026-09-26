# Journal Paper: Abnormal Activity Recognition

- Status: Predeclaring architecture configurations before controlled runs
- Target submission: December 2026
- Manuscript: [`manuscript/main.tex`](manuscript/main.tex)
- Code and detailed experiment plan: [`implementation/README.md`](implementation/README.md)

The paper studies lightweight ConvLSTM-based abnormal human-activity recognition
from surveillance video. Earlier width experiments made `64-32-16-64` a useful
candidate, but not a proven optimum. The revised study will compare the audited
source-paper topology with narrow stacked ConvLSTM models before selecting a
reference architecture for ablation.

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

- [ ] `009-define-ablation-config` — superseded by ticket 017; its preserved WIP
  remains stashed and must not be applied to the current runners.
- [x] `010-prepare-architecture-search` — added the faithful paper baseline,
  sequence-returning ConvLSTM, explicit layer specifications, and lightweight
  stacked model.
- [x] `011-simplify-model-api` — kept one ConvLSTM layer plus the paper and
  custom models, removed obsolete architectures, and migrated every caller.
- [x] `012-audit-and-align-baselines` — verified the published ConvLSTM topology
  and separated the source paper's 3D ResNet-50/101/152 comparisons from this
  study's `r3d_18`, `mc3_18`, and `r2plus1d_18` practical baselines.
- [x] `013-fix-video-augmentation` — replaced augmented copies with optional,
  clip-consistent online augmentation and documented runnable AAD commands.
- [x] `014-standardize-run-artifacts` — separated model, training, evaluation,
  and comparison evidence into local timestamped run directories.
- [x] `015-make-splits-reproducible` — seeded the full pipeline and added a
  fixed, stratified 70:15:15 clip manifest.
- [x] `016-fix-selection-and-metrics` — added full-partition metrics, restored
  validation-selected checkpoints, and isolated final test access.
- [x] `017-connect-ablation-config` — connected one validated JSON schema to
  training and comparisons with explicit CLI overrides and checkpoint
  provenance.
- [ ] `018-shortlist-architecture-results` — **Next:** audit historical runs and
  predeclare a small candidate set for controlled confirmation.
- [ ] `019-confirm-architecture-candidates` — compare the shortlist and approved
  baselines under one protocol.
- [ ] `020-run-reference-ablations` — run approved one-factor comparisons on the
  selected reference model.
- [ ] `021-validate-second-dataset` — evaluate the frozen configuration on VDD.
- [ ] `022-aggregate-ablation-evidence` — produce paper-ready tables, figures,
  uncertainty, efficiency comparisons, and error-analysis inputs.
- [ ] `023-rewrite-experimental-results` — revise the experiment and discussion
  section using only verified outputs.
- [ ] `024-align-paper-claims` — align the abstract, contributions, methods,
  limitations, and conclusion with the final evidence.
- [ ] `025-add-stateful-streaming-inference` — later add chunk-by-chunk stateful
  prediction to the custom model without changing the published baseline.

## Planned results-section flow

1. Experimental Setup
2. Comparison and Ablation Protocol
3. Main Comparative Results (AAD, then VDD)
4. Ablation Studies
5. Error Analysis
6. Efficiency and Deployment Trade-offs
7. Discussion and Limitations
