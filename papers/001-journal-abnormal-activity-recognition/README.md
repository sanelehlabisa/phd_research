# Journal Paper: Abnormal Activity Recognition

- Status: AAD validation run saved; controlled comparisons and final test evaluation pending
- Target submission: December 2026
- Manuscript: [`manuscript/main.tex`](manuscript/main.tex)
- Code and detailed experiment plan: [`implementation/README.md`](implementation/README.md)

The committed AAD run reached 96.25% validation accuracy; it is one run, not a
final result. The Kinetics-600 notebook workflow remains a separate diagnostic.
See the [implementation guide](implementation/README.md) for current commands.

The paper studies lightweight ConvLSTM-based abnormal human-activity recognition
from surveillance video. Earlier width experiments made `64-32-16` a useful
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

## Next tasks

- [ ] [049 AAD dataset auto-resolution](../../agents/work/049-aad-dataset-auto-resolution/prompt.md)
- [ ] [050 overall micro metrics](../../agents/work/050-overall-micro-metrics/prompt.md)
- [ ] [051 save run prediction examples](../../agents/work/051-save-run-prediction-examples/prompt.md)
- [ ] [052 JSON-controlled AAD experiments](../../agents/work/052-json-controlled-aad-experiments/prompt.md)
- [ ] Later, after final results: prepare a small Gradio demo using the selected model for presentation.

## Tasks

- [ ] [`047-larger-colab-training`](../../agents/work/047-larger-colab-training/prompt.md) — In progress as a separate Kinetics notebook task; not a prerequisite for AAD paper experiments.
- [x] [`042-merge-expanded-kinetics-suite`](../../agents/work/042-merge-expanded-kinetics-suite/prompt.md) — Integrated both laptops' work: Kinetics-600 (>2,000 videos before splitting), 14-model screen, spatial/weight-decay comparisons and two-seed confirmation within an eight-hour deadline. 115 local tests passed; real Colab run pending.
- [x] [`043-download-colab-run-artifacts`](../../agents/work/043-download-colab-run-artifacts/prompt.md) — Downloads one ZIP of the current suite's saved artifacts after completion or a handled time limit.
- [x] [`044-organize-notebook-helpers`](../../agents/work/044-organize-notebook-helpers/prompt.md) — Moved notebook-only helpers to `implementation/notebooks/utils/`; kept core CLI modules in `src/`.
- [x] [`045-stabilize-colab-script-runtime`](../../agents/work/045-stabilize-colab-script-runtime/prompt.md) — Added a fresh-process NumPy/import integrity check, targeted wheel repair, concise restart handling, and quiet cache reuse.
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
- [x] `018-shortlist-architecture-results` — marked historical outputs as
  non-comparable and predeclared six named custom candidates with manifest
  provenance.
- [x] `019-cloud-experiment-notebook` — added one guarded local/Colab/Kaggle
  workflow for preview, screening, confirmation, selected training, validation
  inspection, final evaluation, and artifact export.
- [x] `020-expand-controlled-experiments` — prepared the approved 11-model
  screen, practical-baseline confirmation, separate published topology, and
  one-factor weight-decay, augmentation, spatial-size, and sequence ablations.
- [x] [`021-optional-cloud-architecture-expansion`](../../agents/work/021-optional-cloud-architecture-expansion/prompt.md) — **Skipped:** retain the eleven-candidate screen; no optional five-model block.
- [x] `022-interactive-notebook-foundation` — simplified the Colab VS Code
  setup and displayed AAD metadata, balance, training frames, and augmentation.
- [x] `023-interactive-model-inspection` — added playable native, model-ready,
  and augmented clips, manifest-driven model tables, and one labelled
  random-weight GPU check.
- [x] `024-notebook-experiment-workflow` — added guarded single-stage execution
  and visual validation-only result inspection to the Colab notebook.
- [ ] `035-vdd-learnability-check` — **Blocked diagnostic:** verify that the current
  model and optimizer can overfit a tiny balanced VDD subset, then run one
  bounded VDD train/validation check before resuming expensive AAD experiments.
- [ ] `036-kinetics-subset-audit` — inventory the available Kinetics copy and
  propose a small relevant class subset and resource budget; do not train yet.
- [x] `037-modular-colab-notebooks` — split routine Colab work into independent
  dataset, model-inspection, training, and controlled-experiment notebooks while
  preserving the original workflow as the reference.
- [x] `038-notebook-prediction-endings` — ended each modular notebook with five
  partition-labelled examples and gated final test predictions behind validation
  selection and longer confirmation.
- [x] `039-vdd-run-all-notebooks` — shared VDD defaults, playable previews,
  live curves, validation-ranked screen and automatically guarded final testing;
  separate from the controlled AAD study.
- [x] [`040-kinetics-interest-filter`](../../agents/work/040-kinetics-interest-filter/prompt.md) — shared interests, Kinetics-only filtering and selective downloads; VDD results preserved, test access guarded.
- [x] [`041-learning-and-temporal-diagnostics`](../../agents/work/041-learning-and-temporal-diagnostics/prompt.md) — implemented and locally verified: gentle scheduling, temporal audit, 14-model screen, separate native topology and two-seed confirmation under an eight-hour cap. GPU run pending; existing Kinetics test is exploratory, not fresh paper evidence.
- [x] [`041-larger-kinetics-diagnostic`](../../agents/work/041-larger-kinetics-diagnostic/prompt.md) — added selective Kinetics-600 downloads, source-video grouped splits and a more-than-2,000 unique-clip gate. Actual archive count awaits the first Colab preparation.
- [x] [`046-separate-aad-screen-config`](../../agents/work/046-separate-aad-screen-config/prompt.md) — separated the fixed 24-epoch AAD screen from the local longer-training profile.
- [ ] [`025-run-controlled-aad-experiments`](../../agents/work/025-run-controlled-aad-experiments/prompt.md) — Draft/superseded; do not execute the older matrix. Use 052's updated plan.
- [ ] `026-train-selected-model` — confirm and freeze the selected architecture,
  input size, augmentation, and regularisation across both planned seeds.
- [ ] `027-evaluate-selected-model` — perform the one-time final AAD test and
  export complete metrics, confusion output, and playable prediction examples.
- [ ] `028-validate-second-dataset` — define and run the frozen VDD
  generalisation protocol without changing the selected AAD model.
- [ ] `029-aggregate-ablation-evidence` — produce paper-ready tables, figures,
  uncertainty, efficiency comparisons, and error-analysis inputs.
- [ ] `030-rewrite-experimental-results` — revise the experiment and discussion
  section using only verified outputs.
- [ ] `031-align-paper-claims` — align the abstract, contributions, methods,
  limitations, and conclusion with the final evidence.
- [ ] `032-assess-kinetics-transfer` — conditional only if frozen AAD/VDD
  evidence remains weak; define the exact Kinetics variant and use it for
  pretraining/transfer rather than silently replacing the 11-class task.
- [ ] `033-add-stateful-streaming-inference` — later add chunk-by-chunk stateful
  prediction to the custom model without changing the published baseline.

## Planned results-section flow

1. Experimental Setup
2. Comparison and Ablation Protocol
3. Main Comparative Results (AAD, then VDD)
4. Ablation Studies
5. Error Analysis
6. Efficiency and Deployment Trade-offs
7. Discussion and Limitations
