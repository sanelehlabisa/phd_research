# Journal Paper: Abnormal Activity Recognition

- Status: 075 implemented and locally verified. Existing evidence preserved; new AAD/A100 performance and final test evidence await execution.
- Target submission: December 2026
- Manuscript: [`manuscript/main.tex`](manuscript/main.tex)
- Code and detailed experiment plan: [`implementation/README.md`](implementation/README.md)

AAD is the primary dataset for the active study. VDD remains available as an optional dataset
choice in configuration; its existing run artifacts will be kept, but it is not
part of the current study. Do not treat the earlier validation result as final.
See the [implementation guide](implementation/README.md) for the current code
status and ticket links.

The paper studies lightweight ConvLSTM-based abnormal human-activity recognition
from surveillance video. Earlier width experiments made `64-32-16` a useful
candidate, but not a proven optimum. The revised study will compare the audited
source-paper topology with narrow stacked ConvLSTM models before selecting a
reference architecture for ablation.

## Evidence plan

1. Run 075's validation-only search: 32 architectures at depths 1-5,
   8 frames and 32/64px (64 base runs), with a 256-epoch cap. Confirm the top
   five plus declared controls at seed 2026; rank only that shortlist across
   both sizes/seeds and advance three. Keep WD evidence separate.
2. Validate FPS at 16f/64px, then LR/WD; retain the best verified eligible
   combination. Train the three customs, input-adapted paper model, two 3D CNNs
   and two transformers at that common input/recipe, up to 512 epochs each.
   Full-step memory checks freeze common batches; the whole workflow is bounded
   at 142 trainings before reuse. Main reports use unaveraged per-class metrics.
3. Keep one existing split (seed 42); the final comparison uses one
   training seed, 42. Freeze all validation-selected checkpoints and the best
   custom model before eight full test evaluations. Run All passes verified
   paths automatically. The owner confirmed different recordings behind similar
   filenames; this does not establish subject/scene independence.
4. Review the verified ZIP evidence before updating manuscript claims.
   Two search seeds give limited uncertainty evidence, not global optimality or
   unseen-data generalisation. The legacy 068 search remains available unchanged.

See the [October 10 interim analysis](implementation/reports/todays.results.md)
for grouped results, a representative main-paper table and the complete available
search appendix. These are console-derived validation results, not a final test report.
The latest appended review includes two-seed confirmation and WD/temporal tables;
it informs 075's candidate refinement without changing the 64-base-run budget.

## Next tasks

- [x] [075 wider/deeper capacity search](../../agents/work/075-wide-depth-capacity-search/prompt.md) - [Locally verified](../../agents/work/075-wide-depth-capacity-search/completion.md): 32 architectures, depths 1-5; 64 base runs; 256-epoch search cap; 16f/64px final input, selected FPS/recipe, adapted baseline and per-class metrics. Up to 142 trainings overall; real Colab execution pending.
- [x] [074 validated comparison handoff](../../agents/work/074-validated-comparison-handoff/prompt.md) - recipe validation, cache reuse and stable experiment numbering; [70 local tests passed](../../agents/work/074-validated-comparison-handoff/completion.md). Real Colab execution pending.
- [x] [073 focused shape search and Run All](../../agents/work/073-focused-shape-search-run-all/prompt.md) - consolidates 070–072; [locally verified](../../agents/work/073-focused-shape-search-run-all/completion.md), with 128-epoch search, 512-epoch comparison, exact ranking and automatic guarded export. Real Colab execution remains pending.
- [x] [069 staged capacity search](../../agents/work/069-staged-capacity-search/prompt.md) - implemented and locally verified; see [completion](../../agents/work/069-staged-capacity-search/completion.md). Current 068 runs and final-comparison protocol stay intact; actual AAD/A100 runs are separate work.
- [x] [068 multi-resolution search and top-three comparison](../../agents/work/068-multisize-search-top3-comparison/prompt.md) - implementation verified; [36-run Colab search reviewed](implementation/reports/2026-10-09-multiresolution-search-review.md), comparison/test still pending. Results are clip-level validation evidence, not established source-independent generalization.
- [x] [057 config-driven datasets and classes](../../agents/work/057-configurable-datasets-and-classes/prompt.md)
- [x] [058 separate custom search and model-family comparison](../../agents/work/058-separate-search-and-comparison/prompt.md) — two AAD profiles and one local smoke profile on the same runner.
- [x] [059 retire VDD-only workflow code](../../agents/work/059-retire-vdd-specific-workflow/prompt.md) — VDD remains optional through generic dataset configuration; old runs are preserved.
- [x] [060 align documentation](../../agents/work/060-align-dataset-documentation/prompt.md)
- [x] [061 thin Colab runner and artifact download](../../agents/work/061-colab-script-runner-and-artifacts/prompt.md) — one-cell notebook/export runs the two active JSON profiles and downloads run artifacts.
- [ ] [062 update manuscript after verified AAD results](../../agents/work/062-update-manuscript-after-aad-results/prompt.md) — do not execute before the new study is complete.
- [x] [063 prepare PRIS 2026 flash presentation](../../agents/work/063-pris-2026-flash-presentation/prompt.md) — current three-slide deck and speaker notes are ready. Revisit after AAD results to add a verified results visual and useful model/example images.

After final results, prepare a small Gradio demo for presentation.

## Tasks

- [x] [`049-aad-dataset-auto-resolution`](../../agents/work/049-aad-dataset-auto-resolution/prompt.md) — shared local-first AAD resolver and optional Kaggle download for modular runners and notebook selection.
- [x] [`047-larger-colab-training`](../../agents/work/047-larger-colab-training/prompt.md) — larger, eight-hour-bounded notebook-03 profile verified locally; real Colab run pending and separate from AAD experiments.
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
- [x] `035-vdd-learnability-check` — retired by ticket 059; VDD is optional and
  uses the generic dataset pipeline, not a VDD-specific diagnostic.
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
- [ ] `026-train-selected-model` — blocked on 053's current validation evidence; confirm and freeze the selected architecture,
  input size, augmentation, and regularisation across both planned seeds.
- [ ] `027-evaluate-selected-model` — blocked until 026 freezes the checkpoint; perform the one-time final AAD test and
  export complete metrics, confusion output, and playable prediction examples.
- [ ] `028-validate-second-dataset` — optional future VDD generalisation study
  after AAD model selection; not part of the active experiment sequence.
- [ ] `029-aggregate-ablation-evidence` — tracked by 053/054; produce paper-ready tables, figures,
  uncertainty, efficiency comparisons, and error-analysis inputs.
- [ ] `030-rewrite-experimental-results` — tracked by 054; revise the experiment and discussion
  section using only verified outputs.
- [ ] `031-align-paper-claims` — tracked by 054; align the abstract, contributions, methods,
  limitations, and conclusion with the final evidence.
- [ ] `032-assess-kinetics-transfer` — conditional only if frozen AAD/VDD
  evidence remains weak; define the exact Kinetics variant and use it for
  pretraining/transfer rather than silently replacing the 11-class task.
- [ ] `033-add-stateful-streaming-inference` — later add chunk-by-chunk stateful
  prediction to the custom model without changing the published baseline.

## Planned results-section flow

1. Experimental Setup
2. Comparison and Ablation Protocol
3. Main Comparative Results (AAD)
4. Ablation Studies
5. Error Analysis
6. Efficiency and Deployment Trade-offs
7. Discussion and Limitations

VDD may be used for a separate, explicitly configured generalisation study;
its historical runs are preserved but are not new comparable evidence.
